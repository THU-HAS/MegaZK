//! GPU final-polynomial state and reduction pipeline.
//!
//! This module owns the final-poly challenge and scratch buffers. It does not access the
//! transcript, and its service does not depend on the compatibility facade.

use std::time::Instant;

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use boojum_cuda::integration::{batch_pad_coset_ext, ntt_extension_inplace};
use cudart::cuda_kernel;
use cudart::error::get_last_error;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::memory::DeviceAllocation;
use cudart::result::{CudaResult, CudaResultWrap};
use cudart::stream::CudaStream;
use plonky2_field::types::Field;

use crate::util::debug_dump;

cuda_kernel!(Reduce, reduce(
    inputs: *const GoldilocksFieldBoojum,
    alpha: *const GoldilocksFieldBoojum,
    outputs: *mut GoldilocksFieldBoojum,
    degree_bits: u32,
    num_polys: u32,
    offset: u32,
));

cuda_kernel!(Divide, fp_divide(
    inputs: *const GoldilocksFieldBoojum,
    point: *const GoldilocksFieldBoojum,
    outputs: *mut GoldilocksFieldBoojum,
    degree_bits: u32,
    order: u32,
));

cuda_kernel!(ShiftAdd, fp_shift_add(
    inputs: *const GoldilocksFieldBoojum,
    alpha: *const GoldilocksFieldBoojum,
    outputs: *mut GoldilocksFieldBoojum,
    degree_bits: u32,
    num_polys: u32,
));

pub(crate) struct FinalPolyScratch {
    reduction: DeviceAllocation<GoldilocksFieldBoojum>,
    quotient: DeviceAllocation<GoldilocksFieldBoojum>,
    coefficients: DeviceAllocation<GoldilocksFieldBoojum>,
}

impl FinalPolyScratch {
    pub(crate) fn from_parts(
        reduction: DeviceAllocation<GoldilocksFieldBoojum>,
        quotient: DeviceAllocation<GoldilocksFieldBoojum>,
        coefficients: DeviceAllocation<GoldilocksFieldBoojum>,
    ) -> Self {
        Self {
            reduction,
            quotient,
            coefficients,
        }
    }
}

pub(crate) struct FinalPolyState {
    alpha: DeviceAllocation<GoldilocksFieldBoojum>,
    scratch: FinalPolyScratch,
}

impl FinalPolyState {
    pub(crate) fn from_parts(
        alpha: DeviceAllocation<GoldilocksFieldBoojum>,
        scratch: FinalPolyScratch,
    ) -> Self {
        Self { alpha, scratch }
    }

    #[inline]
    pub(crate) fn alpha(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.alpha
    }

    #[inline]
    pub(crate) fn alpha_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.alpha
    }

    #[inline]
    pub(crate) fn coefficients(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.scratch.coefficients
    }
}

#[derive(Copy, Clone)]
pub(crate) struct FinalPolyConfig {
    degree: usize,
    degree_bits: usize,
    rate_bits: usize,
    num_polys: usize,
    num_next: usize,
    next_start: usize,
}

impl FinalPolyConfig {
    pub(crate) fn new(
        degree: usize,
        degree_bits: usize,
        rate_bits: usize,
        num_polys: usize,
        num_next: usize,
        next_start: usize,
    ) -> Self {
        Self {
            degree,
            degree_bits,
            rate_bits,
            num_polys,
            num_next,
            next_start,
        }
    }
}

pub(crate) struct FinalPolyService;

impl FinalPolyService {
    pub(crate) fn compute(
        state: &mut FinalPolyState,
        zeta_g: &DeviceAllocation<GoldilocksFieldBoojum>,
        fp_inputs: &DeviceAllocation<GoldilocksFieldBoojum>,
        first_fri_leaves: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        config: FinalPolyConfig,
    ) {
        let stream = CudaStream::default();
        let s = Instant::now();
        Self::reduce(state, fp_inputs, config, config.num_polys, 0).unwrap();
        println!(" reduce 0: {:?}", s.elapsed());
        debug_dump::write_json("fp0.json", || -> anyhow::Result<Vec<u8>> {
            let values = debug_dump::capture_cuda(&state.scratch.reduction, &stream)?;
            Ok(serde_json::to_vec(&values)?)
        });

        let s = Instant::now();
        Self::divide_shift_add(state, zeta_g, config, 0, config.num_polys).unwrap();
        println!(" divide 0: {:?}", s.elapsed());

        let s = Instant::now();
        Self::reduce(state, fp_inputs, config, config.num_next, config.next_start).unwrap();
        println!(" reduce 1: {:?}", s.elapsed());
        debug_dump::write_json("fp1.json", || -> anyhow::Result<Vec<u8>> {
            let values = debug_dump::capture_cuda(&state.scratch.reduction, &stream)?;
            Ok(serde_json::to_vec(&values)?)
        });

        let s = Instant::now();
        Self::divide_shift_add(state, zeta_g, config, 1, config.num_next).unwrap();
        println!(" divide 1: {:?}", s.elapsed());

        let s = Instant::now();
        Self::coset_fft(state, first_fri_leaves, config).unwrap();
        println!(" coset ntt: {:?}", s.elapsed());
    }

    fn reduce(
        state: &mut FinalPolyState,
        fp_inputs: &DeviceAllocation<GoldilocksFieldBoojum>,
        config: FinalPolyConfig,
        num_polys: usize,
        offset: usize,
    ) -> CudaResult<()> {
        let threads: u32 = 32;
        let blocks = (config.degree as u32 + threads - 1) / threads;
        let stream = CudaStream::default();
        let launch = CudaLaunchConfig::basic(blocks, threads, &stream);
        let args = ReduceArguments::new(
            fp_inputs.as_ptr(),
            state.alpha.as_ptr(),
            state.scratch.reduction.as_mut_ptr(),
            config.degree_bits as u32,
            num_polys as u32,
            offset as u32,
        );
        ReduceFunction(reduce).launch(&launch, &args)?;
        stream.synchronize().unwrap();
        get_last_error().wrap()
    }

    fn divide_shift_add(
        state: &mut FinalPolyState,
        zeta_g: &DeviceAllocation<GoldilocksFieldBoojum>,
        config: FinalPolyConfig,
        order: usize,
        num_polys: usize,
    ) -> CudaResult<()> {
        let stream = CudaStream::default();
        let threads: u32 = 128;
        let launch = CudaLaunchConfig::basic(1, threads, &stream);
        let args = DivideArguments::new(
            state.scratch.reduction.as_ptr(),
            zeta_g.as_ptr(),
            state.scratch.quotient.as_mut_ptr(),
            config.degree_bits as u32,
            order as u32,
        );
        DivideFunction(fp_divide).launch(&launch, &args)?;
        stream.synchronize().unwrap();
        debug_dump::write_json("fp_q1.json", || -> anyhow::Result<Vec<u8>> {
            let quotient = debug_dump::capture_cuda(&state.scratch.quotient, &stream)?;
            Ok(serde_json::to_vec(&quotient)?)
        });

        let threads: u32 = 32;
        let blocks = (config.degree as u32 + (threads * 128 - 1)) / (threads * 128);
        let launch = CudaLaunchConfig::basic(blocks, threads, &stream);
        let args = ShiftAddArguments::new(
            state.scratch.quotient.as_ptr().wrapping_add(2),
            state.alpha.as_ptr(),
            state.scratch.coefficients.as_mut_ptr(),
            config.degree_bits as u32,
            num_polys as u32,
        );
        ShiftAddFunction(fp_shift_add).launch(&launch, &args)?;
        stream.synchronize().unwrap();
        get_last_error().wrap()
    }

    pub(crate) fn coset_fft(
        state: &mut FinalPolyState,
        first_fri_leaves: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        config: FinalPolyConfig,
    ) -> CudaResult<()> {
        let stream = CudaStream::default();
        batch_pad_coset_ext(
            state.scratch.coefficients.as_ptr(),
            first_fri_leaves.as_mut_ptr(),
            GoldilocksFieldBoojum::MULTIPLICATIVE_GROUP_GENERATOR,
            config.degree_bits as u32,
            1,
            config.degree as u32,
            config.degree as u32,
            false,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        ntt_extension_inplace(
            first_fri_leaves,
            (config.degree_bits + config.rate_bits) as u32,
            1,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        debug_dump::write_json("fp_value.json", || -> anyhow::Result<Vec<u8>> {
            let values = debug_dump::capture_cuda(first_fri_leaves, &stream)?;
            Ok(serde_json::to_vec(&values)?)
        });
        get_last_error().wrap()
    }
}
