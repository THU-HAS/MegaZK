//! GPU polynomial-opening state and kernels.
//!
//! This module owns the opening point and evaluation buffers. It does not sample or observe the
//! transcript, and its service does not depend on the compatibility facade.

use std::mem;
use std::time::Instant;

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::cuda_kernel;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::memory::{memory_copy_async, DeviceAllocation, HostAllocation};
use cudart::stream::CudaStream;

use crate::field::extension::{Extendable, FieldExtension};
use crate::field::types::Field;
use crate::hash::hash_types::RichField;
use crate::util::debug_dump;

cuda_kernel!(OpenNew, open_polys_new(
    polys: *const GoldilocksFieldBoojum,
    zeta_g: *const GoldilocksFieldBoojum,
    buffer: *mut GoldilocksFieldBoojum,
    degree_bits: u32,
    num_polys: u32,
    num_next: u32,
    next_start: u32,
));

cuda_kernel!(OpenReduce, open_polys_reduce(
    buffer: *const GoldilocksFieldBoojum,
    zeta_g: *const GoldilocksFieldBoojum,
    openings: *mut GoldilocksFieldBoojum,
    degree_bits: u32,
    num_polys: u32,
    num_next: u32,
    next_start: u32,
));

pub(crate) struct OpeningBuffers {
    reduction: DeviceAllocation<GoldilocksFieldBoojum>,
    device: DeviceAllocation<GoldilocksFieldBoojum>,
}

impl OpeningBuffers {
    pub(crate) fn from_parts(
        reduction: DeviceAllocation<GoldilocksFieldBoojum>,
        device: DeviceAllocation<GoldilocksFieldBoojum>,
    ) -> Self {
        Self { reduction, device }
    }
}

pub(crate) struct OpeningState {
    zeta_g: DeviceAllocation<GoldilocksFieldBoojum>,
    buffers: OpeningBuffers,
}

impl OpeningState {
    pub(crate) fn from_parts(
        zeta_g: DeviceAllocation<GoldilocksFieldBoojum>,
        buffers: OpeningBuffers,
    ) -> Self {
        Self { zeta_g, buffers }
    }

    #[inline]
    pub(crate) fn zeta_g(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.zeta_g
    }

    #[inline]
    pub(crate) fn zeta_g_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.zeta_g
    }

    #[inline]
    pub(crate) fn values(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.buffers.device
    }

    #[inline]
    pub(crate) fn values_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.buffers.device
    }
}

#[derive(Copy, Clone)]
pub(crate) struct OpeningConfig {
    degree_bits: usize,
    num_polys: usize,
    num_next: usize,
    next_start: usize,
}

impl OpeningConfig {
    pub(crate) fn new(
        degree_bits: usize,
        num_polys: usize,
        num_next: usize,
        next_start: usize,
    ) -> Self {
        Self {
            degree_bits,
            num_polys,
            num_next,
            next_start,
        }
    }
}

pub(crate) struct OpeningsService;

impl OpeningsService {
    pub(crate) fn initialize<F: RichField + Extendable<D>, const D: usize>(
        state: &mut OpeningState,
        degree_bits: usize,
        stream: &CudaStream,
    ) -> Vec<GoldilocksFieldBoojum> {
        let g = F::Extension::primitive_root_of_unity(degree_bits);
        let zero = F::Extension::ZERO;
        let zeta_g: Vec<GoldilocksFieldBoojum> = unsafe {
            mem::transmute(vec![zero.to_basefield_array(), g.to_basefield_array()].concat())
        };
        memory_copy_async(&mut state.zeta_g, &zeta_g, stream).unwrap();
        zeta_g
    }

    pub(crate) fn compute(
        state: &mut OpeningState,
        fp_inputs: &DeviceAllocation<GoldilocksFieldBoojum>,
        output: &mut HostAllocation<GoldilocksFieldBoojum>,
        config: OpeningConfig,
    ) {
        let stream = CudaStream::default();
        let s = Instant::now();
        let threads: u32 = 64;
        let blocks = (config.num_polys + config.num_next) as u32;
        let launch = CudaLaunchConfig::basic(blocks, threads, &stream);
        let args = OpenNewArguments {
            polys: fp_inputs.as_ptr(),
            zeta_g: state.zeta_g.as_ptr(),
            buffer: state.buffers.reduction.as_mut_ptr(),
            degree_bits: config.degree_bits as u32,
            num_polys: config.num_polys as u32,
            num_next: config.num_next as u32,
            next_start: config.next_start as u32,
        };
        OpenNewFunction::default().launch(&launch, &args).unwrap();
        stream.synchronize().unwrap();
        println!(" eval segs: {:?}", s.elapsed());

        let s = Instant::now();
        let threads: u32 = 32;
        let blocks = ((config.num_polys + config.num_next) as u32 + threads - 1) / threads;
        let launch = CudaLaunchConfig::basic(blocks, threads, &stream);
        let args = OpenReduceArguments {
            buffer: state.buffers.reduction.as_ptr(),
            zeta_g: state.zeta_g.as_ptr(),
            openings: state.buffers.device.as_mut_ptr(),
            degree_bits: config.degree_bits as u32,
            num_polys: config.num_polys as u32,
            num_next: config.num_next as u32,
            next_start: config.next_start as u32,
        };
        OpenReduceFunction::default()
            .launch(&launch, &args)
            .unwrap();
        stream.synchronize().unwrap();
        println!(" reduce: {:?}", s.elapsed());
        debug_dump::write_json("openings.json", || -> anyhow::Result<Vec<u8>> {
            let openings = debug_dump::capture_cuda(&state.buffers.device, &stream)?;
            Ok(serde_json::to_vec(&openings)?)
        });
        stream.synchronize().unwrap();
        memory_copy_async(output, &state.buffers.device, &stream).unwrap();
    }
}
