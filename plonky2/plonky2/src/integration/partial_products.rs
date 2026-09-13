use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use boojum_cuda::ntt::batch_ntt_in_place;
use boojum_cuda::wires::{
    matrix_trans_gpu, mul_part_product_gpu, scan_part_gpu, scan_part_product_gpu,
    wires_permutation_partial_products_trans_ptr,
};
use cudart::memory::DeviceAllocation;
use cudart::slice::{CudaSlice, CudaSliceMut};
use cudart::stream::CudaStream;

use super::layout::{PolyLayout, PolySegment};

#[derive(Copy, Clone)]
pub(crate) struct PartialProductConfig {
    quotient_degree_factor: usize,
    degree: usize,
    degree_bits: usize,
    num_routed_wires: usize,
    num_challenges: usize,
    num_constants: usize,
}

impl PartialProductConfig {
    pub(crate) fn new(
        quotient_degree_factor: usize,
        degree: usize,
        degree_bits: usize,
        num_routed_wires: usize,
        num_challenges: usize,
        num_constants: usize,
    ) -> Self {
        Self {
            quotient_degree_factor,
            degree,
            degree_bits,
            num_routed_wires,
            num_challenges,
            num_constants,
        }
    }
}

pub(crate) struct PartialProductScratch {
    qv: DeviceAllocation<GoldilocksFieldBoojum>,
    part: DeviceAllocation<GoldilocksFieldBoojum>,
    pp: DeviceAllocation<GoldilocksFieldBoojum>,
}

impl PartialProductScratch {
    fn new(length: usize) -> Self {
        Self {
            qv: DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap(),
            part: DeviceAllocation::<GoldilocksFieldBoojum>::alloc((length + 127) / 128).unwrap(),
            pp: DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap(),
        }
    }
}

pub(crate) struct PartialProductState {
    config: PartialProductConfig,
    betas: DeviceAllocation<GoldilocksFieldBoojum>,
    gammas: DeviceAllocation<GoldilocksFieldBoojum>,
    scratch: PartialProductScratch,
}

impl PartialProductState {
    pub(crate) fn new(config: PartialProductConfig, num_challenges: usize, length: usize) -> Self {
        let scratch = PartialProductScratch::new(length);
        Self {
            config,
            betas: DeviceAllocation::<GoldilocksFieldBoojum>::alloc(num_challenges).unwrap(),
            gammas: DeviceAllocation::<GoldilocksFieldBoojum>::alloc(num_challenges).unwrap(),
            scratch,
        }
    }

    #[inline]
    pub(crate) fn betas(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.betas
    }

    #[inline]
    pub(crate) fn betas_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.betas
    }

    #[inline]
    pub(crate) fn gammas(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.gammas
    }

    #[inline]
    pub(crate) fn gammas_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.gammas
    }
}

pub(crate) struct PartialProductsService;

impl PartialProductsService {
    pub(crate) fn zp(
        state: &mut PartialProductState,
        layout: &PolyLayout,
        fp_inputs: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        subgroup: &DeviceAllocation<GoldilocksFieldBoojum>,
        k_is: &DeviceAllocation<GoldilocksFieldBoojum>,
    ) {
        let stream = CudaStream::default();
        let degree = state.config.quotient_degree_factor;
        let degree2 = state.config.degree;

        let block_x = degree2 as u32;
        let block_y = state.config.num_challenges as u32;
        let thread_x = (state.config.num_routed_wires / degree) as u32;

        stream.synchronize().unwrap();

        let num_routed_wires = state.config.num_routed_wires;
        let degree_bits = state.config.degree_bits;
        let offset = state.config.num_constants * degree2;
        let num_ntts = 2 * num_routed_wires;
        batch_ntt_in_place(
            fp_inputs,
            degree_bits as u32,
            num_ntts as u32,
            offset as u32,
            degree2 as u32,
            false,
            false,
            0,
            0,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        wires_permutation_partial_products_trans_ptr(
            fp_inputs
                .as_ptr()
                .wrapping_add(layout.fp_offset(PolySegment::Wires)),
            subgroup.as_ptr(),
            k_is.as_ptr(),
            fp_inputs.as_ptr().wrapping_add(offset),
            state.scratch.qv.as_mut_ptr(),
            degree,
            degree2,
            state.betas.as_ptr(),
            state.gammas.as_ptr(),
            block_x,
            block_y,
            thread_x,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        batch_ntt_in_place(
            fp_inputs,
            degree_bits as u32,
            num_ntts as u32,
            offset as u32,
            degree2 as u32,
            false,
            true,
            0,
            0,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();

        stream.synchronize().unwrap();

        let part_size = 128;
        let part_len = state.config.num_routed_wires * degree2 / degree;
        let part_num = (part_len + part_size - 1) / part_size;
        let length = state.config.num_challenges * part_len;

        let block_x = part_num as u32;
        let block_y = state.config.num_challenges as u32;
        let thread_x = part_size as u32;

        scan_part_gpu(
            &state.scratch.qv,
            &mut state.scratch.part,
            &mut state.scratch.pp,
            length,
            block_x,
            block_y,
            thread_x,
            &stream,
        )
        .unwrap();

        stream.synchronize().unwrap();

        scan_part_product_gpu(&mut state.scratch.part, part_num, 1, 1, 1, &stream).unwrap();

        stream.synchronize().unwrap();

        mul_part_product_gpu(
            &state.scratch.part,
            &mut state.scratch.pp,
            length,
            block_x,
            block_y,
            thread_x,
            &stream,
        )
        .unwrap();

        stream.synchronize().unwrap();

        let block_x = degree2 as u32;
        let block_y = state.config.num_challenges as u32;
        let thread_x = (state.config.num_routed_wires / degree) as u32;

        matrix_trans_gpu(
            &state.scratch.pp,
            fp_inputs,
            layout.fp_offset(PolySegment::PartialProducts),
            block_x,
            block_y,
            thread_x,
            &stream,
        )
        .unwrap();

        stream.synchronize().unwrap();
    }
}
