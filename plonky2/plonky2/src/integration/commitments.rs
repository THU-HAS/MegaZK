use std::mem;
use std::time::Instant;

use anyhow::Result;
use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use boojum_cuda::integration::batch_pad_coset;
use boojum_cuda::merkle_tree_segmented::{
    build_merkle_tree_in_place, build_merkle_tree_w_partial_ldes,
};
use boojum_cuda::ntt::{batch_lde_transpose_out_of_place_ptr, batch_ntt_in_place};
use boojum_cuda::poseidon::Poseidon;
use cudart::cuda_kernel;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use cudart::slice::{CudaSlice, CudaSliceMut};
use cudart::stream::CudaStream;

use super::layout::{PolyLayout, PolySegment};
use super::overlay::{DeviceView, ScratchCursor};
use super::transcript::TranscriptStep;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleCap;
use crate::plonk::config::Hasher;
use crate::util::log2_strict;

cuda_kernel!(MemcpyOffset, copy_w_offset(
    tree: *const GoldilocksFieldBoojum,
    cap: *mut GoldilocksFieldBoojum,
    offset: u32,
    num: u32,
));

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum CommitmentStage {
    Constants,
    Wires,
    PartialProducts,
    Quotients,
}

impl CommitmentStage {
    pub(crate) const ALL: [Self; 4] = [
        Self::Constants,
        Self::Wires,
        Self::PartialProducts,
        Self::Quotients,
    ];

    pub(crate) fn from_legacy_index(index: usize) -> Self {
        match index {
            0 => Self::Constants,
            1 => Self::Wires,
            2 => Self::PartialProducts,
            3 => Self::Quotients,
            4_usize.. => todo!(),
        }
    }
}

pub(crate) struct MerkleCapBuffers {
    tree_device: DeviceView<GoldilocksFieldBoojum>,
    tree_host: HostAllocation<GoldilocksFieldBoojum>,
    /// Present only for cm0. Delayed trees alias workspace scratch instead.
    #[allow(dead_code)]
    device_owner: Option<DeviceAllocation<GoldilocksFieldBoojum>>,
}

impl MerkleCapBuffers {
    fn new(tree_len: usize) -> Self {
        let mut device_owner = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(tree_len).unwrap();
        let tree_device =
            unsafe { DeviceView::from_raw(device_owner.as_mut_ptr(), device_owner.len()) };
        Self {
            tree_device,
            tree_host: HostAllocation::<GoldilocksFieldBoojum>::alloc(
                tree_len,
                CudaHostAllocFlags::DEFAULT,
            )
            .unwrap(),
            device_owner: Some(device_owner),
        }
    }

    fn unallocated() -> Self {
        Self {
            tree_device: DeviceView::empty(),
            tree_host: HostAllocation::<GoldilocksFieldBoojum>::alloc(
                0,
                CudaHostAllocFlags::DEFAULT,
            )
            .unwrap(),
            device_owner: None,
        }
    }

    fn bind_from_scratch(&mut self, cursor: &mut ScratchCursor, tree_len: usize) {
        assert!(
            self.device_owner.is_none(),
            "owned Merkle tree cannot alias scratch"
        );
        self.tree_device = cursor.take(tree_len);
    }
}

pub(crate) struct CommitmentBuffers {
    tree_len: usize,
    constants: MerkleCapBuffers,
    wires: MerkleCapBuffers,
    partial_products: MerkleCapBuffers,
    quotients: MerkleCapBuffers,
}

impl CommitmentBuffers {
    pub(crate) fn new(tree_len: usize) -> Self {
        Self {
            tree_len,
            constants: MerkleCapBuffers::new(tree_len),
            wires: MerkleCapBuffers::unallocated(),
            partial_products: MerkleCapBuffers::unallocated(),
            quotients: MerkleCapBuffers::unallocated(),
        }
    }

    pub(crate) fn ensure_stage(&mut self, stage: CommitmentStage) {
        if stage == CommitmentStage::Constants {
            return;
        }
        let tree_len = self.tree_len;
        assert_eq!(
            self.stage(stage).tree_device.len(),
            tree_len,
            "delayed {:?} Merkle tree must be bound from scratch after witgen",
            stage
        );
    }

    pub(crate) fn bind_delayed_from_scratch(&mut self, cursor: &mut ScratchCursor) {
        let tree_len = self.tree_len;
        for stage in [
            CommitmentStage::Wires,
            CommitmentStage::PartialProducts,
            CommitmentStage::Quotients,
        ] {
            self.stage_mut(stage).bind_from_scratch(cursor, tree_len);
        }
    }

    fn stage(&self, stage: CommitmentStage) -> &MerkleCapBuffers {
        match stage {
            CommitmentStage::Constants => &self.constants,
            CommitmentStage::Wires => &self.wires,
            CommitmentStage::PartialProducts => &self.partial_products,
            CommitmentStage::Quotients => &self.quotients,
        }
    }

    fn stage_mut(&mut self, stage: CommitmentStage) -> &mut MerkleCapBuffers {
        match stage {
            CommitmentStage::Constants => &mut self.constants,
            CommitmentStage::Wires => &mut self.wires,
            CommitmentStage::PartialProducts => &mut self.partial_products,
            CommitmentStage::Quotients => &mut self.quotients,
        }
    }

    #[inline]
    pub(crate) fn tree_device(
        &self,
        stage: CommitmentStage,
    ) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &*self.stage(stage).tree_device
    }

    #[inline]
    pub(crate) fn tree_device_mut(
        &mut self,
        stage: CommitmentStage,
    ) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut *self.stage_mut(stage).tree_device
    }

    #[inline]
    pub(crate) fn tree_device_ptrs(&self) -> [*const GoldilocksFieldBoojum; 4] {
        CommitmentStage::ALL.map(|stage| self.tree_device(stage).as_ptr())
    }
}

#[derive(Copy, Clone)]
pub(crate) struct CommitmentConfig {
    degree_bits: usize,
    degree: usize,
    rate_bits: usize,
    quotient_degree_factor: usize,
    num_challenges: usize,
}

impl CommitmentConfig {
    pub(crate) fn new(
        degree_bits: usize,
        degree: usize,
        rate_bits: usize,
        quotient_degree_factor: usize,
        num_challenges: usize,
    ) -> Self {
        Self {
            degree_bits,
            degree,
            rate_bits,
            quotient_degree_factor,
            num_challenges,
        }
    }
}

pub(crate) struct CommitmentService;

impl CommitmentService {
    pub(crate) fn commit_constants(
        buffers: &mut CommitmentBuffers,
        layout: &PolyLayout,
        fp_inputs: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        ldes: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        config: CommitmentConfig,
    ) {
        let s = Instant::now();
        let stream1 = CudaStream::default();
        {
            batch_ntt_in_place(
                fp_inputs,
                config.degree_bits as u32,
                layout.polynomial_count(PolySegment::ConstantsSigmas) as u32,
                0,
                config.degree as u32,
                false,
                true,
                0,
                0,
                &stream1,
            )
            .unwrap();
        }
        stream1.synchronize().unwrap();
        println!(" intt: {:?}", s.elapsed());

        let s = Instant::now();
        let stream2 = CudaStream::default();
        {
            if layout.num_ldes(PolySegment::ConstantsSigmas) > 0 {
                batch_lde_transpose_out_of_place_ptr(
                    fp_inputs.as_ptr(),
                    ldes.as_mut_ptr(),
                    config.degree_bits as u32,
                    config.rate_bits as u32,
                    layout.num_ldes(PolySegment::ConstantsSigmas) as u32,
                    config.degree as u32,
                    (config.degree << config.rate_bits) as u32,
                    false,
                    &stream2,
                )
                .unwrap();
            }
        }
        stream2.synchronize().unwrap();
        println!(" ntt: {:?}", s.elapsed());

        let s = Instant::now();
        let stream3 = CudaStream::default();
        {
            let tree = buffers.stage_mut(CommitmentStage::Constants);
            build_merkle_tree_w_partial_ldes::<Poseidon>(
                ldes,
                fp_inputs,
                &mut *tree.tree_device,
                config.degree_bits as u32,
                config.rate_bits as u32,
                layout.polynomial_count(PolySegment::ConstantsSigmas) as u32,
                layout.num_ldes(PolySegment::ConstantsSigmas) as u32,
                layout.lde_offset(PolySegment::ConstantsSigmas),
                layout.fp_offset(PolySegment::ConstantsSigmas),
                &stream3,
                (config.degree_bits + config.rate_bits + 1) as u32,
            )
            .unwrap();
        }
        stream3.synchronize().unwrap();
        {
            let tree = buffers.stage_mut(CommitmentStage::Constants);
            memory_copy_async(&mut tree.tree_host, &tree.tree_device, &stream3).unwrap();
        }
        println!(" poseidon: {:?}", s.elapsed());

        let s = Instant::now();
        stream1.synchronize().unwrap();
        stream2.synchronize().unwrap();
        stream3.synchronize().unwrap();
        println!(" transfer data: {:?}", s.elapsed());
    }

    pub(crate) fn commit_wires(
        buffers: &mut CommitmentBuffers,
        layout: &PolyLayout,
        fp_inputs: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        ldes: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        config: CommitmentConfig,
    ) {
        let s = Instant::now();
        let stream1 = CudaStream::default();
        {
            batch_ntt_in_place(
                fp_inputs,
                config.degree_bits as u32,
                layout.polynomial_count(PolySegment::Wires) as u32,
                layout.fp_offset(PolySegment::Wires) as u32,
                config.degree as u32,
                false,
                true,
                0,
                0,
                &stream1,
            )
            .unwrap();
        }
        stream1.synchronize().unwrap();
        println!(" intt: {:?}", s.elapsed());

        let s = Instant::now();
        let stream2 = CudaStream::default();
        {
            if layout.num_ldes(PolySegment::Wires) > 0 {
                batch_lde_transpose_out_of_place_ptr(
                    fp_inputs
                        .as_ptr()
                        .wrapping_add(layout.fp_offset(PolySegment::Wires)),
                    ldes.as_mut_ptr()
                        .wrapping_add(layout.lde_offset(PolySegment::Wires)),
                    config.degree_bits as u32,
                    config.rate_bits as u32,
                    layout.num_ldes(PolySegment::Wires) as u32,
                    config.degree as u32,
                    (config.degree << config.rate_bits) as u32,
                    false,
                    &stream2,
                )
                .unwrap();
            }
        }
        stream2.synchronize().unwrap();
        println!(" ntt: {:?}", s.elapsed());

        let s = Instant::now();
        let stream3 = CudaStream::default();
        {
            build_merkle_tree_w_partial_ldes::<Poseidon>(
                ldes,
                fp_inputs,
                &mut *buffers.stage_mut(CommitmentStage::Wires).tree_device,
                config.degree_bits as u32,
                config.rate_bits as u32,
                layout.polynomial_count(PolySegment::Wires) as u32,
                layout.num_ldes(PolySegment::Wires) as u32,
                layout.lde_offset(PolySegment::Wires),
                layout.fp_offset(PolySegment::Wires),
                &stream3,
                (config.degree_bits + config.rate_bits + 1) as u32,
            )
            .unwrap();
        }
        stream3.synchronize().unwrap();
        println!(" poseidon: {:?}", s.elapsed());
    }

    pub(crate) fn commit_partial_products(
        buffers: &mut CommitmentBuffers,
        layout: &PolyLayout,
        fp_inputs: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        ldes: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        config: CommitmentConfig,
    ) {
        let s = Instant::now();
        let stream1 = CudaStream::default();
        {
            batch_ntt_in_place(
                fp_inputs,
                config.degree_bits as u32,
                layout.polynomial_count(PolySegment::PartialProducts) as u32,
                layout.fp_offset(PolySegment::PartialProducts) as u32,
                config.degree as u32,
                false,
                true,
                0,
                0,
                &stream1,
            )
            .unwrap();
        }
        stream1.synchronize().unwrap();
        println!(" intt: {:?}", s.elapsed());

        let s = Instant::now();
        let stream2 = CudaStream::default();
        {
            if layout.num_ldes(PolySegment::PartialProducts) > 0 {
                batch_lde_transpose_out_of_place_ptr(
                    fp_inputs
                        .as_ptr()
                        .wrapping_add(layout.fp_offset(PolySegment::PartialProducts)),
                    ldes.as_mut_ptr()
                        .wrapping_add(layout.lde_offset(PolySegment::PartialProducts)),
                    config.degree_bits as u32,
                    config.rate_bits as u32,
                    layout.num_ldes(PolySegment::PartialProducts) as u32,
                    config.degree as u32,
                    (config.degree << config.rate_bits) as u32,
                    false,
                    &stream2,
                )
                .unwrap();
            }
        }
        stream2.synchronize().unwrap();
        println!(" ntt: {:?}", s.elapsed());

        let s = Instant::now();
        let stream3 = CudaStream::default();
        {
            build_merkle_tree_in_place::<Poseidon>(
                fp_inputs,
                &mut *buffers
                    .stage_mut(CommitmentStage::PartialProducts)
                    .tree_device,
                config.degree_bits as u32,
                config.rate_bits as u32,
                layout.polynomial_count(PolySegment::PartialProducts) as u32,
                layout.fp_offset(PolySegment::PartialProducts),
                &stream3,
                (config.degree_bits + config.rate_bits + 1) as u32,
            )
            .unwrap();
        }
        stream3.synchronize().unwrap();
        println!(" poseidon: {:?}", s.elapsed());
    }

    pub(crate) fn commit_quotients(
        buffers: &mut CommitmentBuffers,
        layout: &PolyLayout,
        fp_inputs: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        ldes: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        config: CommitmentConfig,
    ) {
        let s = Instant::now();
        let stream1 = CudaStream::default();
        {
            batch_ntt_in_place(
                fp_inputs,
                (config.degree_bits + log2_strict(config.quotient_degree_factor)) as u32,
                config.num_challenges as u32,
                layout.fp_offset(PolySegment::Quotients) as u32,
                (config.degree * config.quotient_degree_factor) as u32,
                true,
                true,
                0,
                0,
                &stream1,
            )
            .unwrap();
        }
        stream1.synchronize().unwrap();
        {
            let inputs_matrix_ptr = fp_inputs
                .as_ptr()
                .wrapping_add(layout.fp_offset(PolySegment::Quotients));
            let outputs_matrix_ptr = fp_inputs
                .as_mut_ptr()
                .wrapping_add(layout.fp_offset(PolySegment::Quotients));
            batch_pad_coset(
                inputs_matrix_ptr,
                outputs_matrix_ptr,
                (config.degree_bits + log2_strict(config.quotient_degree_factor)) as u32,
                config.num_challenges as u32,
                (config.degree * config.quotient_degree_factor) as u32,
                (config.degree * config.quotient_degree_factor) as u32,
                0,
                true,
                &stream1,
            )
            .unwrap();
        }
        stream1.synchronize().unwrap();
        println!(" intt: {:?}", s.elapsed());

        let s = Instant::now();
        let stream2 = CudaStream::default();
        {
            if layout.num_ldes(PolySegment::Quotients) > 0 {
                batch_lde_transpose_out_of_place_ptr(
                    fp_inputs
                        .as_ptr()
                        .wrapping_add(layout.fp_offset(PolySegment::Quotients)),
                    ldes.as_mut_ptr()
                        .wrapping_add(layout.lde_offset(PolySegment::Quotients)),
                    config.degree_bits as u32,
                    config.rate_bits as u32,
                    layout.num_ldes(PolySegment::Quotients) as u32,
                    config.degree as u32,
                    (config.degree << config.rate_bits) as u32,
                    false,
                    &stream2,
                )
                .unwrap();
            }
        }
        stream2.synchronize().unwrap();
        println!(" ntt: {:?}", s.elapsed());

        let s = Instant::now();
        let stream3 = CudaStream::default();
        {
            build_merkle_tree_in_place::<Poseidon>(
                fp_inputs,
                &mut *buffers.stage_mut(CommitmentStage::Quotients).tree_device,
                config.degree_bits as u32,
                config.rate_bits as u32,
                layout.polynomial_count(PolySegment::Quotients) as u32,
                layout.fp_offset(PolySegment::Quotients),
                &stream3,
                (config.degree_bits + config.rate_bits + 1) as u32,
            )
            .unwrap();
        }
        stream3.synchronize().unwrap();
        println!(" poseidon: {:?}", s.elapsed());
    }

    pub(crate) fn get_cap<F: RichField, H: Hasher<F>>(
        buffers: &CommitmentBuffers,
        stage: CommitmentStage,
        cap_height: usize,
    ) -> MerkleCap<F, H> {
        Self::assemble_cap(&buffers.stage(stage).tree_host, cap_height)
    }

    pub(crate) fn get_cap_device<F: RichField, H: Hasher<F>>(
        buffers: &CommitmentBuffers,
        stage: CommitmentStage,
        cap_height: usize,
    ) -> MerkleCap<F, H> {
        let tree_device = &buffers.stage(stage).tree_device;
        let stream = CudaStream::default();
        let mut tree_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(
            tree_device.len(),
            CudaHostAllocFlags::DEFAULT,
        )
        .unwrap();
        memory_copy_async(&mut tree_host, tree_device, &stream).unwrap();
        stream.synchronize().unwrap();
        Self::assemble_cap(&tree_host, cap_height)
    }

    pub(crate) fn get_proof_caps<F: RichField, H: Hasher<F>>(
        buffers: &CommitmentBuffers,
        cap_height: usize,
    ) -> Result<Vec<MerkleCap<F, H>>> {
        let stream = CudaStream::default();
        let len_cap = TranscriptStep::cap_len(cap_height);
        let offset = TranscriptStep::cap_offset(
            buffers.tree_device(CommitmentStage::Constants).len(),
            cap_height,
        );
        let mut caps = vec![
            HostAllocation::<GoldilocksFieldBoojum>::alloc(len_cap, CudaHostAllocFlags::DEFAULT)
                .unwrap(),
            HostAllocation::<GoldilocksFieldBoojum>::alloc(len_cap, CudaHostAllocFlags::DEFAULT)
                .unwrap(),
            HostAllocation::<GoldilocksFieldBoojum>::alloc(len_cap, CudaHostAllocFlags::DEFAULT)
                .unwrap(),
        ];
        let config = CudaLaunchConfig::basic(1, len_cap as u32, &stream);
        let proof_stages = [
            CommitmentStage::Wires,
            CommitmentStage::PartialProducts,
            CommitmentStage::Quotients,
        ];
        for (cap, stage) in caps.iter_mut().zip(proof_stages) {
            let args = MemcpyOffsetArguments::new(
                buffers.tree_device(stage).as_ptr(),
                cap.as_mut_ptr(),
                offset as u32,
                len_cap as u32,
            );
            MemcpyOffsetFunction(copy_w_offset).launch(&config, &args)?;
        }
        Ok(caps
            .iter()
            .map(|cap| Self::assemble_cap_from_slice(cap, 0, cap.len() / 4))
            .collect())
    }

    fn assemble_cap<F: RichField, H: Hasher<F>>(
        tree_host: &HostAllocation<GoldilocksFieldBoojum>,
        cap_height: usize,
    ) -> MerkleCap<F, H> {
        let start = TranscriptStep::cap_offset(tree_host.len(), cap_height);
        Self::assemble_cap_from_slice(tree_host, start, 1 << cap_height)
    }

    fn assemble_cap_from_slice<F: RichField, H: Hasher<F>>(
        source: &[GoldilocksFieldBoojum],
        start: usize,
        count: usize,
    ) -> MerkleCap<F, H> {
        let mut caps = Vec::<H::Hash>::new();
        for i in 0..count {
            let index = start + 4 * i;
            let h: Vec<F> = unsafe {
                mem::transmute(vec![
                    source[index],
                    source[index + 1],
                    source[index + 2],
                    source[index + 3],
                ])
            };
            caps.push(H::hash_or_noop(&h));
        }
        MerkleCap(caps)
    }
}
