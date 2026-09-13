//! Active GPU FRI commitment kernels and buffer transfers.
//!
//! Transcript observation and challenge sampling live in `orchestrator.rs`.

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use boojum::field::Field;
use boojum_cuda::integration::{
    batch_pad_coset_ext, batch_reduce_ext, batch_transpose, ntt_extension_inplace,
};
use boojum_cuda::poseidon::{build_merkle_tree, Poseidon};
use cudart::cuda_kernel;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::memory::{memory_copy_async, CudaHostAllocFlags, HostAllocation};
use cudart::stream::CudaStream;

use super::state::FriCommitState;
use crate::util::debug_dump;

cuda_kernel!(MemcpyOffset, copy_w_offset(
    tree: *const GoldilocksFieldBoojum,
    cap: *mut GoldilocksFieldBoojum,
    offset: u32,
    num: u32,
));

pub(crate) struct FriCommitService;

impl FriCommitService {
    pub(crate) fn commit_tree(
        commit: &mut FriCommitState,
        round: usize,
        arity_bits: usize,
        layers_count: usize,
        rate_bits: usize,
        stream: &CudaStream,
    ) {
        let round = commit.round_mut(round);
        batch_transpose(
            round.leaves.as_ptr(),
            round.transposed_leaves.as_mut_ptr(),
            (arity_bits + 1) as u32,
            1 << (layers_count - 1) as u32,
            1 << (layers_count - 1),
            1 << (layers_count - 1),
            rate_bits as u32,
            stream,
        )
        .unwrap();
        build_merkle_tree::<Poseidon>(
            &*round.transposed_leaves,
            &mut *round.digests,
            0,
            stream,
            layers_count as u32,
        )
        .unwrap();
    }

    pub(crate) fn copy_cap(
        commit: &FriCommitState,
        commit_phase_caps: &mut Vec<Vec<GoldilocksFieldBoojum>>,
        round: usize,
        offset: usize,
        cap_len: usize,
        stream: &CudaStream,
    ) {
        let mut cap_host =
            HostAllocation::<GoldilocksFieldBoojum>::alloc(cap_len, CudaHostAllocFlags::DEFAULT)
                .unwrap();
        let config = CudaLaunchConfig::basic(1, cap_len as u32, stream);
        let args = MemcpyOffsetArguments::new(
            commit.tree_digests(round).as_ptr(),
            cap_host.as_mut_ptr(),
            offset as u32,
            cap_len as u32,
        );
        MemcpyOffsetFunction(copy_w_offset)
            .launch(&config, &args)
            .unwrap();
        stream.synchronize().unwrap();
        commit_phase_caps.push(cap_host.to_vec());
    }

    pub(crate) fn print_round_debug(
        commit: &FriCommitState,
        round: usize,
        offset: usize,
        stream: &CudaStream,
    ) {
        let mut tree_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(
            commit.tree_digests(round).len(),
            CudaHostAllocFlags::DEFAULT,
        )
        .unwrap();
        memory_copy_async(&mut tree_host, commit.tree_digests(round), stream).unwrap();
        stream.synchronize().unwrap();
        println!("cap: {:?}", tree_host[offset..(offset + 64)].to_vec());

        let mut beta_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(
            commit.beta().len(),
            CudaHostAllocFlags::DEFAULT,
        )
        .unwrap();
        memory_copy_async(&mut beta_host, commit.beta(), stream).unwrap();
        stream.synchronize().unwrap();
        println!("fri_beta: {:?}", beta_host.to_vec());
    }

    pub(crate) fn fold_round(
        commit: &mut FriCommitState,
        final_poly_coefficients: *const GoldilocksFieldBoojum,
        round: usize,
        degree: usize,
        degree_bits: usize,
        arity: usize,
        rate_bits: usize,
        shift: &mut GoldilocksFieldBoojum,
        is_last_round: bool,
        stream: &CudaStream,
    ) {
        let reduction = commit.reduction_buffers_mut(round);
        let reduce_input = if round == 0 {
            final_poly_coefficients
        } else {
            reduction.previous_coefficients.unwrap().as_ptr()
        };
        batch_reduce_ext(
            reduce_input,
            reduction.beta.as_ptr(),
            reduction.output_coefficients.as_mut_ptr(),
            degree as u32,
            arity as u32,
            stream,
        )
        .unwrap();

        if !is_last_round {
            *shift = shift.pow_u64(arity as u64);
            let (coefficients, next_leaves) = commit.fold_output_mut(round);
            batch_pad_coset_ext(
                coefficients.as_ptr(),
                next_leaves.as_mut_ptr(),
                *shift,
                degree_bits as u32,
                1,
                degree as u32,
                degree as u32,
                false,
                stream,
            )
            .unwrap();
            ntt_extension_inplace(next_leaves, (degree_bits + rate_bits) as u32, 1, stream)
                .unwrap();
        }
    }

    pub(crate) fn dump_first_round(commit: &FriCommitState, stream: &CudaStream) {
        debug_dump::write_json("fri_leaves_buffer1.json", || -> anyhow::Result<Vec<u8>> {
            let source = commit
                .transposed_tree_leaves_at(0)
                .ok_or_else(|| anyhow::anyhow!("FRI leaves buffer 0 is unavailable"))?;
            let values = debug_dump::capture_cuda(source, stream)?;
            Ok(serde_json::to_vec(&values)?)
        });
        debug_dump::write_json("fri_leaves2.json", || -> anyhow::Result<Vec<u8>> {
            let source = commit
                .tree_leaves_at(1)
                .ok_or_else(|| anyhow::anyhow!("FRI leaves layer 1 is unavailable"))?;
            let values = debug_dump::capture_cuda(source, stream)?;
            Ok(serde_json::to_vec(&values)?)
        });
        debug_dump::write_json("fri_tree1.json", || -> anyhow::Result<Vec<u8>> {
            let values = debug_dump::capture_cuda(commit.tree_digests(0), stream)?;
            Ok(serde_json::to_vec(&values)?)
        });
        debug_dump::write_json("fri_beta1.json", || -> anyhow::Result<Vec<u8>> {
            let values = debug_dump::capture_cuda(commit.beta(), stream)?;
            Ok(serde_json::to_vec(&values)?)
        });
        debug_dump::write_json("fri_coeffs1.json", || -> anyhow::Result<Vec<u8>> {
            let source = commit
                .coefficients_at(0)
                .ok_or_else(|| anyhow::anyhow!("FRI coefficient layer 0 is unavailable"))?;
            let values = debug_dump::capture_cuda(source, stream)?;
            Ok(serde_json::to_vec(&values)?)
        });
    }

    pub(crate) fn copy_final_coefficients(
        commit: &FriCommitState,
        final_coefficients: &mut HostAllocation<GoldilocksFieldBoojum>,
        final_round: usize,
        stream: &CudaStream,
    ) {
        memory_copy_async(final_coefficients, commit.coefficients(final_round), stream).unwrap();
        stream.synchronize().unwrap();
        debug_dump::write_json("fri_final_coeffs.json", || -> anyhow::Result<Vec<u8>> {
            let values = debug_dump::capture_cuda(commit.coefficients(final_round), stream)?;
            Ok(serde_json::to_vec(&values)?)
        });
    }
}
