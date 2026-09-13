//! Active GPU FRI initial and step query proof kernels.
//!
//! Query challenges are sampled by the global orchestrator in `orchestrator.rs` before this
//! service is called.

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::cuda_kernel;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::memory::{memory_copy_async, DeviceAllocation};
use cudart::stream::CudaStream;

use super::state::{FriCommitState, FriQueryState};
use crate::integration::layout::{PolyLayout, PolySegment};
use crate::integration::proof_assembly::{FriInitialQueryOutputMut, FriStepQueryOutputMut};
use crate::util::debug_dump;

cuda_kernel!(FRIInitialLeavesSeg, fri_initial_leaves_seg(
    inputs: *const GoldilocksFieldBoojum,
    buffer: *mut GoldilocksFieldBoojum,
    x_indexes: *mut GoldilocksFieldBoojum,
    num_querys: u32,
    num_polys: u32,
    log_n: u32,
    rate_bits: u32,
));

cuda_kernel!(FRIInitialLeavesReducePartial, fri_initial_leaves_reduce_partial(
    ldes: *const GoldilocksFieldBoojum,
    buffer: *const GoldilocksFieldBoojum,
    leave: *mut GoldilocksFieldBoojum,
    x_indexes: *mut GoldilocksFieldBoojum,
    num_querys: u32,
    num_polys: u32,
    num_ldes: u32,
    log_n: u32,
    rate_bits: u32,
));

cuda_kernel!(FRIInitialSiblings, fri_initial_siblings(
    digests: *const GoldilocksFieldBoojum,
    siblings: *mut GoldilocksFieldBoojum,
    x_indexes: *const GoldilocksFieldBoojum,
    num_querys: u32,
    num_polys: u32,
    log_n: u32,
    cap_height: u32,
    num_layers: u32,
));

cuda_kernel!(FRIStep, fri_step(
    leaves: *const GoldilocksFieldBoojum,
    digests: *const GoldilocksFieldBoojum,
    leave: *mut GoldilocksFieldBoojum,
    siblings: *mut GoldilocksFieldBoojum,
    x_indexes: *mut GoldilocksFieldBoojum,
    num_querys: u32,
    arity_bits: u32,
    arity_sum: u32,
    arity_acc: u32,
    total_step_siblings: u32,
    acc_step_siblings: u32,
    log_n: u32,
    cap_height: u32,
));

#[derive(Copy, Clone)]
pub(crate) struct FriQueryConfig {
    num_query_rounds: usize,
    degree_bits: usize,
    rate_bits: usize,
    cap_height: usize,
}

impl FriQueryConfig {
    pub(crate) fn new(
        num_query_rounds: usize,
        degree_bits: usize,
        rate_bits: usize,
        cap_height: usize,
    ) -> Self {
        Self {
            num_query_rounds,
            degree_bits,
            rate_bits,
            cap_height,
        }
    }
}

pub(crate) struct FriQueryService;

impl FriQueryService {
    pub(crate) fn initial_proofs(
        query_state: &mut FriQueryState,
        output: FriInitialQueryOutputMut<'_>,
        layout: &PolyLayout,
        fp_inputs: &DeviceAllocation<GoldilocksFieldBoojum>,
        ldes: &DeviceAllocation<GoldilocksFieldBoojum>,
        commitment_tree_digests: &[*const GoldilocksFieldBoojum],
        config: FriQueryConfig,
    ) {
        let query = query_state.initial_buffers_mut();
        let num_querys = config.num_query_rounds as u32;
        let nums = PolySegment::ALL.map(|segment| layout.polynomial_count(segment));
        let inputs_offsets = PolySegment::ALL
            .map(|segment| layout.fp_offset_after_ldes(segment, config.degree_bits));
        let num_ntts = PolySegment::ALL.map(|segment| layout.non_lde_polynomial_count(segment));
        let log_n = config.degree_bits as u32;
        let rate_bits = config.rate_bits as u32;
        let num_layers = config.degree_bits + config.rate_bits - config.cap_height;
        let stream = CudaStream::default();
        for i in 0..4 {
            if num_ntts[i] > 0 {
                let threads: u32 = 64;
                let blocks = (config.num_query_rounds * num_ntts[i]) as u32;
                let launch = CudaLaunchConfig::basic(blocks, threads, &stream);
                let args = FRIInitialLeavesSegArguments::new(
                    fp_inputs.as_ptr().wrapping_add(inputs_offsets[i]),
                    query.scratch.as_mut_ptr(),
                    query.round_challenges.as_mut_ptr(),
                    num_querys,
                    num_ntts[i] as u32,
                    log_n,
                    rate_bits,
                );
                FRIInitialLeavesSegFunction(fri_initial_leaves_seg)
                    .launch(&launch, &args)
                    .unwrap();
                stream.synchronize().unwrap();
            }

            let threads: u32 = 32;
            let blocks = ((config.num_query_rounds * nums[i]) as u32 + threads - 1) / threads;
            let launch = CudaLaunchConfig::basic(blocks, threads, &stream);
            let args = FRIInitialLeavesReducePartialArguments::new(
                ldes.as_ptr()
                    .wrapping_add(layout.lde_offset(PolySegment::ALL[i])),
                query.scratch.as_ptr(),
                output.leaves[i].as_mut_ptr(),
                query.round_challenges.as_mut_ptr(),
                num_querys,
                nums[i] as u32,
                layout.num_ldes(PolySegment::ALL[i]) as u32,
                log_n,
                rate_bits,
            );
            FRIInitialLeavesReducePartialFunction(fri_initial_leaves_reduce_partial)
                .launch(&launch, &args)
                .unwrap();
            stream.synchronize().unwrap();

            let threads: u32 = 1;
            let blocks = config.num_query_rounds as u32;
            let launch = CudaLaunchConfig::basic(blocks, threads, &stream);
            let args = FRIInitialSiblingsArguments::new(
                commitment_tree_digests[i],
                output
                    .siblings
                    .as_mut_ptr()
                    .wrapping_add(i * 4 * num_layers),
                query.round_challenges.as_ptr(),
                num_querys,
                nums[i] as u32,
                (config.degree_bits + config.rate_bits) as u32,
                config.cap_height as u32,
                num_layers as u32,
            );
            FRIInitialSiblingsFunction(fri_initial_siblings)
                .launch(&launch, &args)
                .unwrap();
            stream.synchronize().unwrap();
        }

        debug_dump::write_json("initial_sib.json", || -> anyhow::Result<Vec<u8>> {
            let siblings = debug_dump::capture_cuda(output.siblings, &stream)?;
            Ok(serde_json::to_vec(&siblings)?)
        });
        debug_dump::write_json("initial_leaves_0.json", || -> anyhow::Result<Vec<u8>> {
            let source = output
                .leaves
                .get(0)
                .ok_or_else(|| anyhow::anyhow!("initial FRI leaves group 0 is unavailable"))?;
            let leaves = debug_dump::capture_cuda(source, &stream)?;
            Ok(serde_json::to_vec(&leaves)?)
        });
    }

    pub(crate) fn query_steps(
        commit: &FriCommitState,
        query: &mut FriQueryState,
        output: FriStepQueryOutputMut<'_>,
        reduction_arity_bits: &[usize],
        config: FriQueryConfig,
    ) {
        let arity_sum = query.arity_sum();
        let total_step_siblings = query.total_step_siblings();
        let mut arity_acc: usize = 0;
        let mut acc_step_siblings: usize = 0;
        let stream = CudaStream::default();
        let mut current_step_siblings = config.degree_bits + config.rate_bits - config.cap_height;
        {
            let query_buffers = query.step_buffers_mut();
            for (i, arity_bits) in reduction_arity_bits.to_vec().iter().enumerate() {
                let arity = *arity_bits;
                current_step_siblings -= arity;
                let threads: u32 = 1;
                let blocks = config.num_query_rounds as u32;
                let launch = CudaLaunchConfig::basic(blocks, threads, &stream);
                let args = FRIStepArguments::new(
                    commit.tree_leaves(i).as_ptr(),
                    commit.tree_digests(i).as_ptr(),
                    query_buffers.step_evaluations.as_mut_ptr(),
                    query_buffers.step_siblings.as_mut_ptr(),
                    query_buffers.round_challenges.as_mut_ptr(),
                    config.num_query_rounds as u32,
                    arity as u32,
                    arity_sum as u32,
                    arity_acc as u32,
                    total_step_siblings as u32,
                    acc_step_siblings as u32,
                    (current_step_siblings + config.cap_height) as u32,
                    config.cap_height as u32,
                );
                FRIStepFunction(fri_step).launch(&launch, &args).unwrap();
                acc_step_siblings += current_step_siblings;
                arity_acc += 1 << arity;
            }
        }

        memory_copy_async(output.siblings, query.step_siblings(), &stream).unwrap();
        memory_copy_async(output.evaluations, query.step_evaluations(), &stream).unwrap();
        stream.synchronize().unwrap();
        debug_dump::write_json("step_leave.json", || -> anyhow::Result<Vec<u8>> {
            let evaluations = debug_dump::capture_cuda(query.step_evaluations(), &stream)?;
            Ok(serde_json::to_vec(&evaluations)?)
        });
        debug_dump::write_json("step_siblings.json", || {
            let siblings = output.siblings.to_vec();
            serde_json::to_vec(&siblings)
        });
    }
}
