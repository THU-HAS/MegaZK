//! Owning state for the GPU FRI phase.
//!
//! This module only groups allocations by lifecycle and exposes the accessors used by the active
//! commit, PoW, query, and proof-assembly paths.

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::memory::DeviceAllocation;

use super::super::overlay::{fri_commit_part_lens, DeviceView, ScratchCursor};

pub(crate) struct FriCommitBuffers {
    tree_digests: Vec<DeviceView<GoldilocksFieldBoojum>>,
    tree_leaves: Vec<DeviceView<GoldilocksFieldBoojum>>,
    transposed_tree_leaves: Vec<DeviceView<GoldilocksFieldBoojum>>,
    folded_coefficients: Vec<DeviceView<GoldilocksFieldBoojum>>,
}

impl FriCommitBuffers {
    pub(crate) fn from_parts(
        tree_digests: Vec<DeviceView<GoldilocksFieldBoojum>>,
        tree_leaves: Vec<DeviceView<GoldilocksFieldBoojum>>,
        transposed_tree_leaves: Vec<DeviceView<GoldilocksFieldBoojum>>,
        folded_coefficients: Vec<DeviceView<GoldilocksFieldBoojum>>,
    ) -> Self {
        Self {
            tree_digests,
            tree_leaves,
            transposed_tree_leaves,
            folded_coefficients,
        }
    }

    pub(crate) fn unallocated() -> Self {
        Self::from_parts(Vec::new(), Vec::new(), Vec::new(), Vec::new())
    }

    pub(crate) fn bind_from_scratch(
        cursor: &mut ScratchCursor,
        degree: usize,
        extension_degree: usize,
        rate_bits: usize,
        reduction_arity_bits: &[usize],
    ) -> Self {
        let mut tree_digests = Vec::new();
        let mut tree_leaves = Vec::new();
        let mut transposed_tree_leaves = Vec::new();
        let mut folded_coefficients = Vec::new();
        for part in fri_commit_part_lens(
            degree,
            extension_degree,
            rate_bits,
            reduction_arity_bits,
        ) {
            tree_digests.push(cursor.take(part.digests_len));
            tree_leaves.push(cursor.take(part.leaves_len));
            transposed_tree_leaves.push(cursor.take(part.leaves_len));
            folded_coefficients.push(cursor.take(part.folded_len));
        }
        Self::from_parts(
            tree_digests,
            tree_leaves,
            transposed_tree_leaves,
            folded_coefficients,
        )
    }

    fn is_allocated(&self) -> bool {
        !self.tree_digests.is_empty()
    }
}

pub(crate) struct FriCommitState {
    beta: DeviceAllocation<GoldilocksFieldBoojum>,
    buffers: FriCommitBuffers,
}

pub(crate) struct FriCommitRoundMut<'a> {
    pub(crate) leaves: &'a mut DeviceView<GoldilocksFieldBoojum>,
    pub(crate) transposed_leaves: &'a mut DeviceView<GoldilocksFieldBoojum>,
    pub(crate) digests: &'a mut DeviceView<GoldilocksFieldBoojum>,
}

pub(crate) struct FriReductionBuffersMut<'a> {
    pub(crate) previous_coefficients: Option<&'a DeviceView<GoldilocksFieldBoojum>>,
    pub(crate) output_coefficients: &'a mut DeviceView<GoldilocksFieldBoojum>,
    pub(crate) beta: &'a DeviceAllocation<GoldilocksFieldBoojum>,
}

impl FriCommitState {
    pub(crate) fn from_parts(
        beta: DeviceAllocation<GoldilocksFieldBoojum>,
        buffers: FriCommitBuffers,
    ) -> Self {
        Self { beta, buffers }
    }

    pub(crate) fn ensure_buffers_allocated(&self) {
        assert!(
            self.buffers.is_allocated(),
            "FRI commit buffers must be bound from scratch after witgen"
        );
    }

    pub(crate) fn bind_buffers_from_scratch(
        &mut self,
        cursor: &mut ScratchCursor,
        degree: usize,
        extension_degree: usize,
        rate_bits: usize,
        reduction_arity_bits: &[usize],
    ) {
        self.buffers = FriCommitBuffers::bind_from_scratch(
            cursor,
            degree,
            extension_degree,
            rate_bits,
            reduction_arity_bits,
        );
    }

    pub(crate) fn round_mut(&mut self, round: usize) -> FriCommitRoundMut<'_> {
        let FriCommitBuffers {
            tree_digests,
            tree_leaves,
            transposed_tree_leaves,
            ..
        } = &mut self.buffers;
        FriCommitRoundMut {
            leaves: &mut tree_leaves[round],
            transposed_leaves: &mut transposed_tree_leaves[round],
            digests: &mut tree_digests[round],
        }
    }

    #[inline]
    pub(crate) fn tree_digests(&self, round: usize) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &*self.buffers.tree_digests[round]
    }

    #[inline]
    pub(crate) fn tree_digests_mut(
        &mut self,
        round: usize,
    ) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut *self.buffers.tree_digests[round]
    }

    #[inline]
    pub(crate) fn tree_leaves(&self, round: usize) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &*self.buffers.tree_leaves[round]
    }

    #[inline]
    pub(crate) fn tree_leaves_at(
        &self,
        round: usize,
    ) -> Option<&DeviceAllocation<GoldilocksFieldBoojum>> {
        self.buffers.tree_leaves.get(round).map(|view| &**view)
    }

    #[inline]
    pub(crate) fn tree_leaves_mut(
        &mut self,
        round: usize,
    ) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut *self.buffers.tree_leaves[round]
    }

    #[inline]
    pub(crate) fn transposed_tree_leaves_at(
        &self,
        round: usize,
    ) -> Option<&DeviceAllocation<GoldilocksFieldBoojum>> {
        self.buffers
            .transposed_tree_leaves
            .get(round)
            .map(|view| &**view)
    }

    #[inline]
    pub(crate) fn beta(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.beta
    }

    #[inline]
    pub(crate) fn beta_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.beta
    }

    pub(crate) fn reduction_buffers_mut(&mut self, round: usize) -> FriReductionBuffersMut<'_> {
        let (previous_rounds, current_and_later) =
            self.buffers.folded_coefficients.split_at_mut(round);
        FriReductionBuffersMut {
            previous_coefficients: round
                .checked_sub(1)
                .map(|previous_round| &previous_rounds[previous_round]),
            output_coefficients: &mut current_and_later[0],
            beta: &self.beta,
        }
    }

    pub(crate) fn fold_output_mut(
        &mut self,
        round: usize,
    ) -> (
        &DeviceAllocation<GoldilocksFieldBoojum>,
        &mut DeviceAllocation<GoldilocksFieldBoojum>,
    ) {
        (
            &*self.buffers.folded_coefficients[round],
            &mut *self.buffers.tree_leaves[round + 1],
        )
    }

    #[inline]
    pub(crate) fn coefficients(&self, round: usize) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &*self.buffers.folded_coefficients[round]
    }

    #[inline]
    pub(crate) fn coefficients_at(
        &self,
        round: usize,
    ) -> Option<&DeviceAllocation<GoldilocksFieldBoojum>> {
        self.buffers
            .folded_coefficients
            .get(round)
            .map(|view| &**view)
    }

    #[inline]
    pub(crate) fn coefficients_mut(
        &mut self,
        round: usize,
    ) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut *self.buffers.folded_coefficients[round]
    }
}

pub(crate) struct FriPowBuffers {
    witness_found: DeviceAllocation<u32>,
    witness_found_buffer: DeviceAllocation<u32>,
    witness: DeviceAllocation<GoldilocksFieldBoojum>,
    witness_buffer: DeviceAllocation<GoldilocksFieldBoojum>,
}

pub(crate) struct FriPowKernelBuffersMut<'a> {
    pub(crate) witness_found: &'a mut DeviceAllocation<u32>,
    pub(crate) witness_found_buffer: &'a mut DeviceAllocation<u32>,
    pub(crate) witness: &'a mut DeviceAllocation<GoldilocksFieldBoojum>,
    pub(crate) witness_buffer: &'a mut DeviceAllocation<GoldilocksFieldBoojum>,
}

impl FriPowBuffers {
    pub(crate) fn from_parts(
        witness_found: DeviceAllocation<u32>,
        witness_found_buffer: DeviceAllocation<u32>,
        witness: DeviceAllocation<GoldilocksFieldBoojum>,
        witness_buffer: DeviceAllocation<GoldilocksFieldBoojum>,
    ) -> Self {
        Self {
            witness_found,
            witness_found_buffer,
            witness,
            witness_buffer,
        }
    }

    pub(crate) fn kernel_buffers_mut(&mut self) -> FriPowKernelBuffersMut<'_> {
        FriPowKernelBuffersMut {
            witness_found: &mut self.witness_found,
            witness_found_buffer: &mut self.witness_found_buffer,
            witness: &mut self.witness,
            witness_buffer: &mut self.witness_buffer,
        }
    }

    #[inline]
    pub(crate) fn witness(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.witness
    }

    #[inline]
    pub(crate) fn witness_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.witness
    }
}

pub(crate) struct FriQueryGeometry {
    arity_sum: usize,
    total_step_siblings: usize,
}

impl FriQueryGeometry {
    pub(crate) fn new(arity_sum: usize, total_step_siblings: usize) -> Self {
        Self {
            arity_sum,
            total_step_siblings,
        }
    }
}

pub(crate) struct FriQueryBuffers {
    round_challenges: DeviceAllocation<GoldilocksFieldBoojum>,
    scratch: DeviceAllocation<GoldilocksFieldBoojum>,
    step_evaluations: DeviceAllocation<GoldilocksFieldBoojum>,
    step_siblings: DeviceAllocation<GoldilocksFieldBoojum>,
}

pub(crate) struct FriInitialQueryBuffersMut<'a> {
    pub(crate) round_challenges: &'a mut DeviceAllocation<GoldilocksFieldBoojum>,
    pub(crate) scratch: &'a mut DeviceAllocation<GoldilocksFieldBoojum>,
}

pub(crate) struct FriStepQueryBuffersMut<'a> {
    pub(crate) round_challenges: &'a mut DeviceAllocation<GoldilocksFieldBoojum>,
    pub(crate) step_evaluations: &'a mut DeviceAllocation<GoldilocksFieldBoojum>,
    pub(crate) step_siblings: &'a mut DeviceAllocation<GoldilocksFieldBoojum>,
}

impl FriQueryBuffers {
    pub(crate) fn from_parts(
        round_challenges: DeviceAllocation<GoldilocksFieldBoojum>,
        scratch: DeviceAllocation<GoldilocksFieldBoojum>,
        step_evaluations: DeviceAllocation<GoldilocksFieldBoojum>,
        step_siblings: DeviceAllocation<GoldilocksFieldBoojum>,
    ) -> Self {
        Self {
            round_challenges,
            scratch,
            step_evaluations,
            step_siblings,
        }
    }
}

pub(crate) struct FriQueryState {
    geometry: FriQueryGeometry,
    buffers: FriQueryBuffers,
}

impl FriQueryState {
    pub(crate) fn from_parts(geometry: FriQueryGeometry, buffers: FriQueryBuffers) -> Self {
        Self { geometry, buffers }
    }

    #[inline]
    pub(crate) fn arity_sum(&self) -> usize {
        self.geometry.arity_sum
    }

    #[inline]
    pub(crate) fn total_step_siblings(&self) -> usize {
        self.geometry.total_step_siblings
    }

    #[inline]
    pub(crate) fn round_challenges_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.buffers.round_challenges
    }

    pub(crate) fn initial_buffers_mut(&mut self) -> FriInitialQueryBuffersMut<'_> {
        FriInitialQueryBuffersMut {
            round_challenges: &mut self.buffers.round_challenges,
            scratch: &mut self.buffers.scratch,
        }
    }

    pub(crate) fn step_buffers_mut(&mut self) -> FriStepQueryBuffersMut<'_> {
        FriStepQueryBuffersMut {
            round_challenges: &mut self.buffers.round_challenges,
            step_evaluations: &mut self.buffers.step_evaluations,
            step_siblings: &mut self.buffers.step_siblings,
        }
    }

    #[inline]
    pub(crate) fn step_evaluations(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.buffers.step_evaluations
    }

    #[inline]
    pub(crate) fn step_siblings(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.buffers.step_siblings
    }
}

pub(crate) struct FriState {
    commit: FriCommitState,
    pow: FriPowBuffers,
    query: FriQueryState,
}

impl FriState {
    pub(crate) fn from_parts(
        commit: FriCommitState,
        pow: FriPowBuffers,
        query: FriQueryState,
    ) -> Self {
        Self { commit, pow, query }
    }

    #[inline]
    pub(crate) fn pow_mut(&mut self) -> &mut FriPowBuffers {
        &mut self.pow
    }

    #[inline]
    pub(crate) fn query(&self) -> &FriQueryState {
        &self.query
    }

    #[inline]
    pub(crate) fn query_mut(&mut self) -> &mut FriQueryState {
        &mut self.query
    }

    #[inline]
    pub(crate) fn commit_mut(&mut self) -> &mut FriCommitState {
        &mut self.commit
    }

    pub(crate) fn commit_and_query_mut(&mut self) -> (&FriCommitState, &mut FriQueryState) {
        (&self.commit, &mut self.query)
    }

    #[inline]
    pub(crate) fn first_commit_leaves_mut(
        &mut self,
    ) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        self.commit.tree_leaves_mut(0)
    }

    pub(crate) fn ensure_commit_allocated(&self) {
        self.commit.ensure_buffers_allocated();
    }

    pub(crate) fn bind_commit_from_scratch(
        &mut self,
        cursor: &mut ScratchCursor,
        degree: usize,
        extension_degree: usize,
        rate_bits: usize,
        reduction_arity_bits: &[usize],
    ) {
        self.commit.bind_buffers_from_scratch(
            cursor,
            degree,
            extension_degree,
            rate_bits,
            reduction_arity_bits,
        );
    }
}
