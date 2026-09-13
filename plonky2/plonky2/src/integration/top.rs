//! GPU prover compatibility facade.
//!
//! Resource ownership and construction live in `setup.rs`; complete prove and transcript ordering
//! live in `orchestrator.rs`. Every method here is a compatibility delegate or build-time setup
//! helper.

use std::time::Instant;

use anyhow::Result;
use cudart::result::CudaResult;
use plonky2_field::extension::Extendable;

use super::commitments::CommitmentStage;
use super::orchestrator::GpuProverOrchestrator;
pub(crate) use super::setup::GpuProverSetup;
use super::setup::GpuSetupInitializer;
use super::witgen::WitnessService;
use crate::fri::proof::FriProof;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleCap;
use crate::iop::target::Target;
use crate::plonk::circuit_data::{CommonCircuitData, GpuProverOnlyCircuitData};
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::proof::{OpeningSet, ProofWithPublicInputs};

impl<F: RichField + Extendable<D>, const D: usize> GpuProverSetup<F, D> {
    pub(crate) fn init<C: GenericConfig<D, F = F>>(
        &mut self,
        prover_data: &GpuProverOnlyCircuitData<F, C, D>,
    ) -> CudaResult<()> {
        let (circuit, workspace, proof_state, _) = self.parts_mut();
        GpuSetupInitializer::initialize::<F, C, D>(circuit, workspace, proof_state, prover_data)
    }

    pub(crate) fn sorting_new<C: GenericConfig<D, F = F>>(
        &mut self,
        input_targets: &[Target],
        cmd: &CommonCircuitData<F, D>,
        prover_data: &GpuProverOnlyCircuitData<F, C, D>,
    ) {
        let (circuit, workspace, proof_state, _) = self.parts_mut();
        let post_witgen_bytes = super::overlay::post_witgen_bytes(
            cmd.degree(),
            D,
            cmd.config.fri_config.rate_bits,
            &cmd.fri_params.reduction_arity_bits,
        );
        WitnessService::build(
            &mut circuit.witness_plan,
            &mut workspace.witness_buffers,
            &circuit.layout,
            &mut workspace.fp_inputs,
            &mut workspace.scratch,
            input_targets,
            &prover_data.generators,
            &prover_data.generator_indices_by_watches,
            &prover_data.representative_map,
            cmd.config.num_wires,
            cmd.degree(),
            cmd.degree_bits(),
            post_witgen_bytes,
        );
        // Dummy scatter warmup during instantiate does not absorb the later
        // hash_pi tax. Launching hash_pi itself after instantiate does: C210
        // pays ~3.3s here (build), prove hash_pi is ~10µs, prove ~5.8s.
        if circuit.witness_plan.has_live_graph() {
            let t = Instant::now();
            GpuProverOrchestrator::hash_pi_new(circuit, workspace, proof_state).unwrap();
            println!(
                "  graph build warmup: hash_pi (first prove nongraph) {:?}",
                t.elapsed()
            );
        }
    }

    #[inline]
    pub(crate) fn upload_witness_inputs(&mut self, inputs: &[F]) {
        WitnessService::upload_inputs(&mut self.parts_mut().1.witness_buffers, inputs);
    }

    #[inline]
    pub(crate) fn wg(&mut self) {
        WitnessService::replay(&self.parts().0.witness_plan);
    }

    pub(crate) fn get_openings(&mut self) {
        let (circuit, workspace, proof_state, proof_out) = self.parts_mut();
        GpuProverOrchestrator::get_openings(circuit, workspace, proof_state, proof_out);
    }

    pub(crate) fn get_final_poly(&mut self) {
        let (circuit, workspace, proof_state, _) = self.parts_mut();
        GpuProverOrchestrator::get_final_poly(circuit, workspace, proof_state);
    }

    pub(crate) fn final_poly_coset_fft(&mut self) -> CudaResult<()> {
        let (circuit, _, proof_state, _) = self.parts_mut();
        GpuProverOrchestrator::final_poly_coset_fft(circuit, proof_state)
    }

    pub(crate) fn compute_quotient_polys_partial_gpu(&mut self) {
        let (circuit, workspace, proof_state, _) = self.parts_mut();
        GpuProverOrchestrator::compute_quotient(circuit, workspace, proof_state);
    }

    pub(crate) fn commit_0(&mut self) {
        let (circuit, workspace, _, _) = self.parts_mut();
        GpuProverOrchestrator::commit(circuit, workspace, CommitmentStage::Constants);
    }

    pub(crate) fn commit_1(&mut self) {
        let (circuit, workspace, _, _) = self.parts_mut();
        GpuProverOrchestrator::commit(circuit, workspace, CommitmentStage::Wires);
    }

    pub(crate) fn commit_2(&mut self) {
        let (circuit, workspace, _, _) = self.parts_mut();
        GpuProverOrchestrator::commit(circuit, workspace, CommitmentStage::PartialProducts);
    }

    pub(crate) fn commit_3(&mut self) {
        let (circuit, workspace, _, _) = self.parts_mut();
        GpuProverOrchestrator::commit(circuit, workspace, CommitmentStage::Quotients);
    }

    pub(crate) fn get_cap<H: Hasher<F>>(&mut self, n: usize) -> MerkleCap<F, H> {
        let (circuit, workspace, _, _) = self.parts();
        GpuProverOrchestrator::get_cap(circuit, workspace, CommitmentStage::from_legacy_index(n))
    }

    pub(crate) fn get_cap_device<H: Hasher<F>>(&mut self, n: usize) -> MerkleCap<F, H> {
        let (circuit, workspace, _, _) = self.parts();
        GpuProverOrchestrator::get_cap_device(
            circuit,
            workspace,
            CommitmentStage::from_legacy_index(n),
        )
    }

    pub(crate) fn zp(&mut self) {
        let (circuit, workspace, proof_state, _) = self.parts_mut();
        GpuProverOrchestrator::zp(circuit, workspace, proof_state);
    }

    pub(crate) fn hash_pi_new(&mut self) -> CudaResult<()> {
        let (circuit, workspace, proof_state, _) = self.parts_mut();
        GpuProverOrchestrator::hash_pi_new(circuit, workspace, proof_state)
    }

    pub(crate) fn get_caps<H: Hasher<F>>(&mut self) -> Result<Vec<MerkleCap<F, H>>> {
        let (circuit, workspace, _, _) = self.parts();
        GpuProverOrchestrator::get_caps(circuit, workspace)
    }

    pub(crate) fn get_opening_set(&mut self) -> OpeningSet<F, D> {
        let (circuit, _, proof_state, proof_out) = self.parts();
        GpuProverOrchestrator::get_opening_set(circuit, proof_state, proof_out)
    }

    pub(crate) fn get_fri_proof<H: Hasher<F>>(&mut self) -> FriProof<F, H, D> {
        let (circuit, _, proof_state, proof_out) = self.parts();
        GpuProverOrchestrator::get_fri_proof(circuit, proof_state, proof_out)
    }

    pub(crate) fn fri_committed_trees(&mut self) {
        let (circuit, _, proof_state, proof_out) = self.parts_mut();
        GpuProverOrchestrator::fri_committed_trees(circuit, proof_state, proof_out);
    }

    pub(crate) fn fri_proof_of_work_new(&mut self) {
        let (circuit, _, proof_state, proof_out) = self.parts_mut();
        GpuProverOrchestrator::fri_proof_of_work_new(circuit, proof_state, proof_out);
    }

    pub(crate) fn fri_prover_query_steps(&mut self) {
        let (circuit, _, proof_state, proof_out) = self.parts_mut();
        GpuProverOrchestrator::fri_prover_query_steps(circuit, proof_state, proof_out);
    }

    pub(crate) fn fri_prover_query_rounds(&mut self) {
        let (circuit, workspace, proof_state, proof_out) = self.parts_mut();
        GpuProverOrchestrator::fri_prover_query_rounds(circuit, workspace, proof_state, proof_out);
    }

    pub(crate) fn fri_prove(&mut self) {
        let (circuit, workspace, proof_state, proof_out) = self.parts_mut();
        GpuProverOrchestrator::fri_prove(circuit, workspace, proof_state, proof_out);
    }

    pub(crate) fn get_proof<C: GenericConfig<D, F = F>>(
        &mut self,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let (circuit, workspace, proof_state, proof_out) = self.parts();
        GpuProverOrchestrator::get_proof::<F, C, D>(circuit, workspace, proof_state, proof_out)
    }
}
