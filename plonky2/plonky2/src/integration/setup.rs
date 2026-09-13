//! GPU prover facade composition and long-lived resource ownership.
//!
//! `GpuProverSetup` contains four lifecycle owners. Construction allocates the CUDA context,
//! `fp_inputs`, the constants Merkle tree, and one overlay scratch block sized for
//! `max(witgen SoA, cm1–cm3 + FRI)`. Witgen and later commit/FRI take exclusive views of that
//! scratch; they do not `cudaFree` and re-`cudaMalloc`. Field declaration order keeps the CUDA
//! context guard alive until every other GPU allocation owned by the facade has been dropped.

use std::mem;
use std::time::Instant;

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use boojum_cuda::context::Context;
use cudart::error::get_last_error;
use cudart::memory::{CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use cudart::result::{CudaResult, CudaResultWrap};
use cudart::stream::CudaStream;
use plonky2_field::extension::Extendable;

use super::commitments::CommitmentBuffers;
use super::final_poly::{FinalPolyScratch, FinalPolyState};
use super::fri::{
    FriCommitBuffers, FriCommitState, FriPowBuffers, FriQueryBuffers, FriQueryGeometry,
    FriQueryState, FriState,
};
use super::layout::PolyLayout;
use super::openings::{OpeningBuffers, OpeningState, OpeningsService};
use super::overlay::{self, ScratchCursor};
use super::partial_products::{PartialProductConfig, PartialProductState};
use super::poseidon::ChallengerGpu;
use super::proof_assembly::ProofAssemblyBuffers;
use super::public_inputs::PublicInputBuffers;
use super::quotient::state::{QuotientState, QuotientStatic, QuotientStaticConfig};
use super::quotient::{QuotientInitInputs, QuotientInitWorkspace, QuotientService};
use super::witgen::{WitnessBuffers, WitnessPlan, WitnessService};
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::{CommonCircuitData, GpuProverOnlyCircuitData};
use crate::plonk::config::{GenericConfig, GenericHashOut};
use crate::util::log2_strict;

pub(crate) struct GpuCircuitSetup<F: RichField + Extendable<D>, const D: usize> {
    pub(crate) challenger: ChallengerGpu,
    pub(crate) cmd: CommonCircuitData<F, D>,
    pub(crate) layout: PolyLayout,
    pub(crate) witness_plan: WitnessPlan,
    pub(crate) quotient_static: QuotientStatic,
    // Context::create installs pointers to its ten DeviceAllocations in CUDA global symbols.
    // Keep this field last so those twiddle allocations are the final resources dropped.
    pub(crate) _context_guard: Context,
}

pub(crate) struct GpuWorkspace {
    pub(crate) ldes: DeviceAllocation<GoldilocksFieldBoojum>,
    pub(crate) witness_buffers: WitnessBuffers,
    pub(crate) commitments: CommitmentBuffers,
    pub(crate) fp_inputs: DeviceAllocation<GoldilocksFieldBoojum>,
    /// Witgen SoA. When a CUDA graph stays live through prove, this block is
    /// not freed until `GpuProverSetup` drops the graph first.
    pub(crate) scratch: DeviceAllocation<u8>,
    /// Fresh cm1–cm3 + FRI block used while `scratch` still holds SoA for a
    /// live graph. Empty (null) when we recycle SoA after destroy/streamed.
    pub(crate) fri_scratch: DeviceAllocation<u8>,
}

impl GpuWorkspace {
    pub(crate) fn bind_post_witgen_views(
        &mut self,
        fri: &mut FriState,
        degree: usize,
        extension_degree: usize,
        rate_bits: usize,
        reduction_arity_bits: &[usize],
        keep_soa: bool,
        overlay_soa: bool,
    ) {
        self.witness_buffers.unbind_device_views();
        let post = overlay::post_witgen_bytes(
            degree,
            extension_degree,
            rate_bits,
            reduction_arity_bits,
        );
        if keep_soa {
            cudart::device::device_synchronize().unwrap();
            self.fri_scratch = DeviceAllocation::<u8>::alloc(post).unwrap();
            println!(
                "Post-witgen scratch: {:.2} MiB (extra, SoA+graph still live)",
                post as f64 / (1024.0 * 1024.0)
            );
            let mut cursor = ScratchCursor::new(&mut self.fri_scratch);
            self.commitments.bind_delayed_from_scratch(&mut cursor);
            fri.bind_commit_from_scratch(
                &mut cursor,
                degree,
                extension_degree,
                rate_bits,
                reduction_arity_bits,
            );
            return;
        }
        if overlay_soa {
            // Fallback if extra FRI still misses and the SoA block was sized ≥ FRI.
            // Tight captured SoA (C210 squeeze) cannot take this branch.
            assert!(
                self.scratch.len() >= post,
                "overlay SoA scratch too small: have {} need {post}",
                self.scratch.len()
            );
            println!(
                "Post-witgen scratch: {:.2} MiB (overlay SoA, graph still live)",
                post as f64 / (1024.0 * 1024.0)
            );
            let mut cursor = ScratchCursor::new(&mut self.scratch);
            self.commitments.bind_delayed_from_scratch(&mut cursor);
            fri.bind_commit_from_scratch(
                &mut cursor,
                degree,
                extension_degree,
                rate_bits,
                reduction_arity_bits,
            );
            return;
        }
        cudart::device::device_synchronize().unwrap();
        // Assigning `*scratch = alloc(post)` would cudaMalloc the FRI block
        // *before* dropping SoA. Fib C 1048575 had SoA==post==7407 MiB with
        // only 3790 MiB free → ErrorMemoryAllocation. Free first so peak is
        // max(witgen, FRI), not the sum.
        let old = mem::replace(
            &mut self.scratch,
            unsafe { DeviceAllocation::<u8>::from_raw_parts(std::ptr::null_mut(), 0) },
        );
        drop(old);
        cudart::device::device_synchronize().unwrap();
        self.scratch = DeviceAllocation::<u8>::alloc(post).unwrap();
        println!(
            "Post-witgen scratch: {:.2} MiB (fresh, not SoA reuse)",
            post as f64 / (1024.0 * 1024.0)
        );
        let mut cursor = ScratchCursor::new(&mut self.scratch);
        self.commitments.bind_delayed_from_scratch(&mut cursor);
        fri.bind_commit_from_scratch(
            &mut cursor,
            degree,
            extension_degree,
            rate_bits,
            reduction_arity_bits,
        );
    }
}

pub(crate) struct GpuProofState {
    pub(crate) public_inputs: PublicInputBuffers,
    pub(crate) quotient_state: QuotientState,
    pub(crate) partial_products: PartialProductState,
    pub(crate) openings: OpeningState,
    pub(crate) final_poly: FinalPolyState,
    pub(crate) fri: FriState,
}

pub(crate) struct GpuSetupInitializer;

impl GpuSetupInitializer {
    pub(crate) fn initialize<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        circuit: &mut GpuCircuitSetup<F, D>,
        workspace: &mut GpuWorkspace,
        proof_state: &mut GpuProofState,
        prover_data: &GpuProverOnlyCircuitData<F, C, D>,
    ) -> CudaResult<()> {
        println!("Initializing GPU Setup!");
        let stream = CudaStream::default();
        let inputs = QuotientInitInputs {
            subgroup: &prover_data.subgroup,
            degree_bits: circuit.cmd.degree_bits(),
            num_routed_wires: circuit.cmd.config.num_routed_wires,
        };
        let quotient_workspace = QuotientInitWorkspace {
            static_data: &mut circuit.quotient_static,
        };
        QuotientService::initialize::<F, D>(inputs, quotient_workspace, &stream)?;

        let public_inputs_index = WitnessService::prepare_public_input_indices(
            &mut circuit.witness_plan,
            &mut workspace.witness_buffers,
            &prover_data.public_inputs,
            &prover_data.representative_map,
            circuit.cmd.config.num_wires,
            circuit.cmd.degree(),
            &stream,
        );

        circuit
            .challenger
            .observe_elements(unsafe { mem::transmute(&prover_data.circuit_digest.to_vec()) })
            .unwrap();

        let zeta_g = OpeningsService::initialize::<F, D>(
            &mut proof_state.openings,
            circuit.cmd.degree_bits(),
            &stream,
        );
        if cfg!(feature = "verbose") {
            println!("pi index:{:?}", public_inputs_index);
            println!("zeta_g: {:?}", zeta_g);
        }
        get_last_error().wrap()
    }
}

pub(crate) struct GpuProverSetup<F: RichField + Extendable<D>, const D: usize> {
    // Rust drops fields in declaration order. Circuit state is deliberately last because it owns
    // `_context_guard`, whose CUDA-global twiddle pointers must outlive all dependent resources.
    proof_out: ProofAssemblyBuffers<F>,
    proof_state: GpuProofState,
    workspace: GpuWorkspace,
    circuit: GpuCircuitSetup<F, D>,
}

impl<F: RichField + Extendable<D>, const D: usize> GpuProverSetup<F, D> {
    pub(crate) fn new(
        cmd: &CommonCircuitData<F, D>,
        num_const_sigmas: usize,
        num_ldes: [usize; 4],
    ) -> Self {
        let cmd_cloned = cmd.clone();
        let num_pps = cmd.config.num_routed_wires / cmd.config.max_quotient_degree_factor
            * cmd.config.num_challenges;
        let num_quotients = cmd.config.max_quotient_degree_factor * cmd.config.num_challenges;
        let length = cmd.degree() * cmd.config.num_challenges * cmd.config.num_routed_wires
            / cmd.quotient_degree_factor;
        let partial_product_config = PartialProductConfig::new(
            cmd.quotient_degree_factor,
            cmd.degree(),
            cmd.degree_bits(),
            cmd.config.num_routed_wires,
            cmd.config.num_challenges,
            cmd.num_constants,
        );
        let quotient_static_config = QuotientStaticConfig::new(
            cmd.degree(),
            cmd.quotient_degree_factor,
            cmd.config.max_quotient_degree_factor,
            cmd.config.num_routed_wires,
            log2_strict(cmd.quotient_degree_factor),
        );
        let layout = PolyLayout::new(
            cmd.degree(),
            cmd.config.fri_config.rate_bits,
            num_const_sigmas,
            cmd.config.num_wires,
            num_pps,
            num_quotients,
            num_ldes,
        );
        let num_polys = layout.num_polys();
        let fp_len = layout.fp_len();
        let lde_len = layout.lde_len();

        let fri_initial_leaves = vec![
            HostAllocation::<GoldilocksFieldBoojum>::alloc(
                num_const_sigmas * cmd.fri_params.config.num_query_rounds,
                CudaHostAllocFlags::DEFAULT,
            )
            .unwrap(),
            HostAllocation::<GoldilocksFieldBoojum>::alloc(
                cmd.config.num_wires * cmd.fri_params.config.num_query_rounds,
                CudaHostAllocFlags::DEFAULT,
            )
            .unwrap(),
            HostAllocation::<GoldilocksFieldBoojum>::alloc(
                num_pps * cmd.fri_params.config.num_query_rounds,
                CudaHostAllocFlags::DEFAULT,
            )
            .unwrap(),
            HostAllocation::<GoldilocksFieldBoojum>::alloc(
                num_quotients * cmd.fri_params.config.num_query_rounds,
                CudaHostAllocFlags::DEFAULT,
            )
            .unwrap(),
        ];
        let mut leaves_len = cmd.degree() * D << cmd.config.fri_config.rate_bits;
        let mut arity_sum = 0;
        let mut total_step_siblings = 0;
        let mut current_step_siblings =
            cmd.degree_bits() + cmd.config.fri_config.rate_bits - cmd.config.fri_config.cap_height;
        for arity_bits in &cmd.fri_params.reduction_arity_bits {
            arity_sum += 1 << arity_bits;
            current_step_siblings -= arity_bits;
            total_step_siblings += current_step_siblings;
            // FRI commit trees overlay the same scratch as witgen SoA. Geometry still has to
            // be walked here for query/host buffer sizes and scratch layout.
            leaves_len = leaves_len >> arity_bits;
        }

        if cfg!(feature = "verbose") {
            println!(
                "arity sum: {}, total step siblings: {}",
                arity_sum, total_step_siblings
            );
        }
        let challenger = ChallengerGpu::new();

        let fri_pow_witness_found = DeviceAllocation::<u32>::alloc(1).unwrap();
        let fri_pow_witness_found_buf = DeviceAllocation::<u32>::alloc(128 * 64).unwrap();
        let fri_pow_witness = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(1).unwrap();
        let fri_pow_witness_buf =
            DeviceAllocation::<GoldilocksFieldBoojum>::alloc(128 * 64).unwrap();
        let public_inputs = PublicInputBuffers::new(cmd.num_public_inputs);
        let quotient_state = QuotientState::new(cmd.config.num_challenges);
        let partial_products =
            PartialProductState::new(partial_product_config, cmd.config.num_challenges, length);
        let openings_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(
            D * (num_polys + cmd.config.num_challenges),
            CudaHostAllocFlags::DEFAULT,
        )
        .unwrap();
        let fri_final_coeffs_host =
            HostAllocation::<GoldilocksFieldBoojum>::alloc(leaves_len, CudaHostAllocFlags::DEFAULT)
                .unwrap();
        let fri_steps_evals_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(
            D * arity_sum * cmd.fri_params.config.num_query_rounds,
            CudaHostAllocFlags::DEFAULT,
        )
        .unwrap();
        let fri_steps_siblings_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(
            4 * total_step_siblings * cmd.fri_params.config.num_query_rounds,
            CudaHostAllocFlags::DEFAULT,
        )
        .unwrap();
        let context_guard = Context::create(12, 12).unwrap();
        let ldes = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(lde_len).unwrap();
        let witness_plan = WitnessPlan::new();
        let witness_buffers = WitnessBuffers::new(cmd.num_public_inputs);
        let tree_len = 8 * cmd.degree() << cmd.config.fri_config.rate_bits;
        let commitments = CommitmentBuffers::new(tree_len);
        let post_witgen_scratch = overlay::post_witgen_bytes(
            cmd.degree(),
            D,
            cmd.config.fri_config.rate_bits,
            &cmd.fri_params.reduction_arity_bits,
        );
        println!(
            "Post-witgen FRI block: {:.2} MiB (extra if live-graph fits; SoA scratch is captured-only)",
            post_witgen_scratch as f64 / (1024.0 * 1024.0)
        );
        // Delay the real alloc until witgen sizes are known. Pre-allocating the
        // post-witgen size and later cudaFree+realloc (MVM U2000 231→340 MiB)
        // coincided with FRI verify FAIL; SHA/Fact never hit that grow path.
        let scratch =
            unsafe { DeviceAllocation::<u8>::from_raw_parts(std::ptr::null_mut(), 0) };
        let zeta_g = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(D * 2).unwrap();
        let openings_reduction = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(
            D * (num_polys + cmd.config.num_challenges) * 64,
        )
        .unwrap();
        let openings_device = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(
            D * (num_polys + cmd.config.num_challenges),
        )
        .unwrap();
        let fp_alpha = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(D).unwrap();
        let fp_inputs = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(fp_len).unwrap();
        let fp_buffer = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(cmd.degree() * D).unwrap();
        let fp_quotient =
            DeviceAllocation::<GoldilocksFieldBoojum>::alloc(cmd.degree() * D + 2).unwrap();
        let final_poly_coeffs = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(
            cmd.degree() * D << cmd.config.fri_config.rate_bits,
        )
        .unwrap();
        let fri_beta = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(D).unwrap();
        let fri_query_round_challenges = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(
            cmd.fri_params.config.num_query_rounds,
        )
        .unwrap();
        let fri_buf = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(
            cmd.fri_params.config.num_query_rounds * (cmd.config.num_wires + 1) * 64,
        )
        .unwrap();
        let fri_initial_siblings = HostAllocation::<GoldilocksFieldBoojum>::alloc(
            4 * 4
                * (cmd.degree_bits() + cmd.config.fri_config.rate_bits
                    - cmd.config.fri_config.cap_height)
                * cmd.fri_params.config.num_query_rounds,
            CudaHostAllocFlags::DEFAULT,
        )
        .unwrap();
        let fri_steps_evals = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(
            D * arity_sum * cmd.fri_params.config.num_query_rounds,
        )
        .unwrap();
        let fri_steps_siblings = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(
            4 * total_step_siblings * cmd.fri_params.config.num_query_rounds,
        )
        .unwrap();
        let quotient_static = QuotientStatic::new(quotient_static_config);

        let openings = OpeningState::from_parts(
            zeta_g,
            OpeningBuffers::from_parts(openings_reduction, openings_device),
        );
        let final_poly = FinalPolyState::from_parts(
            fp_alpha,
            FinalPolyScratch::from_parts(fp_buffer, fp_quotient, final_poly_coeffs),
        );
        let fri = FriState::from_parts(
            FriCommitState::from_parts(fri_beta, FriCommitBuffers::unallocated()),
            FriPowBuffers::from_parts(
                fri_pow_witness_found,
                fri_pow_witness_found_buf,
                fri_pow_witness,
                fri_pow_witness_buf,
            ),
            FriQueryState::from_parts(
                FriQueryGeometry::new(arity_sum, total_step_siblings),
                FriQueryBuffers::from_parts(
                    fri_query_round_challenges,
                    fri_buf,
                    fri_steps_evals,
                    fri_steps_siblings,
                ),
            ),
        );
        let proof_out = ProofAssemblyBuffers::from_parts(
            fri_initial_leaves,
            openings_host,
            fri_final_coeffs_host,
            fri_steps_evals_host,
            fri_steps_siblings_host,
            fri_initial_siblings,
            Vec::new(),
            F::ZERO,
        );

        let circuit = GpuCircuitSetup {
            challenger,
            cmd: cmd_cloned,
            layout,
            witness_plan,
            quotient_static,
            _context_guard: context_guard,
        };
        let workspace = GpuWorkspace {
            ldes,
            witness_buffers,
            commitments,
            fp_inputs,
            scratch,
            fri_scratch: unsafe { DeviceAllocation::<u8>::from_raw_parts(std::ptr::null_mut(), 0) },
        };
        let proof_state = GpuProofState {
            public_inputs,
            quotient_state,
            partial_products,
            openings,
            final_poly,
            fri,
        };

        Self {
            proof_out,
            proof_state,
            workspace,
            circuit,
        }
    }

    #[inline]
    pub(crate) fn layout(&self) -> &PolyLayout {
        &self.circuit.layout
    }

    #[inline]
    pub(crate) fn witness_input_count(&self) -> usize {
        self.circuit.witness_plan.witness_input_count()
    }

    #[inline]
    pub(crate) fn parts(
        &self,
    ) -> (
        &GpuCircuitSetup<F, D>,
        &GpuWorkspace,
        &GpuProofState,
        &ProofAssemblyBuffers<F>,
    ) {
        (
            &self.circuit,
            &self.workspace,
            &self.proof_state,
            &self.proof_out,
        )
    }

    #[inline]
    pub(crate) fn parts_mut(
        &mut self,
    ) -> (
        &mut GpuCircuitSetup<F, D>,
        &mut GpuWorkspace,
        &mut GpuProofState,
        &mut ProofAssemblyBuffers<F>,
    ) {
        (
            &mut self.circuit,
            &mut self.workspace,
            &mut self.proof_state,
            &mut self.proof_out,
        )
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Drop for GpuProverSetup<F, D> {
    fn drop(&mut self) {
        let had_graph = self.circuit.witness_plan.has_live_graph();
        let t = Instant::now();
        WitnessService::release_after_replay(&mut self.circuit.witness_plan);
        if had_graph {
            println!(
                "Time taken to drop witgen graph (after prove/verify): {:?}",
                t.elapsed()
            );
        }
    }
}
