//! Global GPU prover and Fiat-Shamir transcript orchestration.
//!
//! This is the only component that knows the complete GPU prove order. Phase services receive
//! only their owning state and borrowed workspace views; none receives `GpuProverSetup` or the
//! challenger.

use alloc::format;
use std::time::Instant;

use anyhow::Result;
use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::memory::{memory_copy_async, memory_get_info, CudaHostAllocFlags, HostAllocation};
use cudart::result::CudaResult;
use cudart::stream::CudaStream;
use plonky2_field::extension::Extendable;

use super::commitments::{CommitmentConfig, CommitmentService, CommitmentStage};
use super::final_poly::{FinalPolyConfig, FinalPolyService};
use super::fri::{FriCommitService, FriPowInputs, FriPowService, FriQueryConfig, FriQueryService};
use super::layout::PolySegment;
use super::openings::{OpeningConfig, OpeningsService};
use super::overlay;
use super::partial_products::PartialProductsService;
use super::proof_assembly::{ProofAssemblyBuffers, ProofAssemblyConfig, ProofAssemblyService};
use super::public_inputs::PublicInputsService;
use super::quotient::{QuotientCircuitData, QuotientInputs, QuotientService, QuotientWorkspace};
use super::setup::{GpuCircuitSetup, GpuProofState, GpuWorkspace};
use super::transcript::TranscriptStep;
use super::witgen::WitnessService;
use crate::fri::proof::FriProof;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleCap;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::proof::{OpeningSet, ProofWithPublicInputs};
use crate::timed;
use crate::util::debug_dump;
use crate::util::timing::TimingTree;

pub(crate) struct GpuProverOrchestrator;

impl GpuProverOrchestrator {
    pub(crate) fn prove<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
        circuit: &mut GpuCircuitSetup<F, D>,
        workspace: &mut GpuWorkspace,
        proof_state: &mut GpuProofState,
        proof_out: &mut ProofAssemblyBuffers<F>,
        inputs: &[F],
        generator_count: usize,
        timing: &mut TimingTree,
    ) -> Result<ProofWithPublicInputs<F, C, D>>
    where
        C::Hasher: Hasher<F>,
        C::InnerHasher: Hasher<F>,
    {
        println!("---Start Proving---");
        let s = Instant::now();
        timed!(timing, &format!("run {} generators", generator_count), {
            WitnessService::upload_inputs(&mut workspace.witness_buffers, inputs);
            WitnessService::replay(&circuit.witness_plan);
        });
        println!("Time taken to generate witness: {:?}", s.elapsed());

        Self::prove_with_prepared_witness::<F, C, D>(
            circuit,
            workspace,
            proof_state,
            proof_out,
            timing,
        )
    }

    pub(crate) fn prove_with_prepared_witness<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        circuit: &mut GpuCircuitSetup<F, D>,
        workspace: &mut GpuWorkspace,
        proof_state: &mut GpuProofState,
        proof_out: &mut ProofAssemblyBuffers<F>,
        _timing: &mut TimingTree,
    ) -> Result<ProofWithPublicInputs<F, C, D>>
    where
        C::Hasher: Hasher<F>,
        C::InnerHasher: Hasher<F>,
    {
        let stream = CudaStream::default();
        let s = Instant::now();

        // First nongraph launch after a live GraphExec+sync is the C210 3.3s
        // tax (hash_pi n_pi=0, launch-side). Paid at build by hashing dummy
        // public inputs; prove hash_pi is then microseconds. Do not destroy
        // the graph here — extra FRI needs it, and Drop after verify is 2.5ms.
        let t = Instant::now();
        Self::hash_pi_new(circuit, workspace, proof_state).unwrap();
        println!("Time taken to hash_pi_new: {:?}", t.elapsed());

        WitnessService::release_uncaptured_after_replay(
            &mut circuit.witness_plan,
            &mut workspace.witness_buffers,
        );

        let post = overlay::post_witgen_bytes(
            circuit.cmd.degree(),
            D,
            circuit.cmd.config.fri_config.rate_bits,
            &circuit.cmd.fri_params.reduction_arity_bits,
        );
        let (free, _) = memory_get_info().unwrap();
        let live_graph = circuit.witness_plan.has_live_graph();
        let extra_fri = live_graph && free >= post.saturating_add(overlay::FRI_KEEP_SOA_SLACK);
        let overlay_soa = live_graph && !extra_fri && workspace.scratch.len() >= post;
        if extra_fri {
            println!(
                "Post-witgen path: extra FRI (live-graph, free={:.0} MiB post={:.0} MiB)",
                free as f64 / (1024.0 * 1024.0),
                post as f64 / (1024.0 * 1024.0)
            );
        } else if overlay_soa {
            println!(
                "Post-witgen path: overlay SoA (live-graph, extra FRI would not fit, free={:.0} MiB post={:.0} MiB)",
                free as f64 / (1024.0 * 1024.0),
                post as f64 / (1024.0 * 1024.0)
            );
        } else {
            if live_graph {
                let t = Instant::now();
                WitnessService::release_after_replay(&mut circuit.witness_plan);
                println!(
                    "Time taken to release_after_replay (SoA+FRI would not fit, free={:.0} MiB post={:.0} MiB): {:?}",
                    free as f64 / (1024.0 * 1024.0),
                    post as f64 / (1024.0 * 1024.0),
                    t.elapsed()
                );
            } else {
                WitnessService::release_after_replay(&mut circuit.witness_plan);
            }
            println!(
                "Post-witgen path: recycle SoA (streamed or extra FRI miss, free={:.0} MiB post={:.0} MiB)",
                free as f64 / (1024.0 * 1024.0),
                post as f64 / (1024.0 * 1024.0)
            );
        }

        let t = Instant::now();
        workspace.bind_post_witgen_views(
            &mut proof_state.fri,
            circuit.cmd.degree(),
            D,
            circuit.cmd.config.fri_config.rate_bits,
            &circuit.cmd.fri_params.reduction_arity_bits,
            extra_fri,
            overlay_soa,
        );
        println!("Time taken to bind_post_witgen_views: {:?}", t.elapsed());

        Self::commit(circuit, workspace, CommitmentStage::Wires);
        println!("Time taken to cm1: {:?}", s.elapsed());
        if cfg!(feature = "verbose") {
            let cap1: MerkleCap<F, <C as GenericConfig<D>>::Hasher> =
                Self::get_cap_device(circuit, workspace, CommitmentStage::Wires);
            println!("cap1: {:?}", cap1);
            circuit.challenger.show_sponge();
        }

        let s = Instant::now();

        if cfg!(feature = "verbose") {
            let public_inputs = &proof_state.public_inputs;
            let mut public_input_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(
                public_inputs.values().len(),
                CudaHostAllocFlags::DEFAULT,
            )
            .unwrap();
            memory_copy_async(&mut public_input_host, public_inputs.values(), &stream).unwrap();
            stream.synchronize().unwrap();
            println!("public inputs: {:?}", public_input_host.to_vec());
        }
        if cfg!(feature = "verbose") {
            circuit.challenger.show_sponge();
        }

        TranscriptStep::observe_elements(
            &mut circuit.challenger,
            proof_state.public_inputs.hash_mut(),
            4,
        )
        .unwrap();
        if cfg!(feature = "verbose") {
            circuit.challenger.show_sponge();
        }
        TranscriptStep::observe_cap(
            &mut circuit.challenger,
            workspace
                .commitments
                .tree_device_mut(CommitmentStage::Wires),
            circuit.cmd.config.fri_config.cap_height,
        )
        .unwrap();
        if cfg!(feature = "verbose") {
            circuit.challenger.show_sponge();
        }

        let num_challenges = circuit.cmd.config.num_challenges;
        TranscriptStep::sample_into(
            &mut circuit.challenger,
            proof_state.partial_products.betas_mut(),
            num_challenges,
        )
        .unwrap();
        TranscriptStep::sample_into(
            &mut circuit.challenger,
            proof_state.partial_products.gammas_mut(),
            num_challenges,
        )
        .unwrap();

        Self::zp(circuit, workspace, proof_state);
        println!("Time taken to pp: {:?}", s.elapsed());

        if cfg!(feature = "verbose") {
            circuit.challenger.show_sponge();
            let mut bt_host =
                HostAllocation::<GoldilocksFieldBoojum>::alloc(2, CudaHostAllocFlags::DEFAULT)
                    .unwrap();
            let mut gm_host =
                HostAllocation::<GoldilocksFieldBoojum>::alloc(2, CudaHostAllocFlags::DEFAULT)
                    .unwrap();
            memory_copy_async(&mut bt_host, proof_state.partial_products.betas(), &stream).unwrap();
            memory_copy_async(&mut gm_host, proof_state.partial_products.gammas(), &stream)
                .unwrap();
            stream.synchronize().unwrap();
            println!(
                "beta: {:?}, gamma: {:?}",
                bt_host.to_vec(),
                gm_host.to_vec()
            );
        }

        debug_dump::write_json("zs.json", || -> Result<Vec<u8>> {
            let partial_products = circuit.layout.fp_range(PolySegment::PartialProducts);
            let partial_products_len = circuit
                .layout
                .polynomial_count(PolySegment::PartialProducts)
                * circuit.cmd.degree();
            debug_assert_eq!(partial_products.len(), partial_products_len);
            let zs_flatten = debug_dump::capture_cuda_bytes(
                &workspace.fp_inputs,
                partial_products.start,
                partial_products_len,
                &stream,
            )?;
            Ok(serde_json::to_vec_pretty(&zs_flatten)?)
        });

        let s = Instant::now();
        Self::commit(circuit, workspace, CommitmentStage::PartialProducts);
        println!("Time taken to cm2: {:?}", s.elapsed());
        if cfg!(feature = "verbose") {
            let cap2: MerkleCap<F, <C as GenericConfig<D>>::Hasher> =
                Self::get_cap_device(circuit, workspace, CommitmentStage::PartialProducts);
            println!("cap2: {:?}", cap2);
            circuit.challenger.show_sponge();
        }

        let s = Instant::now();
        TranscriptStep::observe_cap(
            &mut circuit.challenger,
            workspace
                .commitments
                .tree_device_mut(CommitmentStage::PartialProducts),
            circuit.cmd.config.fri_config.cap_height,
        )
        .unwrap();

        TranscriptStep::sample_into(
            &mut circuit.challenger,
            proof_state.quotient_state.alphas_mut(),
            circuit.cmd.config.num_challenges,
        )
        .unwrap();

        Self::compute_quotient(circuit, workspace, proof_state);
        println!("Time taken to quotient: {:?}", s.elapsed());
        if cfg!(feature = "verbose") {
            let mut alphas =
                HostAllocation::<GoldilocksFieldBoojum>::alloc(2, CudaHostAllocFlags::DEFAULT)
                    .unwrap();
            memory_copy_async(&mut alphas, proof_state.quotient_state.alphas(), &stream).unwrap();
            stream.synchronize().unwrap();
            let mut res_vec: Vec<GoldilocksFieldBoojum> = alphas.to_vec();
            println!("alpha: {:?}", res_vec);
        }

        let s = Instant::now();
        Self::commit(circuit, workspace, CommitmentStage::Quotients);
        println!("Time taken to cm3: {:?}", s.elapsed());

        if cfg!(feature = "verbose") {
            let cap3: MerkleCap<F, <C as GenericConfig<D>>::Hasher> =
                Self::get_cap_device(circuit, workspace, CommitmentStage::Quotients);
            println!("cap3: {:?}", cap3);
        }

        let s = Instant::now();
        TranscriptStep::observe_cap(
            &mut circuit.challenger,
            workspace
                .commitments
                .tree_device_mut(CommitmentStage::Quotients),
            circuit.cmd.config.fri_config.cap_height,
        )
        .unwrap();

        TranscriptStep::sample_into(
            &mut circuit.challenger,
            proof_state.openings.zeta_g_mut(),
            D,
        )
        .unwrap();

        if cfg!(feature = "verbose") {
            let openings = &proof_state.openings;
            let mut zeta_g = HostAllocation::<GoldilocksFieldBoojum>::alloc(
                openings.zeta_g().len(),
                CudaHostAllocFlags::DEFAULT,
            )
            .unwrap();
            memory_copy_async(&mut zeta_g, openings.zeta_g(), &stream).unwrap();
            stream.synchronize().unwrap();
            println!("zeta_g: {:?}", zeta_g.to_vec());
        }

        Self::get_openings(circuit, workspace, proof_state, proof_out);
        println!("Time taken to openings: {:?}", s.elapsed());

        let s = Instant::now();
        let s0 = Instant::now();
        let num_openings = proof_state.openings.values().len();
        if cfg!(feature = "verbose") {
            println!("Before observing openings:");
            circuit.challenger.show_sponge();
            circuit.challenger.show_ibuf();
        }
        TranscriptStep::observe_elements(
            &mut circuit.challenger,
            proof_state.openings.values_mut(),
            num_openings,
        )
        .unwrap();
        if cfg!(feature = "verbose") {
            println!("After observing {} openings:", num_openings);
            circuit.challenger.show_sponge();
        }
        println!(" observe openings: {:?}", s0.elapsed());
        TranscriptStep::sample_into(
            &mut circuit.challenger,
            proof_state.final_poly.alpha_mut(),
            D,
        )
        .unwrap();
        Self::get_final_poly(circuit, workspace, proof_state);
        println!("Time taken to final poly: {:?}", s.elapsed());
        if cfg!(feature = "verbose") {
            let final_poly = &proof_state.final_poly;
            let mut fp_alpha = HostAllocation::<GoldilocksFieldBoojum>::alloc(
                final_poly.alpha().len(),
                CudaHostAllocFlags::DEFAULT,
            )
            .unwrap();
            memory_copy_async(&mut fp_alpha, final_poly.alpha(), &stream).unwrap();
            stream.synchronize().unwrap();
            println!("fp_alpha: {:?}", fp_alpha.to_vec());
        }
        debug_dump::write_json("fp.json", || -> Result<Vec<u8>> {
            let fp = debug_dump::capture_cuda(proof_state.final_poly.coefficients(), &stream)?;
            Ok(serde_json::to_vec(&fp)?)
        });

        let s = Instant::now();
        Self::fri_prove(circuit, workspace, proof_state, proof_out);
        println!("Time taken to fri: {:?}", s.elapsed());

        let s = Instant::now();
        let proof = Self::get_proof::<F, C, D>(circuit, workspace, proof_state, proof_out);
        println!("Time get proof: {:?}", s.elapsed());
        debug_dump::write_text("proof.json", || format!("{:?}", proof));

        proof
    }

    pub(crate) fn commit<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        workspace: &mut GpuWorkspace,
        stage: CommitmentStage,
    ) {
        let config = Self::commitment_config(circuit);
        if stage != CommitmentStage::Constants {
            workspace.commitments.ensure_stage(stage);
        }
        match stage {
            CommitmentStage::Constants => CommitmentService::commit_constants(
                &mut workspace.commitments,
                &circuit.layout,
                &mut workspace.fp_inputs,
                &mut workspace.ldes,
                config,
            ),
            CommitmentStage::Wires => CommitmentService::commit_wires(
                &mut workspace.commitments,
                &circuit.layout,
                &mut workspace.fp_inputs,
                &mut workspace.ldes,
                config,
            ),
            CommitmentStage::PartialProducts => CommitmentService::commit_partial_products(
                &mut workspace.commitments,
                &circuit.layout,
                &mut workspace.fp_inputs,
                &mut workspace.ldes,
                config,
            ),
            CommitmentStage::Quotients => CommitmentService::commit_quotients(
                &mut workspace.commitments,
                &circuit.layout,
                &mut workspace.fp_inputs,
                &mut workspace.ldes,
                config,
            ),
        }
    }

    pub(crate) fn get_cap<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        workspace: &GpuWorkspace,
        stage: CommitmentStage,
    ) -> MerkleCap<F, H> {
        CommitmentService::get_cap(
            &workspace.commitments,
            stage,
            circuit.cmd.config.fri_config.cap_height,
        )
    }

    pub(crate) fn get_cap_device<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        workspace: &GpuWorkspace,
        stage: CommitmentStage,
    ) -> MerkleCap<F, H> {
        CommitmentService::get_cap_device(
            &workspace.commitments,
            stage,
            circuit.cmd.config.fri_config.cap_height,
        )
    }

    pub(crate) fn get_caps<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        workspace: &GpuWorkspace,
    ) -> Result<Vec<MerkleCap<F, H>>> {
        CommitmentService::get_proof_caps(
            &workspace.commitments,
            circuit.cmd.config.fri_config.cap_height,
        )
    }

    pub(crate) fn zp<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        workspace: &mut GpuWorkspace,
        proof_state: &mut GpuProofState,
    ) {
        PartialProductsService::zp(
            &mut proof_state.partial_products,
            &circuit.layout,
            &mut workspace.fp_inputs,
            circuit.quotient_static.subgroup(),
            circuit.quotient_static.k_is(),
        );
    }

    pub(crate) fn hash_pi_new<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        workspace: &GpuWorkspace,
        proof_state: &mut GpuProofState,
    ) -> CudaResult<()> {
        PublicInputsService::hash(
            &mut proof_state.public_inputs,
            &workspace.witness_buffers.witness,
            &workspace.witness_buffers.public_input_indices,
            circuit.cmd.num_public_inputs,
        )
    }

    pub(crate) fn compute_quotient<F: RichField + Extendable<D>, const D: usize>(
        circuit_setup: &GpuCircuitSetup<F, D>,
        workspace: &mut GpuWorkspace,
        proof_state: &GpuProofState,
    ) {
        let circuit = QuotientCircuitData::from_common(&circuit_setup.cmd);
        let inputs = QuotientInputs {
            circuit,
            layout: &circuit_setup.layout,
            static_data: &circuit_setup.quotient_static,
            state: &proof_state.quotient_state,
            public_inputs: &proof_state.public_inputs,
            partial_products: &proof_state.partial_products,
        };
        let quotient_workspace = QuotientWorkspace {
            fp_inputs: &mut workspace.fp_inputs,
            ldes: &workspace.ldes,
        };
        QuotientService::compute(inputs, quotient_workspace);
    }

    pub(crate) fn get_openings<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        workspace: &GpuWorkspace,
        proof_state: &mut GpuProofState,
        proof_out: &mut ProofAssemblyBuffers<F>,
    ) {
        let config = Self::opening_config(circuit);
        OpeningsService::compute(
            &mut proof_state.openings,
            &workspace.fp_inputs,
            proof_out.openings_mut(),
            config,
        );
    }

    pub(crate) fn get_final_poly<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        workspace: &GpuWorkspace,
        proof_state: &mut GpuProofState,
    ) {
        proof_state.fri.ensure_commit_allocated();
        let config = Self::final_poly_config(circuit);
        FinalPolyService::compute(
            &mut proof_state.final_poly,
            proof_state.openings.zeta_g(),
            &workspace.fp_inputs,
            proof_state.fri.first_commit_leaves_mut(),
            config,
        );
    }

    pub(crate) fn final_poly_coset_fft<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        proof_state: &mut GpuProofState,
    ) -> CudaResult<()> {
        let config = Self::final_poly_config(circuit);
        FinalPolyService::coset_fft(
            &mut proof_state.final_poly,
            proof_state.fri.first_commit_leaves_mut(),
            config,
        )
    }

    pub(crate) fn fri_committed_trees<F: RichField + Extendable<D>, const D: usize>(
        circuit: &mut GpuCircuitSetup<F, D>,
        proof_state: &mut GpuProofState,
        proof_out: &mut ProofAssemblyBuffers<F>,
    ) {
        let mut shift = GoldilocksFieldBoojum::MULTIPLICATIVE_GROUP_GENERATOR;
        let stream = CudaStream::default();
        let final_poly = &proof_state.final_poly;
        let commit = proof_state.fri.commit_mut();
        let mut layers_count =
            circuit.cmd.degree_bits() + circuit.cmd.config.fri_config.rate_bits + 1;
        let mut degree = circuit.cmd.degree();
        let mut degree_bits = circuit.cmd.degree_bits();
        let mut flag = debug_dump::enabled();
        for (i, arity_bits) in circuit
            .cmd
            .fri_params
            .reduction_arity_bits
            .iter()
            .enumerate()
        {
            let arity = 1 << arity_bits;
            degree /= arity;
            degree_bits -= arity_bits;
            layers_count -= arity_bits;

            FriCommitService::commit_tree(
                commit,
                i,
                *arity_bits,
                layers_count,
                circuit.cmd.config.fri_config.rate_bits,
                &stream,
            );

            let offset = TranscriptStep::cap_offset(
                commit.tree_digests(i).len(),
                circuit.cmd.config.fri_config.cap_height,
            );
            TranscriptStep::observe_cap(
                &mut circuit.challenger,
                commit.tree_digests_mut(i),
                circuit.cmd.config.fri_config.cap_height,
            )
            .unwrap();
            TranscriptStep::sample_into(&mut circuit.challenger, commit.beta_mut(), D).unwrap();

            let cap_len = TranscriptStep::cap_len(circuit.cmd.config.fri_config.cap_height);
            FriCommitService::copy_cap(
                commit,
                proof_out.fri_commit_phase_caps_mut(),
                i,
                offset,
                cap_len,
                &stream,
            );
            if cfg!(feature = "verbose") {
                FriCommitService::print_round_debug(commit, i, offset, &stream);
                circuit.challenger.show_sponge();
            }

            FriCommitService::fold_round(
                commit,
                final_poly.coefficients().as_ptr(),
                i,
                degree,
                degree_bits,
                arity,
                circuit.cmd.config.fri_config.rate_bits,
                &mut shift,
                i == circuit.cmd.fri_params.reduction_arity_bits.len() - 1,
                &stream,
            );

            if flag {
                FriCommitService::dump_first_round(commit, &stream);
                flag = false;
            }
        }

        let num_arity = circuit.cmd.fri_params.reduction_arity_bits.len();
        let final_round = num_arity - 1;
        FriCommitService::copy_final_coefficients(
            commit,
            proof_out.fri_final_coefficients_mut(),
            final_round,
            &stream,
        );

        let num_final_coeffs =
            commit.coefficients(final_round).len() >> circuit.cmd.config.fri_config.rate_bits;
        TranscriptStep::observe_elements(
            &mut circuit.challenger,
            commit.coefficients_mut(final_round),
            num_final_coeffs,
        )
        .unwrap();
    }

    pub(crate) fn fri_proof_of_work_new<F: RichField + Extendable<D>, const D: usize>(
        circuit: &mut GpuCircuitSetup<F, D>,
        proof_state: &mut GpuProofState,
        proof_out: &mut ProofAssemblyBuffers<F>,
    ) {
        let min_leading_zeros =
            circuit.cmd.fri_params.config.proof_of_work_bits + (64 - F::order().bits()) as u32;
        let inputs = FriPowInputs::new(
            &circuit.challenger.input_buffer,
            &circuit.challenger.sponge_state,
            circuit.challenger.in_idx,
            min_leading_zeros,
        );
        FriPowService::compute(
            proof_state.fri.pow_mut(),
            proof_out.fri_pow_witness_mut(),
            inputs,
        );

        TranscriptStep::observe_elements(
            &mut circuit.challenger,
            proof_state.fri.pow_mut().witness_mut(),
            1,
        )
        .unwrap();
        let response = circuit.challenger.get_n_challenges(1);
        if cfg!(feature = "verbose") {
            println!("min leading zeros: {}", min_leading_zeros);
            println!("pow_response: {:?}", response);
        }
    }

    pub(crate) fn fri_prover_query_steps<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        proof_state: &mut GpuProofState,
        proof_out: &mut ProofAssemblyBuffers<F>,
    ) {
        let config = Self::fri_query_config(circuit);
        let reduction_arity_bits = &circuit.cmd.fri_params.reduction_arity_bits;
        let (commit, query) = proof_state.fri.commit_and_query_mut();
        FriQueryService::query_steps(
            commit,
            query,
            proof_out.fri_step_query_output_mut(),
            reduction_arity_bits,
            config,
        );
    }

    pub(crate) fn fri_prover_query_rounds<F: RichField + Extendable<D>, const D: usize>(
        circuit: &mut GpuCircuitSetup<F, D>,
        workspace: &GpuWorkspace,
        proof_state: &mut GpuProofState,
        proof_out: &mut ProofAssemblyBuffers<F>,
    ) {
        let config = Self::fri_query_config(circuit);
        TranscriptStep::sample_into(
            &mut circuit.challenger,
            proof_state.fri.query_mut().round_challenges_mut(),
            circuit.cmd.fri_params.config.num_query_rounds,
        )
        .unwrap();

        let commitment_tree_digests = workspace.commitments.tree_device_ptrs().to_vec();
        FriQueryService::initial_proofs(
            proof_state.fri.query_mut(),
            proof_out.fri_initial_query_output_mut(),
            &circuit.layout,
            &workspace.fp_inputs,
            &workspace.ldes,
            &commitment_tree_digests,
            config,
        );
        Self::fri_prover_query_steps(circuit, proof_state, proof_out);
    }

    pub(crate) fn fri_prove<F: RichField + Extendable<D>, const D: usize>(
        circuit: &mut GpuCircuitSetup<F, D>,
        workspace: &GpuWorkspace,
        proof_state: &mut GpuProofState,
        proof_out: &mut ProofAssemblyBuffers<F>,
    ) {
        let s = Instant::now();
        Self::fri_committed_trees(circuit, proof_state, proof_out);
        println!(" committed trees: {:?}", s.elapsed());

        let s = Instant::now();
        Self::fri_proof_of_work_new(circuit, proof_state, proof_out);
        println!(" proof of work: {:?}", s.elapsed());

        let s = Instant::now();
        Self::fri_prover_query_rounds(circuit, workspace, proof_state, proof_out);
        println!(" prover query rounds: {:?}", s.elapsed());
    }

    pub(crate) fn get_opening_set<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        proof_state: &GpuProofState,
        proof_out: &ProofAssemblyBuffers<F>,
    ) -> OpeningSet<F, D> {
        let config = Self::proof_assembly_config(circuit, proof_state);
        ProofAssemblyService::get_opening_set(proof_out, &circuit.layout, &config)
    }

    pub(crate) fn get_fri_proof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
        proof_state: &GpuProofState,
        proof_out: &ProofAssemblyBuffers<F>,
    ) -> FriProof<F, H, D> {
        let config = Self::proof_assembly_config(circuit, proof_state);
        ProofAssemblyService::get_fri_proof(proof_out, &circuit.layout, &config)
    }

    pub(crate) fn get_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        circuit: &GpuCircuitSetup<F, D>,
        workspace: &GpuWorkspace,
        proof_state: &GpuProofState,
        proof_out: &ProofAssemblyBuffers<F>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let config = Self::proof_assembly_config(circuit, proof_state);
        ProofAssemblyService::get_final_proof::<F, C, D>(
            proof_out,
            proof_state.public_inputs.values(),
            &workspace.commitments,
            &circuit.layout,
            &config,
        )
    }

    fn commitment_config<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
    ) -> CommitmentConfig {
        CommitmentConfig::new(
            circuit.cmd.degree_bits(),
            circuit.cmd.degree(),
            circuit.cmd.config.fri_config.rate_bits,
            circuit.cmd.quotient_degree_factor,
            circuit.cmd.config.num_challenges,
        )
    }

    fn opening_config<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
    ) -> OpeningConfig {
        OpeningConfig::new(
            circuit.cmd.degree_bits(),
            circuit.layout.num_polys(),
            circuit.cmd.config.num_challenges,
            circuit
                .layout
                .polynomial_offset(PolySegment::PartialProducts),
        )
    }

    fn final_poly_config<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
    ) -> FinalPolyConfig {
        FinalPolyConfig::new(
            circuit.cmd.degree(),
            circuit.cmd.degree_bits(),
            circuit.cmd.config.fri_config.rate_bits,
            circuit.layout.num_polys(),
            circuit.cmd.config.num_challenges,
            circuit
                .layout
                .polynomial_offset(PolySegment::PartialProducts),
        )
    }

    fn fri_query_config<F: RichField + Extendable<D>, const D: usize>(
        circuit: &GpuCircuitSetup<F, D>,
    ) -> FriQueryConfig {
        FriQueryConfig::new(
            circuit.cmd.fri_params.config.num_query_rounds,
            circuit.cmd.degree_bits(),
            circuit.cmd.config.fri_config.rate_bits,
            circuit.cmd.config.fri_config.cap_height,
        )
    }

    fn proof_assembly_config<'a, F: RichField + Extendable<D>, const D: usize>(
        circuit: &'a GpuCircuitSetup<F, D>,
        proof_state: &GpuProofState,
    ) -> ProofAssemblyConfig<'a> {
        ProofAssemblyConfig {
            num_public_inputs: circuit.cmd.num_public_inputs,
            constants_range: circuit.cmd.constants_range(),
            sigmas_range: circuit.cmd.sigmas_range(),
            num_challenges: circuit.cmd.config.num_challenges,
            degree_bits: circuit.cmd.degree_bits(),
            rate_bits: circuit.cmd.config.fri_config.rate_bits,
            cap_height: circuit.cmd.config.fri_config.cap_height,
            num_query_rounds: circuit.cmd.config.fri_config.num_query_rounds,
            reduction_arity_bits: &circuit.cmd.fri_params.reduction_arity_bits,
            arity_sum: proof_state.fri.query().arity_sum(),
            total_step_siblings: proof_state.fri.query().total_step_siblings(),
        }
    }
}
