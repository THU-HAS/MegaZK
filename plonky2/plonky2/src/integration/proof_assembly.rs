//! Host proof-output ownership and proof assembly.
//!
//! This module owns every long-lived host buffer populated by the GPU proof phases. It does not
//! know phase order, operate the transcript, or depend on the compatibility facade.

use std::mem;
use std::ops::Range;

use anyhow::Result;
use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use cudart::stream::CudaStream;
use plonky2_field::extension::Extendable;
use plonky2_field::polynomial::PolynomialCoeffs;

use super::commitments::{CommitmentBuffers, CommitmentService};
use super::layout::{PolyLayout, PolySegment};
use crate::fri::proof::{FriInitialTreeProof, FriProof, FriQueryRound, FriQueryStep};
use crate::hash::hash_types::RichField;
use crate::hash::merkle_proofs::MerkleProof;
use crate::hash::merkle_tree::MerkleCap;
use crate::plonk::config::{GenericConfig, Hasher};
use crate::plonk::proof::{OpeningSet, Proof, ProofWithPublicInputs};

pub(crate) struct ProofAssemblyBuffers<F> {
    fri_initial_leaves: Vec<HostAllocation<GoldilocksFieldBoojum>>,
    openings: HostAllocation<GoldilocksFieldBoojum>,
    fri_final_coefficients: HostAllocation<GoldilocksFieldBoojum>,
    fri_step_evaluations: HostAllocation<GoldilocksFieldBoojum>,
    fri_step_siblings: HostAllocation<GoldilocksFieldBoojum>,
    fri_initial_siblings: HostAllocation<GoldilocksFieldBoojum>,
    fri_commit_phase_caps: Vec<Vec<GoldilocksFieldBoojum>>,
    fri_pow_witness: F,
}

impl<F> ProofAssemblyBuffers<F> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        fri_initial_leaves: Vec<HostAllocation<GoldilocksFieldBoojum>>,
        openings: HostAllocation<GoldilocksFieldBoojum>,
        fri_final_coefficients: HostAllocation<GoldilocksFieldBoojum>,
        fri_step_evaluations: HostAllocation<GoldilocksFieldBoojum>,
        fri_step_siblings: HostAllocation<GoldilocksFieldBoojum>,
        fri_initial_siblings: HostAllocation<GoldilocksFieldBoojum>,
        fri_commit_phase_caps: Vec<Vec<GoldilocksFieldBoojum>>,
        fri_pow_witness: F,
    ) -> Self {
        Self {
            fri_initial_leaves,
            openings,
            fri_final_coefficients,
            fri_step_evaluations,
            fri_step_siblings,
            fri_initial_siblings,
            fri_commit_phase_caps,
            fri_pow_witness,
        }
    }

    #[inline]
    pub(crate) fn openings_mut(&mut self) -> &mut HostAllocation<GoldilocksFieldBoojum> {
        &mut self.openings
    }

    #[inline]
    pub(crate) fn fri_final_coefficients_mut(
        &mut self,
    ) -> &mut HostAllocation<GoldilocksFieldBoojum> {
        &mut self.fri_final_coefficients
    }

    #[inline]
    pub(crate) fn fri_commit_phase_caps_mut(&mut self) -> &mut Vec<Vec<GoldilocksFieldBoojum>> {
        &mut self.fri_commit_phase_caps
    }

    #[inline]
    pub(crate) fn fri_pow_witness_mut(&mut self) -> &mut F {
        &mut self.fri_pow_witness
    }

    pub(crate) fn fri_initial_query_output_mut(&mut self) -> FriInitialQueryOutputMut<'_> {
        FriInitialQueryOutputMut {
            leaves: &mut self.fri_initial_leaves,
            siblings: &mut self.fri_initial_siblings,
        }
    }

    pub(crate) fn fri_step_query_output_mut(&mut self) -> FriStepQueryOutputMut<'_> {
        FriStepQueryOutputMut {
            evaluations: &mut self.fri_step_evaluations,
            siblings: &mut self.fri_step_siblings,
        }
    }
}

pub(crate) struct FriInitialQueryOutputMut<'a> {
    pub(crate) leaves: &'a mut [HostAllocation<GoldilocksFieldBoojum>],
    pub(crate) siblings: &'a mut HostAllocation<GoldilocksFieldBoojum>,
}

pub(crate) struct FriStepQueryOutputMut<'a> {
    pub(crate) evaluations: &'a mut HostAllocation<GoldilocksFieldBoojum>,
    pub(crate) siblings: &'a mut HostAllocation<GoldilocksFieldBoojum>,
}

pub(crate) struct ProofAssemblyConfig<'a> {
    pub(crate) num_public_inputs: usize,
    pub(crate) constants_range: Range<usize>,
    pub(crate) sigmas_range: Range<usize>,
    pub(crate) num_challenges: usize,
    pub(crate) degree_bits: usize,
    pub(crate) rate_bits: usize,
    pub(crate) cap_height: usize,
    pub(crate) num_query_rounds: usize,
    pub(crate) reduction_arity_bits: &'a [usize],
    pub(crate) arity_sum: usize,
    pub(crate) total_step_siblings: usize,
}

pub(crate) struct ProofAssemblyService;

impl ProofAssemblyService {
    pub(crate) fn get_opening_set<F: RichField + Extendable<D>, const D: usize>(
        buffers: &ProofAssemblyBuffers<F>,
        layout: &PolyLayout,
        config: &ProofAssemblyConfig<'_>,
    ) -> OpeningSet<F, D> {
        let openings_vec = buffers.openings.to_vec();
        let mut openings_vec: Vec<F::Extension> = unsafe { mem::transmute(openings_vec) };
        unsafe { openings_vec.set_len(openings_vec.len() / D) };
        let offset = layout.polynomial_offset(PolySegment::PartialProducts);
        let num_const_sigmas = layout.polynomial_count(PolySegment::ConstantsSigmas);
        let num_partial_products = layout.polynomial_count(PolySegment::PartialProducts);
        let num_polys = layout.num_polys();
        OpeningSet {
            constants: openings_vec[config.constants_range.clone()].to_vec(),
            plonk_sigmas: openings_vec[config.sigmas_range.clone()].to_vec(),
            wires: openings_vec[num_const_sigmas..offset].to_vec(),
            plonk_zs: openings_vec[offset..(offset + config.num_challenges)].to_vec(),
            plonk_zs_next: openings_vec[num_polys..].to_vec(),
            partial_products: openings_vec
                [(offset + config.num_challenges)..(offset + num_partial_products)]
                .to_vec(),
            quotient_polys: openings_vec[(offset + num_partial_products)..num_polys].to_vec(),
            lookup_zs: vec![],
            lookup_zs_next: vec![],
        }
    }

    pub(crate) fn get_fri_proof<F: RichField + Extendable<D>, H: Hasher<F>, const D: usize>(
        buffers: &ProofAssemblyBuffers<F>,
        layout: &PolyLayout,
        config: &ProofAssemblyConfig<'_>,
    ) -> FriProof<F, H, D> {
        let _stream = CudaStream::default();
        let final_poly_vec = buffers.fri_final_coefficients.to_vec();
        let mut final_poly_vec: Vec<F::Extension> = unsafe { mem::transmute(final_poly_vec) };
        unsafe { final_poly_vec.set_len(final_poly_vec.len() / D) };
        final_poly_vec = final_poly_vec[..(final_poly_vec.len() >> config.rate_bits)].to_vec();
        let final_poly = PolynomialCoeffs::new(final_poly_vec);

        let len_cap = 4 << config.cap_height;
        let mut commit_phase_merkle_caps = Vec::<MerkleCap<F, H>>::new();
        for cap in &buffers.fri_commit_phase_caps {
            let mut cap_hash = Vec::<H::Hash>::new();
            for i in 0..(len_cap / 4) {
                let h: Vec<F> = unsafe {
                    mem::transmute(vec![
                        cap[4 * i],
                        cap[4 * i + 1],
                        cap[4 * i + 2],
                        cap[4 * i + 3],
                    ])
                };
                cap_hash.push(H::hash_or_noop(&h));
            }
            commit_phase_merkle_caps.push(MerkleCap(cap_hash));
        }

        let mut query_round_proofs = Vec::<FriQueryRound<F, H, D>>::new();
        for _ in 0..config.num_query_rounds {
            let fri_query_round = FriQueryRound {
                initial_trees_proof: FriInitialTreeProof {
                    evals_proofs: Vec::<(Vec<F>, MerkleProof<F, H>)>::new(),
                },
                steps: Vec::<FriQueryStep<F, H, D>>::new(),
            };
            query_round_proofs.push(fri_query_round);
        }

        let nums = PolySegment::ALL.map(|segment| layout.polynomial_count(segment));
        let num_layers = config.degree_bits + config.rate_bits - config.cap_height;
        for r in 0..4 {
            let r_start = r * 4 * num_layers;
            for i in 0..config.num_query_rounds {
                let q_start = i * 4 * 4 * num_layers;
                let mut siblings = Vec::<H::Hash>::new();
                for j in 0..num_layers {
                    let idx = q_start + r_start + 4 * j;
                    let h: Vec<F> = unsafe {
                        mem::transmute(vec![
                            buffers.fri_initial_siblings[idx],
                            buffers.fri_initial_siblings[idx + 1],
                            buffers.fri_initial_siblings[idx + 2],
                            buffers.fri_initial_siblings[idx + 3],
                        ])
                    };
                    siblings.push(H::hash_or_noop(&h));
                }
                let merkle_proof = MerkleProof { siblings };
                let initial_proof = (
                    unsafe {
                        mem::transmute(
                            buffers.fri_initial_leaves[r][(i * nums[r])..((i + 1) * nums[r])]
                                .to_vec(),
                        )
                    },
                    merkle_proof,
                );
                query_round_proofs[i]
                    .initial_trees_proof
                    .evals_proofs
                    .push(initial_proof);
            }
        }

        for i in 0..config.num_query_rounds {
            let q_start = i * config.total_step_siblings * 4;
            let q_start_1 = i * config.arity_sum;
            let mut arity_acc: usize = 0;
            let mut acc_step_siblings: usize = 0;
            let _stream = CudaStream::default();
            let mut current_step_siblings =
                config.degree_bits + config.rate_bits - config.cap_height;
            for (_k, arity_bits) in config.reduction_arity_bits.to_vec().iter().enumerate() {
                let r_start = acc_step_siblings * 4;
                let r_start_1 = arity_acc;
                let ari: usize = *arity_bits;
                current_step_siblings = current_step_siblings - ari;
                let mut siblings = Vec::<H::Hash>::new();
                for j in 0..current_step_siblings {
                    let idx = q_start + r_start + 4 * j;
                    let h: Vec<F> = unsafe {
                        mem::transmute(vec![
                            buffers.fri_step_siblings[idx],
                            buffers.fri_step_siblings[idx + 1],
                            buffers.fri_step_siblings[idx + 2],
                            buffers.fri_step_siblings[idx + 3],
                        ])
                    };
                    siblings.push(H::hash_or_noop(&h));
                }
                let merkle_proof = MerkleProof { siblings };
                let evals_vec = buffers.fri_step_evaluations.to_vec();
                let mut evals_vec: Vec<F::Extension> = unsafe { mem::transmute(evals_vec) };
                unsafe { evals_vec.set_len(evals_vec.len() / D) };
                let _num_polys = 1 << arity_bits;
                let evals = evals_vec
                    [(q_start_1 + r_start_1)..(q_start_1 + r_start_1 + (1 << arity_bits))]
                    .to_vec();
                let step_proof = FriQueryStep {
                    evals,
                    merkle_proof,
                };
                query_round_proofs[i].steps.push(step_proof);
                acc_step_siblings += current_step_siblings;
                arity_acc += 1 << ari;
            }
        }

        if cfg!(feature = "verbose") {
            println!("caps: {:?}", commit_phase_merkle_caps);
            println!("final_poly: {:?}", final_poly);
        }

        FriProof {
            commit_phase_merkle_caps,
            final_poly,
            query_round_proofs,
            pow_witness: buffers.fri_pow_witness,
        }
    }

    pub(crate) fn get_final_proof<
        F: RichField + Extendable<D>,
        C: GenericConfig<D, F = F>,
        const D: usize,
    >(
        buffers: &ProofAssemblyBuffers<F>,
        public_input_values: &DeviceAllocation<GoldilocksFieldBoojum>,
        commitments: &CommitmentBuffers,
        layout: &PolyLayout,
        config: &ProofAssemblyConfig<'_>,
    ) -> Result<ProofWithPublicInputs<F, C, D>> {
        let stream = CudaStream::default();
        let mut public_inputs = HostAllocation::<GoldilocksFieldBoojum>::alloc(
            config.num_public_inputs,
            CudaHostAllocFlags::DEFAULT,
        )
        .unwrap();
        memory_copy_async(&mut public_inputs, public_input_values, &stream).unwrap();
        stream.synchronize().unwrap();
        let public_inputs: Vec<F> = unsafe { mem::transmute(public_inputs.to_vec()) };
        let caps =
            CommitmentService::get_proof_caps::<F, C::Hasher>(commitments, config.cap_height)
                .unwrap();
        let opening_set = Self::get_opening_set(buffers, layout, config);
        let fri_proof = Self::get_fri_proof::<F, C::Hasher, D>(buffers, layout, config);

        let proof = Proof {
            wires_cap: caps[0].clone(),
            plonk_zs_partial_products_cap: caps[1].clone(),
            quotient_polys_cap: caps[2].clone(),
            openings: opening_set,
            opening_proof: fri_proof,
        };

        Ok(ProofWithPublicInputs {
            proof,
            public_inputs,
        })
    }
}
