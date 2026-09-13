//! Quotient phase orchestration.
//!
//! This module preserves quotient-static initialization and active partial quotient phase ordering.
//! It does not own GPU buffers, prepare witness/public-input metadata, interact with transcript
//! state, initialize opening points, or commit output.

use alloc::vec::Vec;
use std::mem;
use std::time::Instant;

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use boojum::field::{Field as FieldBoojum, PrimeField};
use boojum_cuda::integration::batch_mul_exp;
use boojum_cuda::ntt::batch_ntt_internal;
use cudart::cuda_kernel;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::memory::memory_copy_async;
use cudart::result::CudaResult;
use cudart::stream::CudaStream;

use super::evaluator::QuotientEvaluator;
use super::{QuotientInitInputs, QuotientInitWorkspace, QuotientInputs, QuotientWorkspace};
use crate::field::extension::Extendable;
use crate::hash::hash_types::RichField;
use crate::util::debug_dump;

cuda_kernel!(
    Init,
    init_kernel,
    points: *mut GoldilocksFieldBoojum,
    z_h_coset: *mut GoldilocksFieldBoojum,
    k_is: *mut GoldilocksFieldBoojum,
    degree_bits: u32,
    quotient_degree_bits: u32,
    num_routed_wires: u32,
);

init_kernel!(init_q);

pub(crate) struct QuotientService;

impl QuotientService {
    pub(crate) fn initialize<F, const D: usize>(
        inputs: QuotientInitInputs<'_, F>,
        workspace: QuotientInitWorkspace<'_>,
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        F: RichField + Extendable<D>,
    {
        let sbg: Vec<GoldilocksFieldBoojum> = unsafe { mem::transmute(inputs.subgroup.to_vec()) };
        memory_copy_async(workspace.static_data.subgroup_mut(), &sbg, stream).unwrap();
        let threads: u32 = 128;
        let point_per_thread = 32;
        let n = workspace.static_data.points().len() as u32;
        let blocks = (n + threads * point_per_thread - 1) / (threads * point_per_thread);
        let points_ptr = workspace.static_data.points_mut().as_mut_ptr();
        let z_h_coset_ptr = workspace.static_data.z_h_coset_mut().as_mut_ptr();
        let k_is_ptr = workspace.static_data.k_is_mut().as_mut_ptr();
        stream.synchronize().unwrap();

        let config = CudaLaunchConfig::basic(blocks, threads, stream);
        let args = InitArguments::new(
            points_ptr,
            z_h_coset_ptr,
            k_is_ptr,
            inputs.degree_bits as u32,
            workspace.static_data.quotient_degree_bits() as u32,
            inputs.num_routed_wires as u32,
        );
        InitFunction(init_q).launch(&config, &args)
    }

    pub(crate) fn compute<F, const D: usize>(
        inputs: QuotientInputs<'_, F, D>,
        mut workspace: QuotientWorkspace<'_>,
    ) where
        F: RichField + Extendable<D>,
    {
        let gate_names: Vec<_> = inputs
            .circuit
            .gates
            .iter()
            .map(|gate| gate.0.id())
            .collect();
        let gate_num_constraints: Vec<usize> = inputs
            .circuit
            .gates
            .iter()
            .map(|gate| gate.0.num_constraints())
            .collect();

        let degree_bits = inputs.circuit.degree_bits as u32;
        let rate_bits = inputs.circuit.rate_bits as u32;
        let degree = 1 << degree_bits;
        let rate = 1 << rate_bits;
        let mut mul_val = GoldilocksFieldBoojum::MULTIPLICATIVE_GROUP_GENERATOR;
        let lde_twiddle_factor = GoldilocksFieldBoojum::RADIX_2_SUBGROUP_GENERATOR
            .pow_u64(1 << (32 - degree_bits - rate_bits));
        let stream = CudaStream::default();
        let inputs_offsets: Vec<usize> = crate::integration::layout::PolySegment::ALL[..3]
            .iter()
            .map(|&segment| {
                inputs
                    .layout
                    .fp_offset_after_ldes(segment, degree_bits as usize)
            })
            .collect();
        let num_ntts: Vec<usize> = crate::integration::layout::PolySegment::ALL[..3]
            .iter()
            .map(|&segment| inputs.layout.non_lde_polynomial_count(segment))
            .collect();

        for i in 0..rate as usize {
            let s = Instant::now();
            for j in 0..3 {
                if num_ntts[j] > 0 {
                    batch_mul_exp(
                        workspace
                            .fp_inputs
                            .as_mut_ptr()
                            .wrapping_add(inputs_offsets[j]),
                        degree_bits,
                        num_ntts[j] as u32,
                        degree,
                        mul_val,
                        &stream,
                    )
                    .unwrap();
                    batch_ntt_internal(
                        workspace.fp_inputs.as_ptr().wrapping_add(inputs_offsets[j]),
                        workspace
                            .fp_inputs
                            .as_mut_ptr()
                            .wrapping_add(inputs_offsets[j]),
                        degree_bits,
                        num_ntts[j] as u32,
                        degree,
                        degree,
                        false,
                        false,
                        0,
                        0,
                        false,
                        &stream,
                    )
                    .unwrap();
                }
            }
            stream.synchronize().unwrap();

            let i_r = i.reverse_bits() >> (64 - rate_bits);
            let results_offset = inputs.layout.fp_offset_within(
                crate::integration::layout::PolySegment::Quotients,
                i_r << degree_bits,
            );
            let results_ptr = workspace
                .fp_inputs
                .as_mut_ptr()
                .wrapping_add(results_offset);
            Self::compute_part(&inputs, &mut workspace, i, &gate_names);

            let s = Instant::now();
            for j in 0..3 {
                if num_ntts[j] > 0 {
                    batch_ntt_internal(
                        workspace.fp_inputs.as_ptr().wrapping_add(inputs_offsets[j]),
                        workspace
                            .fp_inputs
                            .as_mut_ptr()
                            .wrapping_add(inputs_offsets[j]),
                        degree_bits,
                        num_ntts[j] as u32,
                        degree,
                        degree,
                        true,
                        true,
                        0,
                        0,
                        false,
                        &stream,
                    )
                    .unwrap();
                }
            }
            stream.synchronize().unwrap();

            mul_val = lde_twiddle_factor;
        }

        mul_val = (GoldilocksFieldBoojum::RADIX_2_SUBGROUP_GENERATOR
            .pow_u64((rate - 1) << (32 - degree_bits - rate_bits))
            * GoldilocksFieldBoojum::MULTIPLICATIVE_GROUP_GENERATOR)
            .inverse()
            .expect("Failed to calculate inverse!");
        let s = Instant::now();
        for j in 0..3 {
            if num_ntts[j] > 0 {
                batch_mul_exp(
                    workspace
                        .fp_inputs
                        .as_mut_ptr()
                        .wrapping_add(inputs_offsets[j]),
                    degree_bits,
                    num_ntts[j] as u32,
                    degree,
                    mul_val,
                    &stream,
                )
                .unwrap();
            }
        }
        stream.synchronize().unwrap();

        debug_dump::write_json(
            "quotient_full_partial.json",
            || -> anyhow::Result<Vec<u8>> {
                let len = inputs.circuit.num_challenges
                    << (inputs.circuit.degree_bits + inputs.circuit.rate_bits);
                let stream = CudaStream::default();
                let quotient = debug_dump::capture_cuda_bytes(
                    &*workspace.fp_inputs,
                    inputs
                        .layout
                        .fp_offset(crate::integration::layout::PolySegment::Quotients)
                        * 8,
                    len,
                    &stream,
                )?;
                Ok(serde_json::to_vec(&quotient)?)
            },
        );
    }

    fn compute_part<F, const D: usize>(
        inputs: &QuotientInputs<'_, F, D>,
        workspace: &mut QuotientWorkspace<'_>,
        part: usize,
        gate_names: &Vec<String>,
    ) where
        F: RichField + Extendable<D>,
    {
        let rate_bits = inputs.circuit.rate_bits;
        let i_r = part.reverse_bits() >> (64 - rate_bits);
        let degree_bits = inputs.circuit.degree_bits as u32;
        let results_offset = inputs.layout.fp_offset_within(
            crate::integration::layout::PolySegment::Quotients,
            i_r << degree_bits,
        );
        let results_ptr = workspace
            .fp_inputs
            .as_mut_ptr()
            .wrapping_add(results_offset);
        for (i, gate_name) in gate_names.iter().enumerate() {
            if let Some(pos) = gate_name.find("Gate") {
                if &gate_name[..(pos + 4)] == "" {
                    continue;
                }
                let selector_index = inputs.circuit.selectors_info.selector_indices[i];
                let start = inputs.circuit.selectors_info.groups[selector_index].start;
                let end = inputs.circuit.selectors_info.groups[selector_index].end;
                QuotientEvaluator::evaluate_gate(
                    inputs,
                    workspace,
                    results_ptr,
                    part,
                    gate_name,
                    i,
                    selector_index,
                    start,
                    end,
                )
                .unwrap();
            }
        }
        QuotientEvaluator::evaluate_z(inputs, workspace, part).unwrap();
    }
}
