//! Active quotient constraint evaluation.
//!
//! This module owns the Rust CUDA wrappers and launches for gate and permutation constraints.
//! It does not own buffers, run the quotient phase loop, or interact with the transcript.

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::cuda_kernel;
use cudart::error::get_last_error;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::result::{CudaResult, CudaResultWrap};
use cudart::stream::CudaStream;

use super::{QuotientInputs, QuotientWorkspace};
use crate::integration::layout::PolySegment;

const TILE: u32 = 16;

cuda_kernel!(
    GatePartial,
    gate_partial_kernel,
    ldes_ptr: *const GoldilocksFieldBoojum,
    polys_ptr: *const GoldilocksFieldBoojum,
    public_inputs_hash: *const GoldilocksFieldBoojum,
    output: *mut GoldilocksFieldBoojum,
    alphas: *const GoldilocksFieldBoojum,
    degree_bits: u32,
    rate_bits: u32,
    num_wires: u32,
    num_wires_ldes: u32,
    num_const_sigmas: u32,
    num_const_sigmas_ldes: u32,
    part: u32,
    num_gate_constraints: u32,
    num_selectors: u32,
    num_constants: u32,
    param1: u32,
    param2: u32,
    param3: u32,
    row: u32,
    selector_index: u32,
    start: u32,
    end: u32,
);

gate_partial_kernel!(noop_gate_partial);
gate_partial_kernel!(constant_gate_partial);
gate_partial_kernel!(public_input_gate_partial);
gate_partial_kernel!(arithmetic_gate_partial);
gate_partial_kernel!(poseidon_gate_partial);
gate_partial_kernel!(base_sum_gate_partial);
gate_partial_kernel!(u32_interleave_gate_partial);
gate_partial_kernel!(uninterleave_to_u32_gate_partial);
gate_partial_kernel!(u32_add_many_gate_partial);
gate_partial_kernel!(u32_arithmetic_gate_partial);
gate_partial_kernel!(u32_subtraction_gate_partial);
gate_partial_kernel!(comparison_gate_partial);
gate_partial_kernel!(u32_range_check_gate_partial);
gate_partial_kernel!(random_access_gate_partial);

cuda_kernel!(
    ZpartialPartial,
    z_partial_kernel_partial,
    points: *const GoldilocksFieldBoojum,
    z_h_coset: *const GoldilocksFieldBoojum,
    k_is: *const GoldilocksFieldBoojum,
    alphas: *const GoldilocksFieldBoojum,
    ldes: *const GoldilocksFieldBoojum,
    polys: *const GoldilocksFieldBoojum,
    betas: *const GoldilocksFieldBoojum,
    gammas: *const GoldilocksFieldBoojum,
    output: *mut GoldilocksFieldBoojum,
    degree_bits: u32,
    part: u32,
    num_wires: u32,
    num_wires_ldes: u32,
    num_const_sigmas: u32,
    num_const_sigmas_ldes: u32,
    num_challenges: u32,
    num_routed_wires: u32,
    num_partial_products: u32,
    num_partial_products_ldes: u32,
    num_gate_constraints: u32,
);

z_partial_kernel_partial!(z_partial_partial);

pub(super) struct QuotientEvaluator;

impl QuotientEvaluator {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn evaluate_gate<F, const D: usize>(
        inputs: &QuotientInputs<'_, F, D>,
        workspace: &mut QuotientWorkspace<'_>,
        out_ptr: *mut GoldilocksFieldBoojum,
        part: usize,
        gate_name: &str,
        row: usize,
        selector_index: usize,
        start: usize,
        end: usize,
    ) -> CudaResult<()>
    where
        F: crate::hash::hash_types::RichField + crate::field::extension::Extendable<D>,
    {
        let mut param1: u32 = 0;
        let mut param2: u32 = 0;
        let mut param3: u32 = 0;
        if let Some(pos) = gate_name.find("Gate") {
            let function = match &gate_name[..(pos + 4)] {
                "NoopGate" => noop_gate_partial,
                "ConstantGate" => {
                    if let Some(start) = gate_name.find("num_consts:") {
                        let end = gate_name[start..].find('}').unwrap() + start;
                        param1 = gate_name[start + 11..end].trim().parse().unwrap_or(0);
                    }
                    constant_gate_partial
                }
                "PublicInputGate" => public_input_gate_partial,
                "ArithmeticGate" => {
                    if let Some(start) = gate_name.find("num_ops:") {
                        let end = gate_name[start..].find('}').unwrap() + start;
                        param1 = gate_name[start + 8..end].trim().parse().unwrap_or(0);
                    }
                    arithmetic_gate_partial
                }
                "PoseidonGate" => poseidon_gate_partial,
                "BaseSumGate" => {
                    if let Some(start) = gate_name.find("num_limbs:") {
                        let end = gate_name[start..].find('}').unwrap() + start;
                        param1 = gate_name[start + 10..end].trim().parse().unwrap_or(0);
                    }
                    if let Some(start) = gate_name.find("Base:") {
                        param2 = gate_name[start + 5..].trim().parse().unwrap_or(0);
                    }
                    base_sum_gate_partial
                }
                "U32InterleaveGate" => {
                    if let Some(start) = gate_name.find("num_ops:") {
                        let end = gate_name[start..].find('}').unwrap() + start;
                        param1 = gate_name[start + 8..end].trim().parse().unwrap_or(0);
                    }
                    u32_interleave_gate_partial
                }
                "UninterleaveToU32Gate" => {
                    if let Some(start) = gate_name.find("num_ops:") {
                        let end = gate_name[start..].find('}').unwrap() + start;
                        param1 = gate_name[start + 8..end].trim().parse().unwrap_or(0);
                    }
                    uninterleave_to_u32_gate_partial
                }
                "U32AddManyGate" => {
                    if let Some(start) = gate_name.find("num_addends:") {
                        let end = gate_name[start..].find(',').unwrap() + start;
                        param1 = gate_name[start + 12..end].trim().parse().unwrap_or(0);
                    }
                    if let Some(start) = gate_name.find("num_ops:") {
                        let end = gate_name[start..].find(',').unwrap() + start;
                        param2 = gate_name[start + 8..end].trim().parse().unwrap_or(0);
                    }
                    u32_add_many_gate_partial
                }
                "U32ArithmeticGate" => {
                    if let Some(start) = gate_name.find("num_ops:") {
                        let end = gate_name[start..].find(',').unwrap() + start;
                        param1 = gate_name[start + 8..end].trim().parse().unwrap_or(0);
                    }
                    u32_arithmetic_gate_partial
                }
                "U32SubtractionGate" => {
                    if let Some(start) = gate_name.find("num_ops:") {
                        let end = gate_name[start..].find(',').unwrap() + start;
                        param1 = gate_name[start + 8..end].trim().parse().unwrap_or(0);
                    }
                    u32_subtraction_gate_partial
                }
                "ComparisonGate" => {
                    if let Some(start) = gate_name.find("num_bits:") {
                        let end = gate_name[start..].find(',').unwrap() + start;
                        param1 = gate_name[start + 9..end].trim().parse().unwrap_or(0);
                    }
                    if let Some(start) = gate_name.find("num_chunks:") {
                        let end = gate_name[start..].find(',').unwrap() + start;
                        param2 = gate_name[start + 11..end].trim().parse().unwrap_or(0);
                    }
                    comparison_gate_partial
                }
                "U32RangeCheckGate" => {
                    if let Some(start) = gate_name.find("num_input_limbs:") {
                        let end = gate_name[start..].find(',').unwrap() + start;
                        param1 = gate_name[start + 16..end].trim().parse().unwrap_or(0);
                    }
                    u32_range_check_gate_partial
                }
                "RandomAccessGate" => {
                    if let Some(start) = gate_name.find("bits:") {
                        let end = gate_name[start..].find(',').unwrap() + start;
                        param1 = gate_name[start + 5..end].trim().parse().unwrap_or(0);
                    }
                    if let Some(start) = gate_name.find("num_copies:") {
                        let end = gate_name[start..].find(',').unwrap() + start;
                        param2 = gate_name[start + 11..end].trim().parse().unwrap_or(0);
                    }
                    if let Some(start) = gate_name.find("num_extra_constants:") {
                        let end = gate_name[start..].find(',').unwrap() + start;
                        param3 = gate_name[start + 20..end].trim().parse().unwrap_or(0);
                    }
                    random_access_gate_partial
                }
                _ => {
                    println!("Unknown gate name: {}", gate_name);
                    return Ok(());
                }
            };

            let threads: u32 = 128;
            let point_per_thread = TILE;
            let n = inputs.static_data.points().len() as u32;
            let blocks = (n + threads * point_per_thread - 1) / (threads * point_per_thread);
            let stream = CudaStream::default();
            let config = CudaLaunchConfig::basic(blocks, threads, &stream);
            let args = GatePartialArguments::new(
                workspace.ldes.as_ptr(),
                workspace.fp_inputs.as_ptr(),
                inputs.public_inputs.hash().as_ptr(),
                out_ptr,
                inputs.state.alphas().as_ptr(),
                inputs.circuit.degree_bits as u32,
                inputs.circuit.rate_bits as u32,
                inputs.circuit.num_wires as u32,
                inputs.layout.num_ldes(PolySegment::Wires) as u32,
                inputs.layout.polynomial_count(PolySegment::ConstantsSigmas) as u32,
                inputs.layout.num_ldes(PolySegment::ConstantsSigmas) as u32,
                part as u32,
                inputs.circuit.num_gate_constraints as u32,
                inputs.circuit.selectors_info.num_selectors() as u32,
                inputs.circuit.num_constants as u32,
                param1,
                param2,
                param3,
                row as u32,
                selector_index as u32,
                start as u32,
                end as u32,
            );
            GatePartialFunction(function).launch(&config, &args)?;
            stream.synchronize().unwrap();
            get_last_error().wrap()
        } else {
            Ok(())
        }
    }

    pub(super) fn evaluate_z<F, const D: usize>(
        inputs: &QuotientInputs<'_, F, D>,
        workspace: &mut QuotientWorkspace<'_>,
        part: usize,
    ) -> CudaResult<()>
    where
        F: crate::hash::hash_types::RichField + crate::field::extension::Extendable<D>,
    {
        let rate_bits = inputs.circuit.rate_bits;
        let i_r = part.reverse_bits() >> (64 - rate_bits);
        let degree_bits = inputs.circuit.degree_bits as u32;
        let results_offset = inputs
            .layout
            .fp_offset_within(PolySegment::Quotients, i_r << degree_bits);
        let results_ptr = workspace
            .fp_inputs
            .as_mut_ptr()
            .wrapping_add(results_offset);

        let threads: u32 = 128;
        let point_per_thread = TILE;
        let n = inputs.static_data.points().len() as u32;
        let blocks = (n + threads * point_per_thread - 1) / (threads * point_per_thread);
        let stream = CudaStream::default();
        let config = CudaLaunchConfig::basic(blocks, threads, &stream);
        let args = ZpartialPartialArguments::new(
            inputs.static_data.points().as_ptr(),
            inputs.static_data.z_h_coset().as_ptr(),
            inputs.static_data.k_is().as_ptr(),
            inputs.state.alphas().as_ptr(),
            workspace.ldes.as_ptr(),
            workspace.fp_inputs.as_ptr(),
            inputs.partial_products.betas().as_ptr(),
            inputs.partial_products.gammas().as_ptr(),
            results_ptr,
            inputs.circuit.degree_bits as u32,
            part as u32,
            inputs.circuit.num_wires as u32,
            inputs.layout.num_ldes(PolySegment::Wires) as u32,
            inputs.layout.polynomial_count(PolySegment::ConstantsSigmas) as u32,
            inputs.layout.num_ldes(PolySegment::ConstantsSigmas) as u32,
            inputs.circuit.num_challenges as u32,
            inputs.circuit.num_routed_wires as u32,
            inputs.circuit.num_partial_products as u32,
            inputs.layout.num_ldes(PolySegment::PartialProducts) as u32,
            inputs.circuit.num_gate_constraints as u32,
        );
        ZpartialPartialFunction(z_partial_partial).launch(&config, &args)?;
        stream.synchronize().unwrap();
        get_last_error().wrap()
    }
}
