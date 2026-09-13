//! Quotient module boundary.
//!
//! This module exposes the minimal borrowed phase inputs and service entry used by the GPU facade.
//! Buffer ownership lives in `state` or sibling components; algorithms live in `service` and
//! `evaluator`, and transcript/commitment sequencing remains outside this module.

mod evaluator;
mod service;
pub(super) mod state;

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::memory::DeviceAllocation;
pub(super) use service::QuotientService;

use self::state::{QuotientState, QuotientStatic};
use super::layout::PolyLayout;
use super::partial_products::PartialProductState;
use super::public_inputs::PublicInputBuffers;
use crate::field::extension::Extendable;
use crate::gates::gate::GateRef;
use crate::gates::selectors::SelectorsInfo;
use crate::hash::hash_types::RichField;
use crate::plonk::circuit_data::CommonCircuitData;

pub(super) struct QuotientCircuitData<'a, F: RichField + Extendable<D>, const D: usize> {
    pub(super) gates: &'a [GateRef<F, D>],
    pub(super) selectors_info: &'a SelectorsInfo,
    pub(super) degree_bits: usize,
    pub(super) rate_bits: usize,
    pub(super) num_wires: usize,
    pub(super) num_constants: usize,
    pub(super) num_challenges: usize,
    pub(super) num_routed_wires: usize,
    pub(super) num_gate_constraints: usize,
    pub(super) num_partial_products: usize,
}

impl<'a, F: RichField + Extendable<D>, const D: usize> QuotientCircuitData<'a, F, D> {
    pub(super) fn from_common(common: &'a CommonCircuitData<F, D>) -> Self {
        Self {
            gates: &common.gates,
            selectors_info: &common.selectors_info,
            degree_bits: common.degree_bits(),
            rate_bits: common.config.fri_config.rate_bits,
            num_wires: common.config.num_wires,
            num_constants: common.config.num_constants,
            num_challenges: common.config.num_challenges,
            num_routed_wires: common.config.num_routed_wires,
            num_gate_constraints: common.num_gate_constraints,
            num_partial_products: common.num_partial_products,
        }
    }
}

pub(super) struct QuotientInputs<'a, F: RichField + Extendable<D>, const D: usize> {
    pub(super) circuit: QuotientCircuitData<'a, F, D>,
    pub(super) layout: &'a PolyLayout,
    pub(super) static_data: &'a QuotientStatic,
    pub(super) state: &'a QuotientState,
    pub(super) public_inputs: &'a PublicInputBuffers,
    pub(super) partial_products: &'a PartialProductState,
}

pub(super) struct QuotientWorkspace<'a> {
    pub(super) fp_inputs: &'a mut DeviceAllocation<GoldilocksFieldBoojum>,
    pub(super) ldes: &'a DeviceAllocation<GoldilocksFieldBoojum>,
}

pub(super) struct QuotientInitInputs<'a, F> {
    pub(super) subgroup: &'a [F],
    pub(super) degree_bits: usize,
    pub(super) num_routed_wires: usize,
}

pub(super) struct QuotientInitWorkspace<'a> {
    pub(super) static_data: &'a mut QuotientStatic,
}
