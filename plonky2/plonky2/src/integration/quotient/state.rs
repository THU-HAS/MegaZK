//! Quotient-owned GPU state and construction configuration.
//!
//! This module owns static domain buffers and per-proof alpha storage. It does not launch kernels,
//! orchestrate phases, or interact with transcript and commitment state.

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::memory::DeviceAllocation;

#[derive(Copy, Clone)]
pub(crate) struct QuotientStaticConfig {
    degree: usize,
    quotient_degree_factor: usize,
    max_quotient_degree_factor: usize,
    num_routed_wires: usize,
    quotient_degree_bits: usize,
}

impl QuotientStaticConfig {
    pub(crate) fn new(
        degree: usize,
        quotient_degree_factor: usize,
        max_quotient_degree_factor: usize,
        num_routed_wires: usize,
        quotient_degree_bits: usize,
    ) -> Self {
        Self {
            degree,
            quotient_degree_factor,
            max_quotient_degree_factor,
            num_routed_wires,
            quotient_degree_bits,
        }
    }
}

pub(crate) struct QuotientStatic {
    quotient_degree_bits: usize,
    k_is: DeviceAllocation<GoldilocksFieldBoojum>,
    subgroup: DeviceAllocation<GoldilocksFieldBoojum>,
    points: DeviceAllocation<GoldilocksFieldBoojum>,
    z_h_coset: DeviceAllocation<GoldilocksFieldBoojum>,
}

impl QuotientStatic {
    pub(crate) fn new(config: QuotientStaticConfig) -> Self {
        Self {
            quotient_degree_bits: config.quotient_degree_bits,
            k_is: DeviceAllocation::<GoldilocksFieldBoojum>::alloc(config.num_routed_wires)
                .unwrap(),
            subgroup: DeviceAllocation::<GoldilocksFieldBoojum>::alloc(config.degree).unwrap(),
            points: DeviceAllocation::<GoldilocksFieldBoojum>::alloc(
                config.degree * config.max_quotient_degree_factor,
            )
            .unwrap(),
            z_h_coset: DeviceAllocation::<GoldilocksFieldBoojum>::alloc(
                2 * config.quotient_degree_factor,
            )
            .unwrap(),
        }
    }

    #[inline]
    pub(crate) fn quotient_degree_bits(&self) -> usize {
        self.quotient_degree_bits
    }

    #[inline]
    pub(crate) fn points(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.points
    }

    #[inline]
    pub(crate) fn points_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.points
    }

    #[inline]
    pub(crate) fn z_h_coset(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.z_h_coset
    }

    #[inline]
    pub(crate) fn z_h_coset_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.z_h_coset
    }

    #[inline]
    pub(crate) fn k_is(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.k_is
    }

    #[inline]
    pub(crate) fn k_is_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.k_is
    }

    #[inline]
    pub(crate) fn subgroup(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.subgroup
    }

    #[inline]
    pub(crate) fn subgroup_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.subgroup
    }
}

pub(crate) struct QuotientState {
    alphas: DeviceAllocation<GoldilocksFieldBoojum>,
}

impl QuotientState {
    pub(crate) fn new(num_challenges: usize) -> Self {
        Self {
            alphas: DeviceAllocation::<GoldilocksFieldBoojum>::alloc(num_challenges).unwrap(),
        }
    }

    #[inline]
    pub(crate) fn alphas(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.alphas
    }

    #[inline]
    pub(crate) fn alphas_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.alphas
    }
}
