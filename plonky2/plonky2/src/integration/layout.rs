use core::ops::Range;

/// LDE and polynomial layout segments, in `build_gpu`'s `num_ldes` order.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum GpuPolynomialSegment {
    ConstantsSigmas = 0,
    Wires = 1,
    PartialProducts = 2,
    Quotients = 3,
}

impl GpuPolynomialSegment {
    pub(crate) const ALL: [Self; 4] = [
        Self::ConstantsSigmas,
        Self::Wires,
        Self::PartialProducts,
        Self::Quotients,
    ];

    #[inline]
    const fn index(self) -> usize {
        self as usize
    }
}

pub(crate) type PolySegment = GpuPolynomialSegment;

#[derive(Clone, Debug)]
pub struct PolyLayout {
    degree: usize,
    lde_length: usize,
    num_const_sigmas: usize,
    num_partial_products: usize,
    num_quotients: usize,
    num_polys: usize,
    fp_offsets: [usize; 4],
    num_ldes: [usize; 4],
    num_total_ldes: usize,
    num_polynomials: [usize; 4],
    lde_offsets: [usize; 4],
}

impl PolyLayout {
    pub(crate) fn new(
        degree: usize,
        rate_bits: usize,
        num_const_sigmas: usize,
        num_wires: usize,
        num_partial_products: usize,
        num_quotients: usize,
        num_ldes: [usize; 4],
    ) -> Self {
        assert!(num_ldes[0] <= num_const_sigmas);
        assert!(num_ldes[1] <= num_wires);
        assert!(num_ldes[2] <= num_partial_products);
        assert!(num_ldes[3] <= num_quotients);

        let num_polys = num_const_sigmas + num_wires + num_partial_products + num_quotients;
        let fp_offsets = [
            0,
            degree * num_const_sigmas,
            degree * (num_const_sigmas + num_wires),
            degree * (num_const_sigmas + num_wires + num_partial_products),
        ];
        let lde_length = degree << rate_bits;
        let lde_offsets = [
            0,
            lde_length * num_ldes[0],
            lde_length * (num_ldes[0] + num_ldes[1]),
            lde_length * (num_ldes[0] + num_ldes[1] + num_ldes[2]),
        ];
        let num_total_ldes = num_ldes[0] + num_ldes[1] + num_ldes[2] + num_ldes[3];

        Self {
            degree,
            lde_length,
            num_const_sigmas,
            num_partial_products,
            num_quotients,
            num_polys,
            fp_offsets,
            num_ldes,
            lde_offsets,
            num_polynomials: [
                num_const_sigmas,
                num_wires,
                num_partial_products,
                num_quotients,
            ],
            num_total_ldes,
        }
    }

    #[inline]
    pub fn degree(&self) -> usize {
        self.degree
    }

    #[inline]
    pub fn lde_length(&self) -> usize {
        self.lde_length
    }

    #[inline]
    pub fn num_const_sigmas(&self) -> usize {
        self.num_const_sigmas
    }

    #[inline]
    pub fn num_partial_products(&self) -> usize {
        self.num_partial_products
    }

    #[inline]
    pub fn num_quotients(&self) -> usize {
        self.num_quotients
    }

    #[inline]
    pub fn num_total_ldes(&self) -> usize {
        self.num_total_ldes
    }

    #[inline]
    pub fn fp_offset(&self, segment: GpuPolynomialSegment) -> usize {
        self.fp_offsets[segment.index()]
    }

    #[inline]
    pub fn fp_offset_within(&self, segment: GpuPolynomialSegment, offset: usize) -> usize {
        self.fp_offset(segment) + offset
    }

    #[inline]
    pub fn fp_range(&self, segment: GpuPolynomialSegment) -> Range<usize> {
        let start = self.fp_offset(segment);
        start..start + self.degree * self.polynomial_count(segment)
    }

    #[inline]
    pub fn fp_offset_after_ldes(&self, segment: GpuPolynomialSegment, degree_bits: usize) -> usize {
        debug_assert_eq!(self.degree, 1 << degree_bits);
        self.fp_offset(segment) + (self.num_ldes(segment) << degree_bits)
    }

    #[inline]
    pub fn polynomial_offset(&self, segment: GpuPolynomialSegment) -> usize {
        match segment {
            GpuPolynomialSegment::ConstantsSigmas => 0,
            GpuPolynomialSegment::Wires => self.num_polynomials[0],
            GpuPolynomialSegment::PartialProducts => {
                self.num_polynomials[0] + self.num_polynomials[1]
            }
            GpuPolynomialSegment::Quotients => {
                self.num_polynomials[0] + self.num_polynomials[1] + self.num_polynomials[2]
            }
        }
    }

    #[inline]
    pub fn polynomial_count(&self, segment: GpuPolynomialSegment) -> usize {
        self.num_polynomials[segment.index()]
    }

    #[inline]
    pub fn non_lde_polynomial_count(&self, segment: GpuPolynomialSegment) -> usize {
        self.polynomial_count(segment) - self.num_ldes(segment)
    }

    #[inline]
    pub fn num_ldes(&self, segment: GpuPolynomialSegment) -> usize {
        self.num_ldes[segment.index()]
    }

    #[inline]
    pub fn lde_offset(&self, segment: GpuPolynomialSegment) -> usize {
        self.lde_offsets[segment.index()]
    }

    #[inline]
    pub fn num_polys(&self) -> usize {
        self.num_polys
    }

    #[inline]
    pub fn fp_len(&self) -> usize {
        self.degree * self.num_polys()
    }

    #[inline]
    pub fn lde_len(&self) -> usize {
        self.num_total_ldes * self.lde_length
    }
}
