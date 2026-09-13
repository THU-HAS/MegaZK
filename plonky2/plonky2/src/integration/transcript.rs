use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::memory::DeviceAllocation;
use cudart::result::CudaResult;

use super::poseidon::ChallengerGpu;

pub(crate) struct TranscriptStep;

impl TranscriptStep {
    #[inline]
    pub(crate) fn cap_len(cap_height: usize) -> usize {
        4 << cap_height
    }

    #[inline]
    pub(crate) fn cap_offset(tree_len: usize, cap_height: usize) -> usize {
        tree_len - (8 << cap_height)
    }

    #[inline]
    pub(crate) fn observe_cap(
        challenger: &mut ChallengerGpu,
        cap: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        cap_height: usize,
    ) -> CudaResult<()> {
        let offset = Self::cap_offset(cap.len(), cap_height);
        challenger.observe_elements_gpu(cap, Self::cap_len(cap_height), offset)
    }

    #[inline]
    pub(crate) fn observe_elements(
        challenger: &mut ChallengerGpu,
        elements: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        count: usize,
    ) -> CudaResult<()> {
        challenger.observe_elements_gpu(elements, count, 0)
    }

    #[inline]
    pub(crate) fn sample_into(
        challenger: &mut ChallengerGpu,
        destination: &mut DeviceAllocation<GoldilocksFieldBoojum>,
        count: usize,
    ) -> CudaResult<()> {
        challenger.get_n_challenges_gpu(destination, count, 0)
    }
}
