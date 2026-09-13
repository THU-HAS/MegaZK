use std::time::Instant;

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::cuda_kernel;
use cudart::error::get_last_error;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::memory::DeviceAllocation;
use cudart::result::{CudaResult, CudaResultWrap};
use cudart::stream::CudaStream;

cuda_kernel!(
    HashPI,
    hash_pi_kernel,
    witness: *const GoldilocksFieldBoojum,
    pi_index: *const usize,
    pi: *mut GoldilocksFieldBoojum,
    pi_hash: *mut GoldilocksFieldBoojum,
    num_public_inputs: usize,
);

hash_pi_kernel!(hash_pi);

pub(crate) struct PublicInputBuffers {
    values: DeviceAllocation<GoldilocksFieldBoojum>,
    hash: DeviceAllocation<GoldilocksFieldBoojum>,
}

impl PublicInputBuffers {
    pub(crate) fn new(num_public_inputs: usize) -> Self {
        let hash = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(4).unwrap();
        let values = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(num_public_inputs).unwrap();
        Self { values, hash }
    }

    #[inline]
    pub(crate) fn values(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.values
    }

    #[inline]
    pub(crate) fn hash(&self) -> &DeviceAllocation<GoldilocksFieldBoojum> {
        &self.hash
    }

    #[inline]
    pub(crate) fn hash_mut(&mut self) -> &mut DeviceAllocation<GoldilocksFieldBoojum> {
        &mut self.hash
    }
}

pub(crate) struct PublicInputsService;

impl PublicInputsService {
    pub(crate) fn hash(
        buffers: &mut PublicInputBuffers,
        witness: &DeviceAllocation<GoldilocksFieldBoojum>,
        public_input_indices: &DeviceAllocation<usize>,
        num_public_inputs: usize,
    ) -> CudaResult<()> {
        let t = Instant::now();
        let stream = CudaStream::default();
        let config = CudaLaunchConfig::basic(1, 1, &stream);
        let args = HashPIArguments::new(
            witness.as_ptr(),
            public_input_indices.as_ptr(),
            buffers.values.as_mut_ptr(),
            buffers.hash.as_mut_ptr(),
            num_public_inputs,
        );
        HashPIFunction(hash_pi).launch(&config, &args)?;
        let launch_elapsed = t.elapsed();
        stream.synchronize().unwrap();
        println!(
            "  hash_pi kernel: n_pi={} launch {:?} sync {:?}",
            num_public_inputs,
            launch_elapsed,
            t.elapsed().saturating_sub(launch_elapsed)
        );
        get_last_error().wrap()
    }
}
