//! Active GPU FRI proof-of-work computation.
//!
//! This service receives only the transcript device buffers and index needed by the kernel. It
//! does not own or operate the challenger; observation and response sampling live in
//! `orchestrator.rs`.

use std::mem;
use std::time::Instant;

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::cuda_kernel;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use cudart::stream::CudaStream;

use super::state::FriPowBuffers;

cuda_kernel!(FRIPOWNew, fri_pow_new(
    input_buffer: *const GoldilocksFieldBoojum,
    sponge: *const GoldilocksFieldBoojum,
    witness: *mut GoldilocksFieldBoojum,
    found: *mut u32,
    found_buf: *mut u32,
    buf: *mut GoldilocksFieldBoojum,
    stride: GoldilocksFieldBoojum,
    in_idx: u32,
    min_leading_zeros: u32,
    offset: u32,
));

pub(crate) struct FriPowInputs<'a> {
    input_buffer: &'a DeviceAllocation<GoldilocksFieldBoojum>,
    sponge_state: &'a DeviceAllocation<GoldilocksFieldBoojum>,
    in_idx: u32,
    min_leading_zeros: u32,
}

impl<'a> FriPowInputs<'a> {
    pub(crate) fn new(
        input_buffer: &'a DeviceAllocation<GoldilocksFieldBoojum>,
        sponge_state: &'a DeviceAllocation<GoldilocksFieldBoojum>,
        in_idx: u32,
        min_leading_zeros: u32,
    ) -> Self {
        Self {
            input_buffer,
            sponge_state,
            in_idx,
            min_leading_zeros,
        }
    }
}

pub(crate) struct FriPowService;

impl FriPowService {
    pub(crate) fn compute<F: Copy>(
        buffers: &mut FriPowBuffers,
        pow_witness_output: &mut F,
        inputs: FriPowInputs<'_>,
    ) {
        let stream = CudaStream::default();
        let mut found = HostAllocation::<u32>::alloc(1, CudaHostAllocFlags::DEFAULT).unwrap();
        let threads: u32 = 128;
        let blocks: u32 = 64;
        let config = CudaLaunchConfig::basic(blocks, threads, &stream);
        let mut offset: u32 = 0;
        loop {
            let s = Instant::now();
            let pow = buffers.kernel_buffers_mut();
            let args = FRIPOWNewArguments::new(
                inputs.input_buffer.as_ptr(),
                inputs.sponge_state.as_ptr(),
                pow.witness.as_mut_ptr(),
                pow.witness_found.as_mut_ptr(),
                pow.witness_found_buffer.as_mut_ptr(),
                pow.witness_buffer.as_mut_ptr(),
                GoldilocksFieldBoojum::from_nonreduced_u64((threads * blocks) as u64),
                inputs.in_idx,
                inputs.min_leading_zeros,
                offset,
            );
            FRIPOWNewFunction(fri_pow_new)
                .launch(&config, &args)
                .unwrap();
            memory_copy_async(&mut found, pow.witness_found, &stream).unwrap();
            stream.synchronize().unwrap();
            if found.to_vec()[0] == 1 {
                break;
            }
            println!("  pow round: {:?}", s.elapsed());
            offset += threads * blocks * 8;
        }

        let mut pow_witness =
            HostAllocation::<GoldilocksFieldBoojum>::alloc(1, CudaHostAllocFlags::DEFAULT).unwrap();
        memory_copy_async(&mut pow_witness, buffers.witness(), &stream).unwrap();
        stream.synchronize().unwrap();
        let pow_witness_host: Vec<F> = unsafe { mem::transmute(pow_witness.to_vec()) };
        *pow_witness_output = pow_witness_host[0];
    }
}
