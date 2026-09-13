use alloc::vec::Vec;
use boojum::field::Field;
use std::time::Instant;
use itertools::concat;

use crate::util::{log2_strict, transpose};

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use cudart::stream::CudaStream;
use boojum_cuda::poseidon::{Poseidon, build_merkle_tree, build_merkle_tree_leaves};


use cudart::cuda_kernel;
use cudart::error::get_last_error;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::result::{CudaResult, CudaResultWrap};

pub fn merkle_tree_gpu(
    leaves_: Vec<Vec<GoldilocksFieldBoojum>>, 
    cap_height: usize
) -> Vec<GoldilocksFieldBoojum> {
    let s = Instant::now();
    let leaves = transpose(&leaves_);
    println!("Time taken to transpose: {:?}", s.elapsed());
    let log_n = log2_strict(leaves[0].len());
    //let values_per_row: usize = leaves[0].len();
    let n: usize = 1 << log_n;
    let layers_count: u32 = (log_n + 1) as u32;
    let values_host = concat(leaves.clone());

    let mut results_host = vec![GoldilocksFieldBoojum::ZERO; n * 4 * 2];
    let stream = CudaStream::default();
    let mut values_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(values_host.len()).unwrap();
    let mut results_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(results_host.len()).unwrap();
    //set_to_zero(&mut results_device, &stream).unwrap();
    let s = Instant::now();
    memory_copy_async(&mut values_device, &values_host, &stream).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken to copy h2d: {:?}", s.elapsed());
    let s = Instant::now();
    build_merkle_tree::<Poseidon>(
        &values_device,
        &mut results_device,
        0,
        &stream,
        layers_count,
    )
    .unwrap();
    stream.synchronize().unwrap();
    println!("Time taken to calculate: {:?}", s.elapsed());
    let s = Instant::now();
    memory_copy_async(&mut results_host, &results_device, &stream).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken to copy d2h: {:?}", s.elapsed());
    //let (nodes, nodes_remaining) = results_host.split_at(results_host.len() >> 1);

    let result = results_host.to_vec();
    result
}

pub fn leaves_gpu(
    leaves: Vec<Vec<GoldilocksFieldBoojum>>, 
) -> Vec<GoldilocksFieldBoojum> {
    let log_n = log2_strict(leaves.len());
    //let values_per_row: usize = leaves[0].len();
    let n: usize = 1 << log_n;
    let values_host = concat(transpose(&leaves));

    let mut results_host = vec![GoldilocksFieldBoojum::ZERO; n * 4];
    let stream = CudaStream::default();
    let mut values_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(values_host.len()).unwrap();
    let mut results_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(results_host.len()).unwrap();
    //set_to_zero(&mut results_device, &stream).unwrap();
    memory_copy_async(&mut values_device, &values_host, &stream).unwrap();
    build_merkle_tree_leaves::<Poseidon>(
        &values_device,
        &mut results_device,
        0,
        false,
        false,
        &stream,
    )
    .unwrap();
    memory_copy_async(&mut results_host, &results_device, &stream).unwrap();
    stream.synchronize().unwrap();
    //let (nodes, nodes_remaining) = results_host.split_at(results_host.len() >> 1);
    results_host
    
}

cuda_kernel!(
    Hashing,
    hashing_kernel,
    elements: *mut GoldilocksFieldBoojum,
    sponge: *mut GoldilocksFieldBoojum,
    input_buffer: *mut GoldilocksFieldBoojum,
    output_buffer: *mut GoldilocksFieldBoojum,
    num_elements: u32,
    in_idx: u32,
    out_idx: u32,
    offset: u32,
);

hashing_kernel!(observe);
hashing_kernel!(get);

pub struct ChallengerGpu {
    pub sponge_state: DeviceAllocation<GoldilocksFieldBoojum>,
    pub input_buffer: DeviceAllocation<GoldilocksFieldBoojum>,
    pub output_buffer: DeviceAllocation<GoldilocksFieldBoojum>,
    pub in_idx: u32,
    pub out_idx: u32,
}

impl ChallengerGpu {
    pub fn new() -> Self {
        let mut sponge_state = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(12).unwrap();
        let mut input_buffer = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(8).unwrap();
        let mut output_buffer = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(8).unwrap();
        let zero8 = vec![GoldilocksFieldBoojum::from_nonreduced_u64(0); 8];
        let zero12 = vec![GoldilocksFieldBoojum::from_nonreduced_u64(0); 12];
        let stream = CudaStream::default();
        memory_copy_async(&mut sponge_state, &zero12, &stream).unwrap();
        memory_copy_async(&mut input_buffer, &zero8, &stream).unwrap();
        memory_copy_async(&mut output_buffer, &zero8, &stream).unwrap();
        stream.synchronize().unwrap();
        Self {
            sponge_state,
            input_buffer,
            output_buffer,
            in_idx: 0,
            out_idx: 0,
        }
    }

    pub fn show_sponge(&self) {
        let mut sponge_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(12, CudaHostAllocFlags::DEFAULT).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut sponge_host, &self.sponge_state, &stream).unwrap();
        stream.synchronize().unwrap();
        println!("{:?}" , sponge_host.to_vec());
    }

    pub fn show_ibuf(&self) {
        let mut sponge_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(8, CudaHostAllocFlags::DEFAULT).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut sponge_host, &self.input_buffer, &stream).unwrap();
        stream.synchronize().unwrap();
        println!("{:?}" , sponge_host.to_vec());
        println!("in_idx: {}", self.in_idx);
    }

    pub fn observe_elements_gpu(&mut self, elts: &mut DeviceAllocation<GoldilocksFieldBoojum>, num_elts: usize, offset: usize) -> CudaResult<()> {
        let stream = CudaStream::default();
        let config = CudaLaunchConfig::basic(1, 1, &stream);
        let args = HashingArguments::new(
            elts.as_mut_ptr(),
            self.sponge_state.as_mut_ptr(),
            self.input_buffer.as_mut_ptr(),
            self.output_buffer.as_mut_ptr(),
            num_elts as u32,
            self.in_idx,
            self.out_idx,
            offset as u32,
        );
        HashingFunction(observe).launch(&config, &args)?;
        self.in_idx = (self.in_idx + elts.len() as u32) % 8;
        self.out_idx = if self.in_idx + elts.len() as u32 >= 8 {8} else {self.out_idx};
        stream.synchronize().unwrap();
        // self.show_sponge();
        get_last_error().wrap()
    }

    pub fn observe_elements(&mut self, elts: &Vec<GoldilocksFieldBoojum>) -> CudaResult<()> {
        let mut elements_device = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(elts.len()).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut elements_device, elts, &stream).unwrap();
        stream.synchronize().unwrap();
        self.observe_elements_gpu(&mut elements_device, elts.len(), 0)
    }

    pub fn get_n_challenges_gpu(&mut self, clgs: &mut DeviceAllocation<GoldilocksFieldBoojum>, n: usize, offset: usize) -> CudaResult<()> {
        let stream = CudaStream::default();
        let config = CudaLaunchConfig::basic(1, 1, &stream);
        let args = HashingArguments::new(
            clgs.as_mut_ptr(),
            self.sponge_state.as_mut_ptr(),
            self.input_buffer.as_mut_ptr(),
            self.output_buffer.as_mut_ptr(),
            n as u32,
            self.in_idx,
            self.out_idx,
            offset as u32
        );
        HashingFunction(get).launch(&config, &args).unwrap();
        self.out_idx = (if self.in_idx != 0 {(8 - (n as u32) % 8) % 8} else {(self.out_idx + 8 - (n as u32) % 8) % 8}).try_into().unwrap();
        self.in_idx = 0;
        get_last_error().wrap()
    }

    pub fn get_n_challenges(&mut self, n: usize)  -> Vec<GoldilocksFieldBoojum> {
        let mut challenges_device = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(n).unwrap();
        let mut challenges_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(n, CudaHostAllocFlags::DEFAULT).unwrap();
        let stream = CudaStream::default();
        self.get_n_challenges_gpu(&mut challenges_device, n, 0).unwrap();
        // println!("{}", self.out_idx);
        memory_copy_async(&mut challenges_host, &challenges_device, &stream).unwrap();
        stream.synchronize().unwrap();
        challenges_host.to_vec()
    }
}