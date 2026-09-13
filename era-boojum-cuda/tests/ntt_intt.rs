use boojum::field::goldilocks::GoldilocksField;
use boojum::field::{Field, PrimeField};
use boojum::gadgets::num;
use boojum_cuda::integration::{batch_mul_exp, batch_pad_coset};
use boojum_cuda::ntt::*;
use boojum_cuda::context::Context;
use boojum_cuda::poseidon::Poseidon;
use boojum_cuda::poseidon::build_merkle_tree;
use cudart::memory::{DeviceAllocation, HostAllocation};
use cudart::memory::CudaHostAllocFlags;
use cudart::memory::memory_copy_async;
use cudart::stream::CudaStream;
use std::mem;
use std::time::Instant;
use boojum_cuda::merkle_tree_segmented::*;

#[test]
fn reverse() {
    let num_ntts = 1;
    let log_n = 5;
    let length: usize = num_ntts << log_n;
    let degree: usize = 1 << log_n;
    let rate_bits: u32 = 3;
    let mut ctx = Context::create(12, 12).unwrap();
    let mut inputs_matrix_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
    let mut outputs_matrix_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
    let mut outputs_matrix_host = HostAllocation::<GoldilocksField>::alloc(length, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut lde_matrix_device = DeviceAllocation::<GoldilocksField>::alloc(length << rate_bits).unwrap();
    let mut lde_matrix_host = HostAllocation::<GoldilocksField>::alloc(length << rate_bits, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut poseidon_device = DeviceAllocation::<GoldilocksField>::alloc(8 << (log_n + rate_bits)).unwrap();
    let mut poseidon_host = HostAllocation::<GoldilocksField>::alloc(8 << (log_n + rate_bits), CudaHostAllocFlags::DEFAULT).unwrap();
    let mut poseidon_ref_device = DeviceAllocation::<GoldilocksField>::alloc(8 << (log_n + rate_bits)).unwrap();
    let mut poseidon_ref_host = HostAllocation::<GoldilocksField>::alloc(8 << (log_n + rate_bits), CudaHostAllocFlags::DEFAULT).unwrap();

    let stream = CudaStream::default();

    let matrix = (0..length)
        .map(|i| GoldilocksField::from_nonreduced_u64(i as u64))
        .collect::<Vec<_>>();
    println!("matrix: {:?}", matrix);
    memory_copy_async(&mut inputs_matrix_device, &matrix, &stream).unwrap();
    stream.synchronize().unwrap();
    let mul_val = GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR.inverse().expect("Failed to calculate inverse");
    // let mul_val = GoldilocksField::RADIX_2_SUBGROUP_GENERATOR.pow_u64(1);
    let mul_val = GoldilocksField::from_nonreduced_u64(1753635133440165772).pow_u64(2 << (32 - log_n - rate_bits));
    let s = Instant::now();
    batch_lde_transpose_out_of_place(
        &mut inputs_matrix_device,
        &mut lde_matrix_device,
        log_n as u32,
        rate_bits,
        num_ntts as u32,
        0,
        degree as u32,
        (degree << rate_bits) as u32,
        false,
        &stream,
    ).unwrap();
    batch_pad_coset(
        inputs_matrix_device.as_ptr(), 
        outputs_matrix_device.as_mut_ptr(), 
        log_n as u32, 
        num_ntts as u32, 
        degree as u32, 
        degree as u32, 
        0, 
        false, 
        &stream
    ).unwrap();
    batch_mul_exp(
        outputs_matrix_device.as_mut_ptr(), 
        log_n as u32, 
        num_ntts as u32, 
        degree as u32, 
        mul_val, 
        &stream,
    ).unwrap();
    batch_ntt_internal(
        outputs_matrix_device.as_ptr(), 
        outputs_matrix_device.as_mut_ptr(), 
        log_n as u32, 
        num_ntts as u32, 
        degree as u32, 
        degree as u32, 
        false, 
        false, 
        0, 
        0,
        false,
        &stream
    ).unwrap();
    batch_ntt_internal(
        outputs_matrix_device.as_ptr(), 
        outputs_matrix_device.as_mut_ptr(), 
        log_n as u32, 
        num_ntts as u32, 
        degree as u32, 
        degree as u32, 
        true, 
        true, 
        0, 
        0,
        false,
        &stream
    ).unwrap();
    batch_mul_exp(
        outputs_matrix_device.as_mut_ptr(), 
        log_n as u32, 
        num_ntts as u32, 
        degree as u32, 
        (GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR * mul_val).inverse().expect("Failed to calculate inverse"), 
        &stream,
    ).unwrap();
    build_merkle_tree_in_place::<Poseidon>(
        &mut outputs_matrix_device, 
        &mut poseidon_device, 
        log_n as u32, 
        rate_bits as u32, 
        num_ntts as u32, 
        0,
        &stream,
        (log_n + rate_bits + 1) as u32,
    ).unwrap();
    build_merkle_tree::<Poseidon>(
        &lde_matrix_device,
        &mut poseidon_ref_device,
        0,
        &stream,
        (log_n + rate_bits + 1) as u32,
    ).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken in gpu: {:?}", s.elapsed());
    memory_copy_async(&mut outputs_matrix_host, &outputs_matrix_device, &stream).unwrap();
    memory_copy_async(&mut lde_matrix_host, &lde_matrix_device, &stream).unwrap();
    memory_copy_async(&mut poseidon_host, &poseidon_device, &stream).unwrap();
    memory_copy_async(&mut poseidon_ref_host, &poseidon_ref_device, &stream).unwrap();
    stream.synchronize().unwrap();
    let res_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(outputs_matrix_host.to_vec())};
    let res_lde_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(lde_matrix_host.to_vec())};
    let res_poseidon_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(poseidon_host.to_vec())};
    let res_poseidon_ref_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(poseidon_ref_host.to_vec())};
    println!("res_gpu: {:?}", res_gpu);
    println!("res_lde_gpu: {:?}", res_lde_gpu);    
    println!("res_poseidon_gpu: {:?}", res_poseidon_gpu);
    println!("res_poseidon_ref_gpu: {:?}", res_poseidon_ref_gpu);
    assert_eq!(res_poseidon_ref_gpu, res_poseidon_gpu);
    ctx.destroy().unwrap();

}

#[test]
fn partial() {
    let num_ntts = 50;
    let log_n = 5;
    let length: usize = num_ntts << log_n;
    let degree: usize = 1 << log_n;
    let rate_bits: u32 = 3;
    let mut ctx = Context::create(12, 12).unwrap();
    let mut inputs_matrix_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
    let mut outputs_matrix_device = DeviceAllocation::<GoldilocksField>::alloc(length).unwrap();
    let mut outputs_matrix_host = HostAllocation::<GoldilocksField>::alloc(length, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut lde_matrix_device = DeviceAllocation::<GoldilocksField>::alloc(length << rate_bits).unwrap();
    let mut lde_matrix_host = HostAllocation::<GoldilocksField>::alloc(length << rate_bits, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut poseidon_device = DeviceAllocation::<GoldilocksField>::alloc(8 << (log_n + rate_bits)).unwrap();
    let mut poseidon_host = HostAllocation::<GoldilocksField>::alloc(8 << (log_n + rate_bits), CudaHostAllocFlags::DEFAULT).unwrap();
    let mut poseidon_part_device = DeviceAllocation::<GoldilocksField>::alloc(8 << (log_n + rate_bits)).unwrap();
    let mut poseidon_part_host = HostAllocation::<GoldilocksField>::alloc(8 << (log_n + rate_bits), CudaHostAllocFlags::DEFAULT).unwrap();
    let mut poseidon_ref_device = DeviceAllocation::<GoldilocksField>::alloc(8 << (log_n + rate_bits)).unwrap();
    let mut poseidon_ref_host = HostAllocation::<GoldilocksField>::alloc(8 << (log_n + rate_bits), CudaHostAllocFlags::DEFAULT).unwrap();

    let stream = CudaStream::default();

    let matrix = (0..length)
        .map(|i| GoldilocksField::from_nonreduced_u64(i as u64))
        .collect::<Vec<_>>();
    // println!("matrix: {:?}", matrix);
    memory_copy_async(&mut inputs_matrix_device, &matrix, &stream).unwrap();
    stream.synchronize().unwrap();
    let mul_val = GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR.inverse().expect("Failed to calculate inverse");
    // let mul_val = GoldilocksField::RADIX_2_SUBGROUP_GENERATOR.pow_u64(1);
    let mul_val = GoldilocksField::from_nonreduced_u64(1753635133440165772).pow_u64(2 << (32 - log_n - rate_bits));
    let s = Instant::now();
    batch_lde_transpose_out_of_place(
        &mut inputs_matrix_device,
        &mut lde_matrix_device,
        log_n as u32,
        rate_bits,
        num_ntts as u32,
        0,
        degree as u32,
        (degree << rate_bits) as u32,
        false,
        &stream,
    ).unwrap();
    build_merkle_tree_in_place::<Poseidon>(
        &mut inputs_matrix_device, 
        &mut poseidon_device, 
        log_n as u32, 
        rate_bits as u32, 
        num_ntts as u32, 
        0,
        &stream,
        (log_n + rate_bits + 1) as u32,
    ).unwrap();
    stream.synchronize().unwrap();
    let num_ldes = 2;
    build_merkle_tree_w_partial_ldes::<Poseidon>(
        &lde_matrix_device, 
        &mut inputs_matrix_device, 
        &mut poseidon_part_device, 
        log_n as u32, 
        rate_bits as u32, 
        num_ntts as u32, 
        num_ldes as u32,
        0,
        0, 
        &stream,
        (log_n + rate_bits + 1) as u32,
    ).unwrap();
    build_merkle_tree::<Poseidon>(
        &lde_matrix_device,
        &mut poseidon_ref_device,
        0,
        &stream,
        (log_n + rate_bits + 1) as u32,
    ).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken in gpu: {:?}", s.elapsed());
    memory_copy_async(&mut outputs_matrix_host, &outputs_matrix_device, &stream).unwrap();
    memory_copy_async(&mut lde_matrix_host, &lde_matrix_device, &stream).unwrap();
    memory_copy_async(&mut poseidon_host, &poseidon_device, &stream).unwrap();
    memory_copy_async(&mut poseidon_part_host, &poseidon_part_device, &stream).unwrap();
    memory_copy_async(&mut poseidon_ref_host, &poseidon_ref_device, &stream).unwrap();
    stream.synchronize().unwrap();
    let res_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(outputs_matrix_host.to_vec())};
    let res_lde_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(lde_matrix_host.to_vec())};
    let res_poseidon_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(poseidon_host.to_vec())};
    let res_poseidon_part_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(poseidon_part_host.to_vec())};
    let res_poseidon_ref_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(poseidon_ref_host.to_vec())};
    // println!("res_gpu: {:?}", res_gpu);
    // println!("res_lde_gpu: {:?}", res_lde_gpu);    
    // println!("res_poseidon_gpu: {:?}", res_poseidon_gpu);
    assert_eq!(res_poseidon_ref_gpu, res_poseidon_gpu);
    assert_eq!(res_poseidon_ref_gpu, res_poseidon_part_gpu);
    ctx.destroy().unwrap();

}