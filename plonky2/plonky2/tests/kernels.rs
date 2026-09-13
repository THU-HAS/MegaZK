use std::mem;
use std::ops::Mul;
//use std::fmt::format;
use std::time::Instant;

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use boojum_cuda::context::Context;
use boojum_cuda::integration::{batch_pad_coset, batch_transpose};
use boojum_cuda::ntt::batch_ntt_out_of_place;
use cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use cudart::stream::CudaStream;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::util::transpose;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

pub const GODILOCKS_PRIME: u64 = (1 << 32 - 1) * (1 << 32) + 1;

fn ntt_matrix_generate(log_count: usize, num_ntt: usize) -> Vec<GoldilocksField> {
    let mut matrix = Vec::<u64>::new();
    let count: u32 = 1 << log_count;
    let mut rand = ChaChaRng::from_entropy();
    for _i in 0..num_ntt {
        // let mut col = Vec::<GoldilocksField>::new();
        for _j in 0..count {
            matrix.push(rand.gen_range(0..GODILOCKS_PRIME));
        }
        // let col = PolynomialValues::<GoldilocksField>::new(col);
        // matrix.push(col);
    }

    unsafe { mem::transmute(matrix) }
}

fn ntt_check(res_cpu: Vec<GoldilocksField>, res_gpu: Vec<GoldilocksField>, log_count: usize) {
    let length = res_cpu.len();
    assert_eq!(length, res_gpu.len());
    let mut count: usize = 0;
    for i in 0..length {
        // assert_eq!(res_cpu[i], res_gpu[i]);
        if res_cpu[i] != res_gpu[i] {
            println!(
                "{:?} != {:?} at ntt {:?}, {:?}",
                res_cpu[i],
                res_gpu[i],
                i >> log_count,
                i - (i >> log_count << log_count)
            );
            count = count + 1;
            // break;
        }
    }
    println!("error count: {:?}", count);
}

fn ntt_gpu_packed(
    matrix: Vec<GoldilocksField>,
    log_count: usize,
    num_ntt: usize,
    inverse: bool,
) -> Vec<GoldilocksField> {
    let matrix: Vec<GoldilocksFieldBoojum> = unsafe { mem::transmute(matrix) };
    let length: usize = num_ntt << log_count;
    let degree: usize = 1 << log_count;
    let ctx = Context::create(12, 12).unwrap();
    let mut inputs_matrix_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut intt_matrix_device = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut intt_matrix_host =
        HostAllocation::<GoldilocksFieldBoojum>::alloc(length, CudaHostAllocFlags::DEFAULT)
            .unwrap();
    let stream = CudaStream::default();
    memory_copy_async(&mut inputs_matrix_device, &matrix, &stream).unwrap();
    stream.synchronize().unwrap();
    let s = Instant::now();
    batch_ntt_out_of_place(
        &mut inputs_matrix_device,
        &mut intt_matrix_device,
        log_count as u32,
        num_ntt as u32,
        0,
        0,
        degree as u32,
        degree as u32,
        false,
        inverse,
        0,
        0,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();
    println!("Time taken in gpu: {:?}", s.elapsed());
    memory_copy_async(&mut intt_matrix_host, &intt_matrix_device, &stream).unwrap();
    stream.synchronize().unwrap();
    let res_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(intt_matrix_host.to_vec()) };

    ctx.destroy().unwrap();
    res_gpu
}

#[test]
fn ntt_test() {
    let log_count = 21;
    let num_ntt = 20;
    let matrix = ntt_matrix_generate(log_count, num_ntt);
    let s = Instant::now();
    let res_cpu: Vec<GoldilocksField> = matrix
        .clone()
        .par_chunks(1 << log_count)
        .map(|v| {
            PolynomialCoeffs::<GoldilocksField>::new(v.to_vec())
                .fft()
                .values
        })
        .collect::<Vec<_>>()
        .concat();
    println!("Time taken in cpu ntt: {:?}", s.elapsed());
    let s = Instant::now();
    let res_gpu = ntt_gpu_packed(matrix, log_count, num_ntt, false);
    println!("Time taken in gpu ntt: {:?}", s.elapsed());
    ntt_check(res_cpu, res_gpu, log_count);
}

#[test]
fn intt_test() {
    let log_count = 10;
    let num_ntt = 1;
    let matrix = ntt_matrix_generate(log_count, num_ntt);
    let s = Instant::now();
    let res_cpu: Vec<GoldilocksField> = matrix
        .clone()
        .par_chunks(1 << log_count)
        .map(|v| {
            PolynomialValues::<GoldilocksField>::new(v.to_vec())
                .ifft()
                .coeffs
        })
        .collect::<Vec<_>>()
        .concat();
    println!("Time taken in cpu ntt: {:?}", s.elapsed());
    let s = Instant::now();
    let res_gpu = ntt_gpu_packed(matrix, log_count, num_ntt, true);
    println!("Time taken in gpu ntt: {:?}", s.elapsed());
    ntt_check(res_cpu, res_gpu, log_count);
}

fn pad_coset_check(
    matrix: Vec<GoldilocksField>,
    res_gpu: Vec<GoldilocksField>,
    log_count: usize,
    num_ntt: usize,
    rate_bits: usize,
    inverse: bool,
) {
    let length = res_gpu.len();
    assert_eq!(num_ntt << log_count << rate_bits, res_gpu.len());
    for i in 0..num_ntt {
        // assert_eq!(res_cpu[i], res_gpu[i]);
        let mut a = GoldilocksField::from_noncanonical_u64(1);
        for j in 0..(1 << log_count) {
            let res_cpu = matrix[(i << log_count) + j].mul(a);
            if res_cpu != res_gpu[(i << log_count << rate_bits) + j] {
                println!("{:?} != {:?} at ntt {:?}, {:?}", res_cpu, res_gpu[i], i, j);
                break;
            }
            if !inverse {
                a = a.mul(GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR);
            } else {
                a = a.mul(GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR.inverse());
            }
        }
        for j in (1 << log_count)..(1 << log_count << rate_bits) {
            if res_gpu[(i << log_count << rate_bits) + j] != GoldilocksField::ZERO {
                println!("{:?} != 0 at ntt {:?}, {:?}", res_gpu[i], i, j);
                break;
            }
        }
    }
}

fn pad_coset_packed(
    matrix: Vec<GoldilocksField>,
    log_count: usize,
    num_ntt: usize,
    rate_bits: usize,
    inverse: bool,
) -> Vec<GoldilocksField> {
    let matrix: Vec<GoldilocksFieldBoojum> = unsafe { mem::transmute(matrix) };
    let length: usize = num_ntt << log_count;
    let degree: usize = 1 << log_count;
    let ctx = Context::create(12, 12).unwrap();
    let mut inputs_matrix_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length << rate_bits).unwrap();
    let mut outputs_matrix_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(
        length << rate_bits,
        CudaHostAllocFlags::DEFAULT,
    )
    .unwrap();
    let stream = CudaStream::default();
    memory_copy_async(&mut inputs_matrix_device, &matrix, &stream).unwrap();
    stream.synchronize().unwrap();
    let inputs_ptr = inputs_matrix_device.as_ptr();
    let outputs_ptr = outputs_matrix_device.as_mut_ptr();
    let s = Instant::now();
    batch_pad_coset(
        inputs_ptr,
        outputs_ptr,
        log_count as u32,
        num_ntt as u32,
        degree as u32,
        (degree << rate_bits) as u32,
        rate_bits as u32,
        inverse,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();
    println!("Time taken in gpu: {:?}", s.elapsed());
    memory_copy_async(&mut outputs_matrix_host, &outputs_matrix_device, &stream).unwrap();
    stream.synchronize().unwrap();
    let res_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(outputs_matrix_host.to_vec()) };

    ctx.destroy().unwrap();
    res_gpu
}

#[test]
fn pad_coset_test() {
    let log_count = 16;
    let num_ntt = 2;
    let rate_bits = 3;
    let inverse = false;
    let matrix = vec![GoldilocksField::from_noncanonical_u64(1); num_ntt << log_count];
    let res_gpu = pad_coset_packed(matrix.clone(), log_count, num_ntt, rate_bits, inverse);

    pad_coset_check(matrix, res_gpu, log_count, num_ntt, rate_bits, inverse);
    // println!("length: {:?}, last: {:?}", res_gpu.len(), res_gpu.last());
}

fn transpose_packed(
    matrix: Vec<GoldilocksField>,
    log_count: usize,
    num_ntt: usize,
) -> Vec<GoldilocksField> {
    let matrix: Vec<GoldilocksFieldBoojum> = unsafe { mem::transmute(matrix) };
    let length: usize = num_ntt << log_count;
    let degree: usize = 1 << log_count;
    let ctx = Context::create(12, 12).unwrap();
    let mut inputs_matrix_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_host =
        HostAllocation::<GoldilocksFieldBoojum>::alloc(length, CudaHostAllocFlags::DEFAULT)
            .unwrap();
    let stream = CudaStream::default();
    memory_copy_async(&mut inputs_matrix_device, &matrix, &stream).unwrap();
    stream.synchronize().unwrap();
    let inputs_ptr = inputs_matrix_device.as_ptr();
    let outputs_ptr = outputs_matrix_device.as_mut_ptr();
    let s = Instant::now();
    batch_transpose(
        inputs_ptr,
        outputs_ptr,
        log_count as u32,
        num_ntt as u32,
        degree as u32,
        degree as u32,
        0,
        &stream,
    )
    .unwrap();
    stream.synchronize().unwrap();
    println!("Time taken in gpu: {:?}", s.elapsed());
    memory_copy_async(&mut outputs_matrix_host, &outputs_matrix_device, &stream).unwrap();
    stream.synchronize().unwrap();
    let res_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(outputs_matrix_host.to_vec()) };

    ctx.destroy().unwrap();
    res_gpu
}

fn transpose_check(res_cpu: Vec<GoldilocksField>, res_gpu: Vec<GoldilocksField>, num_ntt: usize) {
    let length = res_cpu.len();
    assert_eq!(length, res_gpu.len());
    let mut count: usize = 0;
    for i in 0..length {
        // assert_eq!(res_cpu[i], res_gpu[i]);
        if res_cpu[i] != res_gpu[i] {
            println!(
                "{:?} != {:?} at row {:?}, col {:?}",
                res_cpu[i],
                res_gpu[i],
                i / num_ntt,
                i % num_ntt
            );
            count = count + 1;
            // break;
        }
    }
    println!("error count: {:?}", count);
}

#[test]
fn transpose_test() {
    let log_count = 21;
    let num_ntt = 234;
    // let matrix = vec![GoldilocksField::from_noncanonical_u64(1); num_ntt << log_count];
    let matrix: Vec<u64> = (0..(num_ntt << log_count) as u64).collect();
    let matrix: Vec<GoldilocksField> = unsafe { mem::transmute(matrix) };
    let res_gpu = transpose_packed(matrix.clone(), log_count, num_ntt);
    let res_cpu: Vec<Vec<GoldilocksField>> = matrix
        .clone()
        .par_chunks(1 << log_count)
        .map(|v| v.to_vec())
        .collect::<Vec<_>>();
    let res_cpu = transpose(&res_cpu).concat();
    transpose_check(res_cpu, res_gpu, num_ntt);
}

fn lde_packed(
    matrix: Vec<GoldilocksField>,
    log_count: usize,
    num_ntt: usize,
    rate_bits: usize,
) -> Vec<GoldilocksField> {
    let matrix: Vec<GoldilocksFieldBoojum> = unsafe { mem::transmute(matrix) };
    let length: usize = num_ntt << log_count;
    let degree: usize = 1 << log_count;
    let ctx = Context::create(12, 12).unwrap();
    let mut inputs_matrix_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length << rate_bits).unwrap();
    let mut buffer_matrix_device =
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length << rate_bits).unwrap();
    let mut outputs_matrix_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(
        length << rate_bits,
        CudaHostAllocFlags::DEFAULT,
    )
    .unwrap();
    let stream = CudaStream::default();
    memory_copy_async(&mut inputs_matrix_device, &matrix, &stream).unwrap();
    stream.synchronize().unwrap();
    // let inputs_ptr = inputs_matrix_device.as_ptr();
    // let outputs_ptr = outputs_matrix_device.as_mut_ptr();
    let s = Instant::now();
    // batch_lde_transpose_out_of_place(
    //     &inputs_matrix_device,
    //     &mut outputs_matrix_device,
    //     &mut buffer_matrix_device,
    //     log_count as u32,
    //     rate_bits as u32,
    //     num_ntt as u32,
    //     degree as u32,
    //     (degree << rate_bits) as u32,
    //     false,
    //     &stream,
    // ).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken in gpu: {:?}", s.elapsed());
    memory_copy_async(&mut outputs_matrix_host, &outputs_matrix_device, &stream).unwrap();
    stream.synchronize().unwrap();
    let res_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(outputs_matrix_host.to_vec()) };

    ctx.destroy().unwrap();
    res_gpu
}

#[test]
fn lde_test() {
    // don't use this!!
    let log_count = 18;
    let num_ntt = 20;
    let rate_bits = 3;
    let matrix = ntt_matrix_generate(log_count, num_ntt);
    let res_gpu = lde_packed(matrix.clone(), log_count, num_ntt, rate_bits);
    let res_cpu = matrix
        .clone()
        .par_chunks(1 << log_count)
        .map(|v| {
            PolynomialCoeffs::<GoldilocksField>::new(v.to_vec())
                .lde(rate_bits)
                .coset_fft_with_options(GoldilocksField::coset_shift(), Some(rate_bits), None)
                .values
        })
        .collect::<Vec<_>>();
    let res_cpu = transpose(&res_cpu).concat();
    ntt_check(res_cpu, res_gpu, log_count + rate_bits);
}
