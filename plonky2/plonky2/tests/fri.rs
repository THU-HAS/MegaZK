// use boojum::field::{Field, U64Representable};
use boojum::field::U64Representable;
use boojum::sha3::digest::typenum::Unsigned;
use boojum::utils::PipeOp;
use cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use cudart::stream::CudaStream;
// use num::{PrimInt, Unsigned};
// use plonky2::integration::quotient::QuotientSetup;
use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2_field::extension::quadratic::QuadraticExtension;
use plonky2_field::polynomial::PolynomialCoeffs;
use plonky2_field::types::{Field, PrimeField64};
use plonky2_field::zero_poly_coset::ZeroPolyOnCoset;
use plonky2::plonk::config::Hasher;
use plonky2::util::reverse_index_bits_in_place;

use boojum_cuda::integration::ntt_extension_inplace;
use std::mem;

#[test]
fn ntt_idx() {
    let log_n: u32 = 3;
    let length: usize = 1 << log_n;
    let mut inputs_matrix_device = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length * 2).unwrap();
    let input_data: Vec<_> = (0..length).map(|i| {
        vec![GoldilocksFieldBoojum::from_nonreduced_u64(i as u64), GoldilocksFieldBoojum::from_nonreduced_u64(0)]
    }).collect::<Vec<_>>().concat();
    println!("{:?}", input_data);
    let stream = CudaStream::default();
    memory_copy_async(&mut inputs_matrix_device, &input_data, &stream).unwrap();

    // let mut outputs_matrix_device = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    // let mut outputs_matrix_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(length, CudaHostAllocFlags::DEFAULT).unwrap();
    // memory_copy_async(&mut inputs_matrix_device, &matrix, &stream).unwrap();
    // stream.synchronize().unwrap();
    ntt_extension_inplace(&mut inputs_matrix_device, log_n, 1, &stream).unwrap();

}

#[test]
fn ntt_ext() {
    let log_n: u32 = 15;
    let length: usize = 1 << log_n;
    let mut inputs_matrix_device = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length * 2).unwrap();
    let mut outputs_matrix_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(length * 2, CudaHostAllocFlags::DEFAULT).unwrap();
    let input_data: Vec<_> = (0..length).map(|i| {
        vec![GoldilocksFieldBoojum::from_nonreduced_u64(i as u64), GoldilocksFieldBoojum::from_nonreduced_u64(0)]
    }).collect::<Vec<_>>().concat();
    // println!("{:?}", input_data);
    let stream = CudaStream::default();
    memory_copy_async(&mut inputs_matrix_device, &input_data, &stream).unwrap();

    // let mut outputs_matrix_device = DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    // let mut outputs_matrix_host = HostAllocation::<GoldilocksFieldBoojum>::alloc(length, CudaHostAllocFlags::DEFAULT).unwrap();
    // memory_copy_async(&mut inputs_matrix_device, &matrix, &stream).unwrap();
    // stream.synchronize().unwrap();
    ntt_extension_inplace(&mut inputs_matrix_device, log_n, 1, &stream).unwrap();
    memory_copy_async(&mut outputs_matrix_host, &inputs_matrix_device, &stream).unwrap();
    stream.synchronize().unwrap();
    let res_gpu: Vec<GoldilocksField> = unsafe { mem::transmute(outputs_matrix_host.to_vec())};
    // println!("gpu: {:?}", res_gpu);

    let coeffs = (0..length).map(|i| {
        QuadraticExtension::<GoldilocksField>::from_canonical_u64(i as u64)
    }).collect::<Vec<_>>();
    let input_data_cpu = PolynomialCoeffs::<QuadraticExtension<GoldilocksField>>::new(coeffs);
    let mut res_cpu = input_data_cpu.coset_fft(QuadraticExtension::<GoldilocksField>::from_canonical_u64(1));
    // println!("cpu input: {:?}", input_data_cpu);
    reverse_index_bits_in_place(&mut res_cpu.values);
    // println!("cpu: {:x?}", res_cpu.values);

    for i in 0..length {
        assert_eq!(res_cpu.values[i].0[0].to_noncanonical_u64(), res_gpu[2 * i].to_noncanonical_u64());
        assert_eq!(res_cpu.values[i].0[1].to_noncanonical_u64(), res_gpu[2 * i + 1].to_noncanonical_u64());
    }
}