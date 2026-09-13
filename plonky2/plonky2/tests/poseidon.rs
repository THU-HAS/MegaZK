// use boojum::field::Field;
use std::mem;
//use std::fmt::format;
use std::time::Instant;

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use boojum::implementations::poseidon_goldilocks_naive::poseidon_permutation;
// use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use cudart::stream::CudaStream;
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2::fri::oracle::PolynomialBatch;
use plonky2::hash::hash_types::HashOut;
use plonky2::hash::merkle_tree::MerkleTree;
use plonky2::hash::poseidon::Poseidon;
use plonky2::integration::poseidon::{leaves_gpu, merkle_tree_gpu, ChallengerGpu};
use plonky2::iop::challenger::Challenger;
use plonky2::plonk::config::{GenericConfig, Hasher, PoseidonGoldilocksConfig};
use plonky2_field::packed::PackedField;
//use plonky2_field::packed::PackedField;
use plonky2_field::types::{Field, PrimeField64};
use plonky2_maybe_rayon::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

pub const GODILOCKS_PRIME: u64 = (1 << 32 - 1) * (1 << 32) + 1;

fn leaves_generate(log_n: usize, values_per_row: usize) -> Vec<Vec<GoldilocksFieldBoojum>> {
    let mut matrix = Vec::<Vec<GoldilocksFieldBoojum>>::new();
    let count: usize = 1 << log_n;
    let mut rand = ChaChaRng::from_entropy();
    for _i in 0..count {
        let mut col = Vec::<GoldilocksFieldBoojum>::new();
        for _j in 0..values_per_row {
            col.push(GoldilocksFieldBoojum::from_nonreduced_u64(
                rand.gen_range(0..GODILOCKS_PRIME),
            ));
        }
        matrix.push(col);
    }

    matrix
}

fn merkle_tree_check(log_n: usize, values_per_row: usize, cap_height: usize) {
    let leaves = leaves_generate(log_n, values_per_row);
    let mut leaves_copy = Vec::<Vec<GoldilocksField>>::new();
    for i in 0..(1 << log_n) {
        let mut leave = vec![GoldilocksField::ZERO; values_per_row];
        for j in 0..values_per_row {
            leave[j] = GoldilocksField::from_noncanonical_u64(leaves[i][j].to_nonreduced_u64());
        }
        leaves_copy.push(leave);
    }
    //println!("{:?}", leaves_copy);

    let mtree_cpu = MerkleTree::<
        GoldilocksField,
        <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher,
    >::new(leaves_copy, cap_height);
    //println!("cpu: {:?}", mtree_cpu);

    let mtree_gpu = merkle_tree_gpu(leaves.clone(), cap_height);
    let length = mtree_gpu.len();
    let cap = mtree_gpu[(length - (8 << cap_height))..(length - (4 << cap_height))].to_vec();

    println!(
        "gpu digest: {:?}",
        mtree_gpu[..(length - (4 << cap_height))].to_vec()
    );
    println!("gpu cap: {:?}", cap);
    println!();

    // let n_digest = mtree_cpu.digests.len();
    // assert_eq!(n_digest, mtree_cpu.digests.len() * 4);
    // for i in 0..n_digest {
    //     for j in 0..4 {
    //         assert_eq!(mtree_gpu.digests[i * 4 + j].to_reduced_u64(), mtree_cpu.digests[i].elements[j].to_canonical_u64());
    //     }
    // }

    // let n_cap = mtree_cpu.cap.len();
    // assert_eq!(n_cap, mtree_cpu.cap.len() * 4);
    // for i in 0..n_cap {
    //     for j in 0..4 {
    //         assert_eq!(mtree_gpu.cap[i * 4 + j].to_reduced_u64(), mtree_cpu.cap.0[i].elements[j].to_canonical_u64());
    //     }
    // }
}

#[test]
fn merkle_tree_test() {
    for log_n in vec![3 as usize] {
        for values_per_row in vec![1 as usize] {
            merkle_tree_check(log_n, values_per_row, 1);
        }
    }
}

#[test]
fn one_poseidon() {
    let nums =
        // vec![8314404939152679185, 10675796363475469191, 6571029491187171278, 7535015655168091968,
        //  9957013299265509049, 14613711300596661171, 13527720676955999632, 16005260041927912660 as u64];
        vec![1 as u64; 8];
    let len = nums.len();

    let leave = unsafe { mem::transmute::<Vec<u64>, Vec<GoldilocksField>>(nums) };
    let res = <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher::hash_or_noop(&leave);
    println!("{:?}", res);
    let res_hash = <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher::hash_or_noop(&leave);
    // let res_hash = unsafe {
    //     mem::transmute::<HashOut<GoldilocksField>, Vec<GoldilocksField>>(res_hash)
    // };
}

#[test]
fn digests_test() {
    let a: usize = 9;
    let b: usize = 4;
    println!("{:?}", a / b);
}

#[test]
fn digest_test() {
    let n = 1345734;
    // let leaves = leaves_generate(log_n, values_per_row);
    let leaves = vec![vec![GoldilocksFieldBoojum::from_nonreduced_u64(n); 15]; 1];
    let leave = vec![GoldilocksField::from_noncanonical_u64(n); 15];
    // let mut leaves_0 = vec![GoldilocksFieldBoojum::from_nonreduced_u64(n); 8];
    // leaves_0.append(&mut vec![GoldilocksFieldBoojum::from_nonreduced_u64(0); 4]);
    // let leaves_0: &mut [GoldilocksFieldBoojum; 12] = match leaves_0.as_mut_slice().try_into() {
    //     Ok(s) => s,
    //     Err(_) => panic!()
    // };
    //leaves_copy[0].append(&mut vec![GoldilocksField::ZERO; 11]);
    println!("{:x?}", leaves);
    // println!("{:x?}", leaves_0);
    let res_gpu = leaves_gpu(leaves);
    println!(" res gpu: {:x?}", res_gpu);
    let res_cpu = <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher::hash_or_noop(&leave);
    println!("res cpu: {:x?}", res_cpu);
    // poseidon_permutation(leaves_0);
    // println!("{:x?}", leaves_0);
}

#[test]
fn poseidon_cpu_test() {
    let n = 0;
    // plonky2
    let leave = vec![GoldilocksField::from_noncanonical_u64(n); 8];
    let mut leave_padded = leave.clone();
    leave_padded.append(&mut vec![GoldilocksField::ZERO; 4]);
    let res = <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher::hash_or_noop(&leave);
    // let res_padded = <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher::hash_or_noop(&leave_padded);
    println!("{:x?}", res);
    // println!("{:x?}", res_padded);

    // boojum
    let leave = vec![GoldilocksFieldBoojum::from_nonreduced_u64(n); 8];
    let mut leave_padded = leave.clone();
    leave_padded.append(&mut vec![GoldilocksFieldBoojum::from_nonreduced_u64(0); 4]);
    // let leave: &mut [GoldilocksFieldBoojum; 12] = match leave.as_mut_slice().try_into() {
    //     Ok(s) => s,
    //     Err(_) => panic!()
    // };
    let leave_padded: &mut [GoldilocksFieldBoojum; 12] =
        match leave_padded.as_mut_slice().try_into() {
            Ok(s) => s,
            Err(_) => panic!(),
        };
    //poseidon_permutation(leave);
    poseidon_permutation(leave_padded);
    // println!("{:x?}", leave);
    println!("{:x?}", leave_padded);
}

#[test]
fn challenger_test() {
    let n_ch = 9;
    let c = 2;
    let n_in = 29;
    let mut clg = ChallengerGpu::new();
    let elements = vec![GoldilocksFieldBoojum::from_nonreduced_u64(c); n_in];
    clg.observe_elements(&elements).unwrap();
    clg.observe_elements(&elements).unwrap();
    let res_gpu = clg.get_n_challenges(n_ch);
    println!("{:?}", res_gpu);
    clg.observe_elements(&elements).unwrap();
    let res_gpu = clg.get_n_challenges(n_ch);
    println!("{:?}", res_gpu);

    let mut sponge_host =
        HostAllocation::<GoldilocksFieldBoojum>::alloc(12, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut ib_host =
        HostAllocation::<GoldilocksFieldBoojum>::alloc(8, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut ob_host =
        HostAllocation::<GoldilocksFieldBoojum>::alloc(8, CudaHostAllocFlags::DEFAULT).unwrap();
    let stream = CudaStream::default();
    memory_copy_async(&mut sponge_host, &clg.sponge_state, &stream).unwrap();
    memory_copy_async(&mut ib_host, &clg.input_buffer, &stream).unwrap();
    memory_copy_async(&mut ob_host, &clg.output_buffer, &stream).unwrap();
    stream.synchronize().unwrap();
    // println!("{:?}", sponge_host.to_vec());
    // println!("{:?}", ib_host.to_vec());
    // println!("{:?}", ob_host.to_vec());

    let mut challenger =
        Challenger::<GoldilocksField, <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher>::new(
        );
    let input = vec![GoldilocksField::from_canonical_u64(c); n_in];
    challenger.observe_elements(&input);
    challenger.observe_elements(&input);
    let res_cpu = challenger.get_n_challenges(n_ch);
    println!("{:x?}", res_cpu);
    challenger.observe_elements(&input);
    let res_cpu = challenger.get_n_challenges(n_ch);
    println!("{:x?}", res_cpu);
}

#[test]
fn challenger_test1() {
    let n_ch = 4;
    let c = 2;
    let n_in = 4;
    let mut clg = ChallengerGpu::new();
    let elements = vec![GoldilocksFieldBoojum::from_nonreduced_u64(c); n_in];
    let elements3 = vec![GoldilocksFieldBoojum::from_nonreduced_u64(c); 4 * n_in];
    clg.observe_elements(&elements).unwrap();
    clg.observe_elements(&elements).unwrap();
    // let res_gpu = clg.get_n_challenges(n_ch);
    // println!("rg {:?}", res_gpu);
    clg.observe_elements(&elements).unwrap();
    clg.observe_elements(&elements).unwrap();
    let res_gpu = clg.get_n_challenges(n_ch);
    println!("rg {:?}", res_gpu);

    let mut clg1 = ChallengerGpu::new();
    clg1.show_sponge();
    clg1.observe_elements(&elements3).unwrap();
    let res_gpu = clg1.get_n_challenges(n_ch);
    println!("rg {:?}", res_gpu);

    let mut sponge_host =
        HostAllocation::<GoldilocksFieldBoojum>::alloc(12, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut ib_host =
        HostAllocation::<GoldilocksFieldBoojum>::alloc(8, CudaHostAllocFlags::DEFAULT).unwrap();
    let mut ob_host =
        HostAllocation::<GoldilocksFieldBoojum>::alloc(8, CudaHostAllocFlags::DEFAULT).unwrap();
    let stream = CudaStream::default();
    memory_copy_async(&mut sponge_host, &clg.sponge_state, &stream).unwrap();
    memory_copy_async(&mut ib_host, &clg.input_buffer, &stream).unwrap();
    memory_copy_async(&mut ob_host, &clg.output_buffer, &stream).unwrap();
    stream.synchronize().unwrap();
    // println!("{:?}", sponge_host.to_vec());
    // println!("{:?}", ib_host.to_vec());
    // println!("{:?}", ob_host.to_vec());

    let mut challenger =
        Challenger::<GoldilocksField, <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher>::new(
        );
    let input = vec![GoldilocksField::from_canonical_u64(c); n_in];
    let input3 = vec![GoldilocksField::from_canonical_u64(c); 4 * n_in];
    challenger.observe_elements(&input);
    challenger.observe_elements(&input);
    // let res_cpu = challenger.get_n_challenges(n_ch);
    // println!("rc {:x?}", res_cpu);
    challenger.observe_elements(&input);
    challenger.observe_elements(&input);
    let res_cpu = challenger.get_n_challenges(n_ch);
    println!("rc {:x?}", res_cpu);

    let mut challenger =
        Challenger::<GoldilocksField, <PoseidonGoldilocksConfig as GenericConfig<2>>::Hasher>::new(
        );
    challenger.observe_elements(&input3);
    let res_cpu = challenger.get_n_challenges(n_ch);
    println!("rc {:x?}", res_cpu);
}
