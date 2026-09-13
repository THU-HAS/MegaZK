use std::mem;
use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use plonky2::field::goldilocks_field::GoldilocksField;

use plonky2::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;
pub const GODILOCKS_PRIME: u64 = (1 << 32 - 1) * (1 << 32) + 1;

use plonky2::integration::ntt::{intt_gpu, intt_gpu_nocast, ntt_gpu};


fn intt_matrix_generate(
    log_count: usize,
    ncol: usize,
) -> Vec<PolynomialValues<GoldilocksField>> {
    let mut matrix = Vec::<PolynomialValues<GoldilocksField>>::new();
    let count: u32 = 1 << log_count;
    let mut rand = ChaChaRng::from_entropy();
    for _i in 0..ncol {
        let mut col = Vec::<GoldilocksField>::new();
        for _j in 0..count {
            col.push(GoldilocksField::from_canonical_u64(rand.gen_range(0..GODILOCKS_PRIME)));
        }
        let col = PolynomialValues::<GoldilocksField>::new(col);
        matrix.push(col);
    }

    matrix
}

fn intt_check(
    log_count: usize,
    num_ntts: usize,
    rate_bits: usize,
    coset: bool,
) {
    // generate data
    let matrix = intt_matrix_generate(log_count, num_ntts);
    let matrix_copied = matrix.clone();
    
    // gpu intt
    let s0 = Instant::now();
    let res_gpu = 
        intt_gpu_nocast(matrix, rate_bits, coset);
    let t0 = format!("{:?}", s0.elapsed());
    // cpu intt
    let s1 = Instant::now();
    let res_cpu = 
        matrix_copied.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>();
    let t1 = format!("{:?}", s1.elapsed());

    // check result
    assert_eq!(num_ntts, res_gpu.len());
    assert_eq!(1 << log_count, res_gpu[0].coeffs.len());
    // let mut wrong_count = vec![0 as usize; num_ntts];
    // let mut zero_count = vec![0 as usize; num_ntts];
    // for i in 0..num_ntts {
    //     for j in 0..(1 << log_count) {
    //         if res_gpu[i].coeffs[j] != res_cpu[i].coeffs[j] {
    //             wrong_count[i] += 1;
    //             if res_gpu[i].coeffs[j] == GoldilocksField::ZERO {
    //                 zero_count[i] += 1;
    //             }
    //         }
    //         //*
    //         assert_eq!(
    //             res_gpu[i].coeffs[j], res_cpu[i].coeffs[j],
    //             "Natural to bitrev in-place results incorrect for size 2^{}, ntt {}/{}, coset {}, index {}",
    //             log_count, i, num_ntts, coset, j
    //         )
    //         //*/
    //     }
    // }
    // assert_eq!(
    //                 res_gpu[0].coeffs[4], res_cpu[0].coeffs[4],
    //                 "Natural to bitrev in-place results incorrect for size 2^{}, ntt {}/{}, coset {}, index {}",
    //                 log_count, 0, num_ntts, coset, 4
    //             );
    println!(
        "size 2^{}, ntt {}, coset {}, gpu {}, cpu {}",
        log_count, num_ntts, coset, t0, t1
    )
}

fn try_test() {
    intt_check(18, 1, 0, false);
    for log_count in vec![15, 18, 21] {
        for num_ntts in vec![1, 10] {
            // let log_count = 18;
            // let num_ntts = 1;
            let rate_bits = 0;
            let coset = false;
            // generate data
            let matrix = intt_matrix_generate(log_count, num_ntts);
            let matrix_copied = matrix.clone();
            
            // gpu intt
            let s0 = Instant::now();
            let res_gpu = matrix.into_par_iter()
                .map(|v| {
                    intt_gpu_nocast(vec![v], rate_bits, coset)
                })
                .collect::<Vec<_>>();
            // let res_gpu = 
            //     intt_gpu_nocast(matrix, rate_bits, coset);
            let t0 = format!("{:?}", s0.elapsed());
            // cpu intt
            let s1 = Instant::now();
            let res_cpu = 
                matrix_copied.into_par_iter().map(|v| v.ifft()).collect::<Vec<_>>();
            let t1 = format!("{:?}", s1.elapsed());

            assert_eq!(
                res_gpu[0][0].coeffs[4], res_cpu[0].coeffs[4],
                "Natural to bitrev in-place results incorrect for size 2^{}, ntt {}/{}, coset {}, index {}",
                log_count, 0, num_ntts, coset, 4
            );
            println!(
                "size 2^{}, ntt {}, coset {}, gpu {}, cpu {}",
                log_count, num_ntts, coset, t0, t1
            )
        }
    }
}

fn type_cast_test() {
    for (log_count, num_ntts) in vec![(15usize, 200usize), (18usize, 200usize), (21usize, 200usize)] {
        println!("Generating Matrix...");
        let values = intt_matrix_generate(log_count, num_ntts);
        println!("2^{log_count}, num {num_ntts}");
        let s = Instant::now();
        let mut inputs_matrix_host = Vec::<GoldilocksFieldBoojum>::new();
        (0..num_ntts).for_each(|i| {
            inputs_matrix_host.append(&mut unsafe {
                mem::transmute::<PolynomialValues<GoldilocksField>, Vec<GoldilocksFieldBoojum>>(values[i].clone())
            });
        });
        println!("castA: {:?}", s.elapsed());

        let s = Instant::now();
        let inputs_matrix_host: Vec<Vec<GoldilocksFieldBoojum>> = values.par_iter()
            .map(|v| unsafe {mem::transmute::<PolynomialValues<GoldilocksField>, Vec<GoldilocksFieldBoojum>>(v.clone())})
            .collect();
        println!("castB 0: {:?}", s.elapsed());
        let s = Instant::now();
        let inputs_matrix_host = inputs_matrix_host.concat();
        println!("castB 1: {:?}", s.elapsed());

        let s = Instant::now();
        let inputs_matrix_host: Vec<GoldilocksFieldBoojum> = values.par_iter()
            .flat_map(|v| unsafe {mem::transmute::<PolynomialValues<GoldilocksField>, Vec<GoldilocksFieldBoojum>>(v.clone())})
            .collect();
        println!("castC: {:?}", s.elapsed());

        let s = Instant::now();
        let inputs_matrix_host = values.into_par_iter()
            .map(|v| unsafe {mem::transmute::<PolynomialValues<GoldilocksField>, Vec<GoldilocksFieldBoojum>>(v)})
            .collect::<Vec<_>>()
            .concat();
        println!("castD 0: {:?}", s.elapsed());
        // let s = Instant::now();
        // let inputs_matrix_host = inputs_matrix_host.concat();
        // println!("castD 1: {:?}", s.elapsed());
    }
}

fn intt_test() {
    intt_check(18, 1, 0, false);
    for log_count in vec![21 as usize] {
        for num_ntts in vec![234 as usize] {
            intt_check(log_count, num_ntts, 0, false);
        }
    }
}

pub fn main() {
    intt_test();
}