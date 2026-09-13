
use alloc::vec::Vec;
use plonky2_field::goldilocks_field::GoldilocksField;
use std::time::Instant;
use std::mem;

use plonky2_maybe_rayon::*;

use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::hash::hash_types::RichField;
use crate::util::{log2_strict, reverse_index_bits_in_place};

use boojum::field::goldilocks::GoldilocksField as GoldilocksFieldBoojum;
use cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};
use cudart::stream::CudaStream;
use boojum_cuda::context::Context;
use boojum_cuda::ntt::batch_ntt_out_of_place;

pub fn intt_gpu<F: RichField>(
    values: Vec<PolynomialValues<F>>,
    rate_bits: usize,
    coset: bool,
) -> Vec<PolynomialCoeffs<F>> {
    let num_ntts = values.len();
    let count_o = values[0].values.len();
    let count = values[0].values.len() << rate_bits;
    let log_count = log2_strict(count);
    let length = count * num_ntts;
    assert_eq!(count, 1 << log_count);
    println!("Start intt 2^{}, num {}", log_count, num_ntts);
    let s = Instant::now();
    let ctx = Context::create(12, 12).unwrap();
    let mut inputs_matrix_device = 
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_device = 
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_host = 
        HostAllocation::<GoldilocksFieldBoojum>::alloc(length, CudaHostAllocFlags::DEFAULT).unwrap();
    let stream = CudaStream::default();
    println!("Time taken to allocate the memory: {:?}", s.elapsed());

    let s = Instant::now();
    //let mut inputs_matrix_host = vec![GoldilocksFieldBoojum::ZERO; length];
    let mut inputs_matrix_host = Vec::<GoldilocksFieldBoojum>::new();
    // inputs_matrix_host.resize(length, ...zero)

    // // initialize a full zero vector with length length
    // // begin time
    // let sx = Instant::now();
    // let data = vec![123u64; length];
    // println!("Time: {:?}", sx.elapsed());

    // let sx = Instant::now();
    // let data2 = data.iter().map(|e| GoldilocksFieldBoojum::from_nonreduced_u64(*e)).collect::<Vec<_>>();

    // println!("Time: {:?}", sx.elapsed());
    // //use 
    // let sum = data2[3];
    // println!("Sum: {:?} {:?}", data[3], sum);

    if !coset {
        // (0..num_ntts as usize).into_par_iter().for_each(|i| {
        //     //let mut values_padded = values[i].values.clone();
        //     //values_padded.resize(count, F::ZERO);
        //     (0..count_o).for_each(|j| {
        //         inputs_matrix_host[i * count + j] = GoldilocksFieldBoojum::from_nonreduced_u64(values[i].values[j].to_noncanonical_u64());
        // });
        // });
        (0..num_ntts as usize).for_each(|i| {
            let mut values_padded = values[i].values.clone();
            values_padded.resize(count, F::ZERO);
            inputs_matrix_host.append(&mut values_padded
                .into_par_iter()
                .map(|v| GoldilocksFieldBoojum::from_nonreduced_u64(v.to_canonical_u64()))
                .collect::<Vec<GoldilocksFieldBoojum>>()
            )
        });
        

        // (0..num_ntts as usize).for_each(|i| {
        //     let mut values_padded = values[i].values.clone();
        //     values_padded.resize(count, F::ZERO);
        //     inputs_matrix_host.append(&mut values_padded
        //         .into_par_iter()
        //         //.map(|v| GoldilocksFieldBoojum::from_nonreduced_u64(v.to_canonical_u64()))
        //         .map(|v| {
        //             let u: GoldilocksFieldBoojum;
        //             unsafe { u = mem::transmute::<u64, GoldilocksFieldBoojum>(v.to_noncanonical_u64())}
        //             u
        //         })
        //         .collect::<Vec<GoldilocksFieldBoojum>>()
        //     )
        // });

        // values.par_iter()
        //     .enumerate()
        //     .for_each(|(i, v)| {
        //         let mut vec_padded = v.values.clone();
        //         vec_padded.resize(count, F::ZERO);
        //         inputs_matrix_host.append(
        //             &mut vec_padded
        //             .into_iter()
        //             .map(|v| GoldilocksFieldBoojum::from_nonreduced_u64(v.to_canonical_u64()))
        //             .collect::<Vec<GoldilocksFieldBoojum>>()
        //     )
        //     })

        // let s1 = Instant::now();
        // input_matrix_host.par_chunk(count).zip(0..num_ntt).map({|(dest, idx)| {
        //     let &source= coeeff[idx].coeef;
        //     source.zipwithindex(|(eleemnt, rowidx)| {
        //         dest[rowIdx] = xxxxx (eleemtn;)
        //     }

        //     // second half, zero , nothing todo as we are already zero initialized
        // }})


        // // // let result = values.into_par_iter().flat_map(|vec| {
        // // let result = values.into_iter().flat_map(|vec| {
        // //     let mut vec_padded = vec.values.clone();
        // //     vec_padded.resize(count, F::ZERO);

        // //     vec_padded.iter().map(|t| {
        // //         // let x = GoldilocksFieldBoojum::from_nonreduced_u64(t.to_noncanonical_u64());
        // //         GoldilocksFieldBoojum::from_nonreduced_u64(0)
        // //     }).collect::<Vec<_>>()

        // // }).collect::<Vec<_>>();
        // let result = vec![GoldilocksFieldBoojum::ZERO; length];
        // println!("Time taken to type cast NEW: {:?}", s1.elapsed());
        // inputs_matrix_host = result;
    } else {
        (0..num_ntts as usize).for_each(|i| {
            let mut values_padded = values[i].values.clone();
            values_padded.resize(count, F::ZERO);
            inputs_matrix_host.append(&mut F::MULTIPLICATIVE_GROUP_GENERATOR
                .powers()
                .zip(values_padded)
                .map(|(c, m)| {
                    GoldilocksFieldBoojum::from_nonreduced_u64((c * m).to_canonical_u64())
                })
                .collect::<Vec<GoldilocksFieldBoojum>>()
            )
        });
    }
    println!("Time taken to cast type: {:?}", s.elapsed());
    let s = Instant::now();
    memory_copy_async(&mut inputs_matrix_device, &inputs_matrix_host, &stream).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken to copy h2d: {:?}", s.elapsed());

    let s = Instant::now();
    {
        batch_ntt_out_of_place(
            &mut inputs_matrix_device,
            &mut outputs_matrix_device,
            log_count as u32,
            num_ntts as u32,
            0,
            0,
            count as u32,
            count as u32,
            false,
            true,
            0,
            0,
            &stream,
        ).unwrap();
    }
    stream.synchronize().unwrap();
    println!("Time taken to calculate: {:?}", s.elapsed());
    let s = Instant::now();
    memory_copy_async(&mut outputs_matrix_host, &outputs_matrix_device, &stream).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken to copy d2h: {:?}", s.elapsed());

    let s = Instant::now();
    let mut coeffs = Vec::<PolynomialCoeffs<F>>::new();
    (0..num_ntts as usize).for_each(|i| {
        coeffs.push(PolynomialCoeffs::<F>::new(Vec::<F>::new()));
        outputs_matrix_host[(i * count)..((i + 1) * count)]
            .into_par_iter()
            .map(|c| F::from_noncanonical_u64(c.to_nonreduced_u64()))
            .collect_into_vec(&mut coeffs[i].coeffs);
        reverse_index_bits_in_place(&mut coeffs[i].coeffs)
    });
    println!("Time taken to cast type: {:?}", s.elapsed());

    stream.destroy().unwrap();
    inputs_matrix_device.free().unwrap();
    outputs_matrix_device.free().unwrap();
    outputs_matrix_host.free().unwrap();
    ctx.destroy().unwrap();
    coeffs
}

pub fn ntt_gpu<F: RichField>(
    coeffs: Vec<PolynomialCoeffs<F>>,
    rate_bits: usize,
    coset: bool,
) -> Vec<PolynomialValues<F>> {
    let num_ntts = coeffs.len();
    let count = coeffs[0].coeffs.len() << rate_bits;
    let log_count = log2_strict(count);
    let length = count * num_ntts;
    assert_eq!(count, 1 << log_count);
    println!("Start ntt 2^{}, num {}", log_count, num_ntts);
    let s = Instant::now();
    let ctx = Context::create(12, 12).unwrap();
    let mut inputs_matrix_device = 
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_device = 
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_host = 
        HostAllocation::<GoldilocksFieldBoojum>::alloc(length, CudaHostAllocFlags::DEFAULT).unwrap();
    let stream = CudaStream::default();
    println!("Time taken to allocate memory: {:?}", s.elapsed());

    let s = Instant::now();
    let mut inputs_matrix_host = Vec::<GoldilocksFieldBoojum>::new();
    if !coset {
        // (0..num_ntts as usize).for_each(|i| {
        //     let mut coeffs_padded = coeffs[i].coeffs.clone();
        //     coeffs_padded.resize(count, F::ZERO);

        //    let mut x= coeffs_padded.par_chunks_mut(1024).flat_map(
        //         |chunk| {
        //             chunk.iter().map(|t| {
        //                 GoldilocksFieldBoojum::from_nonreduced_u64(t.to_canonical_u64())
        //             })
        //         }.collect::<Vec<_>>()
        //     ).collect::<Vec<_>>();

        //     inputs_matrix_host.append(&mut x)
            
        // });
        let result = coeffs.into_par_iter().flat_map(|vec| {
            let mut vec_padded = vec.coeffs.clone();
            vec_padded.resize(count, F::ZERO);

            vec_padded.iter().map(|t| GoldilocksFieldBoojum::from_nonreduced_u64(t.to_canonical_u64())).collect::<Vec<_>>()

        }).collect::<Vec<_>>();
        inputs_matrix_host = result;
    } else {
        (0..num_ntts as usize).for_each(|i| {
            let mut coeffs_padded = coeffs[i].coeffs.clone();
            coeffs_padded.resize(count, F::ZERO);
            inputs_matrix_host.append(&mut F::MULTIPLICATIVE_GROUP_GENERATOR
                .powers()
                .zip(coeffs_padded)
                .map(|(c, m)| {
                    GoldilocksFieldBoojum::from_nonreduced_u64((c * m).to_canonical_u64())
                })
                .collect::<Vec<GoldilocksFieldBoojum>>()
            )
        });
    }
    println!("Time taken to cast type: {:?}", s.elapsed());
    let s = Instant::now();
    memory_copy_async(&mut inputs_matrix_device, &inputs_matrix_host, &stream).unwrap();
    println!("Time taken to copy h2d: {:?}", s.elapsed());
    
    let s = Instant::now();
    {
        batch_ntt_out_of_place(
            &mut inputs_matrix_device,
            &mut outputs_matrix_device,
            log_count as u32,
            num_ntts as u32,
            0,
            0,
            count as u32,
            count as u32,
            false,
            false,
            0,
            0,
            &stream,
        ).unwrap();
    }
    println!("Time taken to calculate: {:?}", s.elapsed());
    let s = Instant::now();
    memory_copy_async(&mut outputs_matrix_host, &outputs_matrix_device, &stream).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken to copy d2h: {:?}", s.elapsed());

    let s = Instant::now();
    let mut values = Vec::<PolynomialValues<F>>::new();
    (0..num_ntts as usize).for_each(|i| {
        values.push(PolynomialValues::<F>::new(Vec::<F>::new()));
        outputs_matrix_host[(i * count)..((i + 1) * count)]
            .into_par_iter()
            .map(|c| F::from_noncanonical_u64(c.to_nonreduced_u64()))
            .collect_into_vec(&mut values[i].values);
        // reverse_index_bits_in_place(&mut values[i].values)
    });
    println!("Time taken to cast type: {:?}", s.elapsed());

    stream.destroy().unwrap();
    inputs_matrix_device.free().unwrap();
    outputs_matrix_device.free().unwrap();
    outputs_matrix_host.free().unwrap();
    ctx.destroy().unwrap();
    values
}

pub fn lde_gpu<F: RichField>(
    coeffs: &Vec<PolynomialCoeffs<F>>,
    rate_bits: usize,
    coset: bool,
) -> Vec<Vec<F>> {
    let num_ntts = coeffs.len();
    let count = coeffs[0].coeffs.len() << rate_bits;
    let log_count = log2_strict(count);
    let length = count * num_ntts;
    assert_eq!(count, 1 << log_count);
    println!("Start lde 2^{}, rate bits {}, num {}", log_count, rate_bits, num_ntts);
    let s = Instant::now();
    let ctx = Context::create(12, 12).unwrap();
    let mut inputs_matrix_device = 
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_device = 
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_host = 
        HostAllocation::<GoldilocksFieldBoojum>::alloc(length, CudaHostAllocFlags::DEFAULT).unwrap();
    let stream = CudaStream::default();
    println!("Time taken to allocate the memory: {:?}", s.elapsed());

    let s = Instant::now();
    let mut inputs_matrix_host = Vec::<GoldilocksFieldBoojum>::new();
    if !coset {
        (0..num_ntts as usize).for_each(|i| {
            let mut coeffs_padded = coeffs[i].coeffs.clone();
            coeffs_padded.resize(count, F::ZERO);
            inputs_matrix_host.append(&mut coeffs_padded
                .into_par_iter()
                .map(|v| GoldilocksFieldBoojum::from_nonreduced_u64(v.to_canonical_u64()))
                .collect::<Vec<GoldilocksFieldBoojum>>()
            )
        });
    } else {
        (0..num_ntts as usize).for_each(|i| {
            let mut coeffs_padded = coeffs[i].coeffs.clone();
            coeffs_padded.resize(count, F::ZERO);
            inputs_matrix_host.append(&mut F::MULTIPLICATIVE_GROUP_GENERATOR
                .powers()
                .zip(coeffs_padded)
                .map(|(c, m)| {
                    GoldilocksFieldBoojum::from_nonreduced_u64((c * m).to_canonical_u64())
                })
                .collect::<Vec<GoldilocksFieldBoojum>>()
            )
        });
    }
    println!("Time taken to cast type: {:?}", s.elapsed());
    let s = Instant::now();
    memory_copy_async(&mut inputs_matrix_device, &inputs_matrix_host, &stream).unwrap();
    println!("Time taken to copy h2d: {:?}", s.elapsed());
    
    let s = Instant::now();
    {
        batch_ntt_out_of_place(
            &mut inputs_matrix_device,
            &mut outputs_matrix_device,
            log_count as u32,
            num_ntts as u32,
            0,
            0,
            count as u32,
            count as u32,
            false,
            false,
            0,
            0,
            &stream,
        ).unwrap();
    }
    println!("Time taken to calculate: {:?}", s.elapsed());
    let s = Instant::now();
    memory_copy_async(&mut outputs_matrix_host, &outputs_matrix_device, &stream).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken to copy d2h: {:?}", s.elapsed());

    let s = Instant::now();
    let mut values = Vec::<Vec<F>>::new();
    (0..num_ntts as usize).for_each(|i| {
        values.push(Vec::<F>::new());
        outputs_matrix_host[(i * count)..((i + 1) * count)]
            .into_par_iter()
            .map(|c| F::from_noncanonical_u64(c.to_nonreduced_u64()))
            .collect_into_vec(&mut values[i]);
        reverse_index_bits_in_place(&mut values[i])
    });
    println!("Time taken to cast type: {:?}", s.elapsed());

    stream.destroy().unwrap();
    inputs_matrix_device.free().unwrap();
    outputs_matrix_device.free().unwrap();
    outputs_matrix_host.free().unwrap();
    ctx.destroy().unwrap();
    values
}

pub fn intt_gpu_nocast(
    values: Vec<PolynomialValues<GoldilocksField>>,
    rate_bits: usize,
    coset: bool,
) -> Vec<PolynomialCoeffs<GoldilocksField>> {
    let num_ntts = values.len();
    let count_o = values[0].len();
    let count = values[0].len() << rate_bits;
    let log_count = log2_strict(count);
    let length = count * num_ntts;
    assert_eq!(count, 1 << log_count);
    println!("Start intt 2^{}, num {}", log_count, num_ntts);
    let s = Instant::now();
    let ctx = Context::create(12, 12).unwrap();
    println!("Time taken to create context: {:?}", s.elapsed());
    let s = Instant::now();
    let mut inputs_matrix_device = 
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_device = 
        DeviceAllocation::<GoldilocksFieldBoojum>::alloc(length).unwrap();
    let mut outputs_matrix_host = 
        HostAllocation::<GoldilocksFieldBoojum>::alloc(length, CudaHostAllocFlags::DEFAULT).unwrap();
    let stream = CudaStream::default();
    println!("Time taken to allocate the memory: {:?}", s.elapsed());

    let s = Instant::now();
    //let mut inputs_matrix_host = vec![GoldilocksFieldBoojum::ZERO; length];
    // let mut inputs_matrix_host = Vec::<GoldilocksFieldBoojum>::new();

    // (0..num_ntts).for_each(|i| {
    //     inputs_matrix_host.append(&mut unsafe {
    //         mem::transmute::<PolynomialValues<GoldilocksField>, Vec<GoldilocksFieldBoojum>>(values[i].clone())
    //     });
    // });
    //inputs_matrix_host = concat(values);
    let inputs_matrix_host: Vec<Vec<GoldilocksFieldBoojum>> = values.into_par_iter()
        .map(|v| unsafe {mem::transmute::<PolynomialValues<GoldilocksField>, Vec<GoldilocksFieldBoojum>>(v)})
        .collect();
    println!("Time taken to cast type 0: {:?}", s.elapsed());
    let s = Instant::now();
    let inputs_matrix_host = inputs_matrix_host.concat();

    println!("Time taken to cast type: {:?}", s.elapsed());
    let s = Instant::now();
    memory_copy_async(&mut inputs_matrix_device, &inputs_matrix_host, &stream).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken to copy h2d: {:?}", s.elapsed());

    let s = Instant::now();
    {
        batch_ntt_out_of_place(
            &mut inputs_matrix_device,
            &mut outputs_matrix_device,
            log_count as u32,
            num_ntts as u32,
            0,
            0,
            count as u32,
            count as u32,
            false,
            true,
            0,
            0,
            &stream,
        ).unwrap();
    }
    stream.synchronize().unwrap();
    println!("Time taken to calculate: {:?}", s.elapsed());
    let s = Instant::now();
    memory_copy_async(&mut outputs_matrix_host, &outputs_matrix_device, &stream).unwrap();
    stream.synchronize().unwrap();
    println!("Time taken to copy d2h: {:?}", s.elapsed());

    // let s = Instant::now();
    // let mut coeffs = Vec::<PolynomialCoeffs<GoldilocksField>>::new();
    // //outputs_matrix_host.clone_into(&mut coeffs);
    // (0..num_ntts as usize).for_each(|i| {
    //     coeffs.push(unsafe {
    //         mem::transmute::<Vec<GoldilocksFieldBoojum>, PolynomialCoeffs<GoldilocksField>>(outputs_matrix_host[(i * count)..((i + 1) * count)].to_vec())
    //     });
    //     //reverse_index_bits_in_place(&mut coeffs[i].coeffs)
    // });
    // println!("Time taken to cast type: {:?}", s.elapsed());
    let mut coeffs = Vec::<PolynomialCoeffs<GoldilocksField>>::new();
    let s = Instant::now();
    outputs_matrix_host.par_chunks(count)
        .map(|v| unsafe {mem::transmute::<Vec<GoldilocksFieldBoojum>, PolynomialCoeffs<GoldilocksField>>(v.to_vec())})
        .collect_into_vec(&mut coeffs);
    println!("Time taken to cast type: {:?}", s.elapsed());

    // let s = Instant::now();
    // stream.destroy().unwrap();
    // inputs_matrix_device.free().unwrap();
    // outputs_matrix_device.free().unwrap();
    // outputs_matrix_host.free().unwrap();
    // ctx.destroy().unwrap();
    // println!("Time taken to free resources: {:?}", s.elapsed());
    coeffs
}