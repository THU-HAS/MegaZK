use crate::integration::batch_mul_exp;
use crate::integration::batch_pad_coset;
use crate::ntt::*;
use crate::poseidon::*;
use std::mem;
use std::time::Instant;

use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};
use crate::BaseField;
use boojum::field::goldilocks::GoldilocksField;
use boojum::field::Field;
use boojum::field::PrimeField;
use boojum::implementations::poseidon_goldilocks_params::*;
use cudart::cuda_kernel;
use cudart::error::get_last_error;
use cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use cudart::result::{CudaResult, CudaResultWrap};
use cudart::slice::CudaSliceMut;
use cudart::slice::DeviceSlice;
use cudart::stream::CudaStream;
use cudart_sys::cudaMemset;

type BF = BaseField;

cuda_kernel!(
    Leaves,
    leaves_kernel,
    values: *const BF,
    results: *mut BF,
    rows_count: u32,
    cols_count: u32,
    count: u32,
    load_intermediate: bool,
    store_intermediate: bool,
);

leaves_kernel!(poseidon_single_thread_leaves_kernel);

cuda_kernel!(
    LeavesPartial,
    leaves_partial_kernel,
    static_inputs: *const BF,
    dynamic_inputs: *const BF,
    results: *mut BF,
    log_n: u32,
    rate_bits: u32,
    cols_count: u32,
    lde_cols_count: u32,
    part: u32,
);

leaves_partial_kernel!(poseidon_single_thread_leaves_partial_kernel);

pub fn launch_leaves_kernel_ptr<P: PoseidonImpl>(
    // kernel_function: LeavesSignature,
    get_grid_block_fn: fn(u32) -> (Dim3, Dim3),
    values: *const GoldilocksField,
    results: *mut GoldilocksField,
    values_len: usize,
    results_len: usize,
    log_rows_per_hash: u32,
    load_intermediate: bool,
    store_intermediate: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    P::unique_asserts();
    // let values_len = values.len();
    // let results_len = results.len();
    // assert_eq!(results_len % CAPACITY, 0);
    let count = results_len / CAPACITY / 3;
    // assert_eq!(values_len % (count << log_rows_per_hash), 0);
    // let values = values.as_ptr();
    // let results = results.as_mut_ptr();
    let rows_count = 1u32 << log_rows_per_hash;
    let cols_count = values_len / (count << log_rows_per_hash);
    assert!(cols_count <= u32::MAX as usize);
    let cols_count = cols_count as u32;
    assert!(count <= u32::MAX as usize);
    // If this launch computes an intermediate result for a partial set of columns,
    // the kernels assume we'll complete a permutation for a full state before writing
    // the result for the current columns. This imposes a restriction on the number
    // of columns we may include in the partial set.
    assert!(!store_intermediate || ((rows_count * cols_count) % RATE as u32 == 0));
    let count = count as u32;
    let (grid_dim, block_dim) = get_grid_block_fn(count);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = LeavesArguments::new(
        values,
        results,
        rows_count,
        cols_count,
        count,
        load_intermediate,
        store_intermediate,
    );
    LeavesFunction(poseidon_single_thread_leaves_kernel).launch(&config, &args)
}

pub fn build_merkle_tree_leaves_segmented<P: PoseidonImpl>(
    values: &DeviceSlice<GoldilocksField>,
    buffer: &mut DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    log_n: u32,
    rate_bits: u32,
    num_ntts: u32,
    inputs_offset: u32,
    log_rows_per_hash: u32,
    // load_intermediate: bool,
    // store_intermediate: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    let values_len = values.len();
    let results_len = results.len();
    let buffer_len = buffer.len();
    assert_eq!(results_len % CAPACITY, 0);
    let leaves_count = results_len / CAPACITY / 3;
    // assert_eq!(values_len % leaves_count, 0);
    // select 1/8 values
    // calculate lde and place into buffer
    // calculate intermediate hash and place into poseidon_device
    // repeat
    let degree = 1 << log_n;
    let rate: u32 = 1 << rate_bits;
    let buffer_lde_num = (buffer_len >> log_n + rate_bits) as u32;
    let num_ntt_per_round = buffer_lde_num / (RATE as u32) * (RATE as u32);
    let not_divisible = num_ntts % num_ntt_per_round != 0;
    let total_rounds: u32 = (num_ntts / num_ntt_per_round) + not_divisible as u32;
    for i in 0..total_rounds {
        let offset: usize = (inputs_offset + (i * num_ntt_per_round << log_n)) as usize;
        let round_ntt = if i < total_rounds - 1 {
            num_ntt_per_round
        } else {
            num_ntts - num_ntt_per_round * i
        };
        unsafe {
            let _ = cudaMemset((*buffer).as_mut_c_void_ptr(), 0, buffer_len * 8);
        }
        batch_lde_transpose_out_of_place_ptr(
            values.as_ptr().wrapping_add(offset),
            buffer.as_mut_ptr(),
            log_n,
            rate_bits,
            round_ntt,
            degree,
            degree << rate_bits,
            false,
            stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        launch_leaves_kernel_ptr::<P>(
            P::get_grid_block_leaves_single_thread,
            buffer.as_ptr(),
            results.as_mut_ptr(),
            (round_ntt << log_n + rate_bits) as usize,
            results_len,
            log_rows_per_hash,
            i > 0,
            i < total_rounds - 1,
            stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
    }
    get_last_error().wrap()
}

pub fn build_merkle_tree_segmented<P: PoseidonImpl>(
    values: &DeviceSlice<GoldilocksField>,
    buffer: &mut DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    log_n: u32,
    rate_bits: u32,
    num_ntts: u32,
    inputs_offset: u32,
    log_rows_per_hash: u32,
    stream: &CudaStream,
    layers_count: u32,
) -> CudaResult<()> {
    let s = Instant::now();
    build_merkle_tree_leaves_segmented::<P>(
        values,
        buffer,
        results,
        log_n,
        rate_bits,
        num_ntts,
        inputs_offset,
        log_rows_per_hash,
        stream,
    )?;
    stream.synchronize().unwrap();
    println!("  leaves: {:?}", s.elapsed());
    let results_len = results.len();
    // let (nodes, nodes_remaining) = results.split_at_mut(results.len() >> 1);
    let (nodes, nodes_remaining0) = results.split_at_mut(results_len / 3);
    let (nodes_remaining, unused) = nodes_remaining0.split_at_mut(results_len / 3);
    let s = Instant::now();
    build_merkle_tree_nodes::<P>(nodes, nodes_remaining, layers_count - 1, stream)?;
    stream.synchronize().unwrap();
    println!("  tree: {:?}", s.elapsed());
    get_last_error().wrap()
}

pub fn build_merkle_tree_leaves_part<P: PoseidonImpl>(
    values: &DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    rate_bits: u32,
    inputs_offset: usize,
    inputs_len: usize,
    outputs_offset: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    P::unique_asserts();
    let values_len = inputs_len;
    let results_len = results.len() >> rate_bits;
    assert_eq!(results_len % CAPACITY, 0);
    let count = results_len / CAPACITY;
    assert_eq!(values_len % count, 0);
    let values = values.as_ptr().wrapping_add(inputs_offset);
    let results = results.as_mut_ptr().wrapping_add(outputs_offset);
    let rows_count = 1;
    let cols_count = values_len / count;
    assert!(cols_count <= u32::MAX as usize);
    let cols_count = cols_count as u32;
    assert!(count <= u32::MAX as usize);
    // If this launch computes an intermediate result for a partial set of columns,
    // the kernels assume we'll complete a permutation for a full state before writing
    // the result for the current columns. This imposes a restriction on the number
    // of columns we may include in the partial set.
    let count = count as u32;
    let (grid_dim, block_dim) = P::get_grid_block_leaves_single_thread(count);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = LeavesArguments::new(values, results, rows_count, cols_count, count, false, false);
    LeavesFunction(P::LEAVES_SINGLE_THREAD_FUNCTION).launch(&config, &args)
}

pub fn build_merkle_tree_leaves_part_w_partial_ldes<P: PoseidonImpl>(
    static_input_ptr: *const GoldilocksField,
    dynamic_input_ptr: *const GoldilocksField,
    output_ptr: *mut GoldilocksField,
    log_n: u32,
    rate_bits: u32,
    num_ntts: u32,
    num_ldes: u32,
    part: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    // let values_len = inputs_len;
    // let results_len = results.len() >> rate_bits;
    // assert_eq!(results_len % CAPACITY, 0);
    // let count = results_len / CAPACITY;
    // assert_eq!(values_len % count, 0);
    // let values = values.as_ptr().wrapping_add(inputs_offset);
    // let results = results.as_mut_ptr().wrapping_add(outputs_offset);
    // let rows_count = 1;
    // let cols_count = values_len / count;
    // assert!(cols_count <= u32::MAX as usize);
    // let cols_count = cols_count as u32;
    // assert!(count <= u32::MAX as usize);
    // If this launch computes an intermediate result for a partial set of columns,
    // the kernels assume we'll complete a permutation for a full state before writing
    // the result for the current columns. This imposes a restriction on the number
    // of columns we may include in the partial set.
    // let count = count as u32;
    let (grid_dim, block_dim) = P::get_grid_block_leaves_single_thread(1 << log_n);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = LeavesPartialArguments::new(
        static_input_ptr,
        dynamic_input_ptr,
        output_ptr,
        log_n,
        rate_bits,
        num_ntts,
        num_ldes,
        part,
    );
    LeavesPartialFunction(poseidon_single_thread_leaves_partial_kernel).launch(&config, &args)
}

pub fn build_merkle_tree_in_place<P: PoseidonImpl>(
    values: &mut DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    log_n: u32,
    rate_bits: u32,
    num_ntts: u32,
    inputs_offset: usize,
    stream: &CudaStream,
    layers_count: u32,
) -> CudaResult<()> {
    // coset & NTT (n2b)
    // compress
    // INTT (b2n) & mul coset & NTT (n2b)
    // compress
    // ...
    // build merkle tree nodes
    let degree = 1 << log_n;
    let rate = 1 << rate_bits;
    let mut mul_val = GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR;
    let lde_twiddle_factor =
        GoldilocksField::RADIX_2_SUBGROUP_GENERATOR.pow_u64(1 << (32 - log_n - rate_bits));
    let (nodes, nodes_remaining) = results.split_at_mut(results.len() >> 1);
    for i in 0..rate as usize {
        batch_mul_exp(
            values.as_mut_ptr().wrapping_add(inputs_offset),
            log_n,
            num_ntts,
            degree,
            mul_val,
            stream,
        )
        .unwrap();

        // ntt_n2b
        batch_ntt_internal(
            values.as_ptr().wrapping_add(inputs_offset),
            values.as_mut_ptr().wrapping_add(inputs_offset),
            log_n as u32,
            num_ntts as u32,
            degree,
            degree,
            false,
            false,
            0,
            0,
            false,
            &stream,
        )
        .unwrap();
        // build_merkle_tree_leaves_part
        build_merkle_tree_leaves_part::<P>(
            values,
            nodes,
            rate_bits,
            inputs_offset,
            (num_ntts << log_n) as usize,
            (i.reverse_bits() >> (64 - rate_bits)) * (nodes.len() >> rate_bits),
            &stream,
        )
        .unwrap();

        // intt_b2n
        batch_ntt_internal(
            values.as_ptr().wrapping_add(inputs_offset),
            values.as_mut_ptr().wrapping_add(inputs_offset),
            log_n as u32,
            num_ntts as u32,
            degree,
            degree,
            true,
            true,
            0,
            0,
            false,
            &stream,
        )
        .unwrap();

        mul_val = lde_twiddle_factor;
    }

    // remove coset (perhaps)
    mul_val = (GoldilocksField::RADIX_2_SUBGROUP_GENERATOR
        .pow_u64((rate - 1) << (32 - log_n - rate_bits))
        * GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR)
        .inverse()
        .expect("Failed to calculate inverse!");
    batch_mul_exp(
        values.as_mut_ptr().wrapping_add(inputs_offset),
        log_n,
        num_ntts,
        degree,
        mul_val,
        stream,
    )
    .unwrap();
    build_merkle_tree_nodes::<P>(nodes, nodes_remaining, layers_count - 1, stream)?;

    get_last_error().wrap()
}

pub fn build_merkle_tree_w_partial_ldes<P: PoseidonImpl>(
    ldes: &DeviceSlice<GoldilocksField>,
    values: &mut DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    log_n: u32,
    rate_bits: u32,
    num_ntts: u32,
    num_ldes: u32,
    ldes_offset: usize,
    values_offset: usize,
    stream: &CudaStream,
    layers_count: u32,
) -> CudaResult<()> {
    // coset & NTT (n2b)
    // compress
    // INTT (b2n) & mul coset & NTT (n2b)
    // compress
    // ...
    // build merkle tree nodes
    let degree = 1 << log_n;
    let rate = 1 << rate_bits;
    let inputs_offset = values_offset + (num_ldes << log_n) as usize;
    let mut mul_val = GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR;
    let lde_twiddle_factor =
        GoldilocksField::RADIX_2_SUBGROUP_GENERATOR.pow_u64(1 << (32 - log_n - rate_bits));
    let (nodes, nodes_remaining) = results.split_at_mut(results.len() >> 1);
    for i in 0..rate as usize {
        if num_ntts - num_ldes > 0 {
            batch_mul_exp(
                values.as_mut_ptr().wrapping_add(inputs_offset),
                log_n,
                num_ntts - num_ldes,
                degree,
                mul_val,
                stream,
            )
            .unwrap();

            // ntt_n2b
            batch_ntt_internal(
                values.as_ptr().wrapping_add(inputs_offset),
                values.as_mut_ptr().wrapping_add(inputs_offset),
                log_n as u32,
                num_ntts - num_ldes,
                degree,
                degree,
                false,
                false,
                0,
                0,
                false,
                &stream,
            )
            .unwrap();
        }
        // build_merkle_tree_leaves_part
        let rev_i = i.reverse_bits() >> (64 - rate_bits);
        build_merkle_tree_leaves_part_w_partial_ldes::<P>(
            ldes.as_ptr().wrapping_add(ldes_offset),
            values.as_mut_ptr().wrapping_add(values_offset),
            nodes
                .as_mut_ptr()
                .wrapping_add(rev_i * (nodes.len() >> rate_bits)),
            log_n,
            rate_bits,
            num_ntts,
            num_ldes,
            rev_i as u32,
            stream,
        )
        .unwrap();

        // intt_b2n
        if num_ntts - num_ldes > 0 {
            batch_ntt_internal(
                values.as_ptr().wrapping_add(inputs_offset),
                values.as_mut_ptr().wrapping_add(inputs_offset),
                log_n as u32,
                num_ntts - num_ldes,
                degree,
                degree,
                true,
                true,
                0,
                0,
                false,
                &stream,
            )
            .unwrap();

            mul_val = lde_twiddle_factor;
        }
    }

    // remove coset (perhaps)
    if num_ntts - num_ldes > 0 {
        mul_val = (GoldilocksField::RADIX_2_SUBGROUP_GENERATOR
            .pow_u64((rate - 1) << (32 - log_n - rate_bits))
            * GoldilocksField::MULTIPLICATIVE_GROUP_GENERATOR)
            .inverse()
            .expect("Failed to calculate inverse!");
        batch_mul_exp(
            values.as_mut_ptr().wrapping_add(inputs_offset),
            log_n,
            num_ntts - num_ldes,
            degree,
            mul_val,
            stream,
        )
        .unwrap();
    }
    build_merkle_tree_nodes::<P>(nodes, nodes_remaining, layers_count - 1, stream)?;

    get_last_error().wrap()
}
