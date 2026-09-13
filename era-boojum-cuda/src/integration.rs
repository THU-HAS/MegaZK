use crate::context::OMEGA_LOG_ORDER;
use boojum::field::goldilocks::GoldilocksField;
use boojum::gadgets::num;
use cudart::cuda_kernel;
use cudart::error::get_last_error;
use cudart::execution::{CudaLaunchConfig, KernelFunction};
use cudart::result::{CudaResult, CudaResultWrap};
use cudart::slice::DeviceSlice;
use cudart::stream::CudaStream;

cuda_kernel!(
    Integration,
    integration_kernel,
    inputs_matrix: *const GoldilocksField,
    outputs_matrix: *mut GoldilocksField,
    log_n: u32,
    num_ntts: u32,
    stride_between_input_arrays: u32,
    stride_between_output_arrays: u32,
    rate_bits: u32,
    inverse: bool,
);

integration_kernel!(reverse_index_bits);
integration_kernel!(pad_coset);
integration_kernel!(copy_gpu);
integration_kernel!(transpose_gpu);

pub fn batch_pad_coset(
    inputs_ptr: *const GoldilocksField,
    outputs_ptr: *mut GoldilocksField,
    log_n: u32,
    num_ntts: u32,
    stride_between_input_arrays: u32,
    stride_between_output_arrays: u32,
    rate_bits: u32,
    inverse: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    let threads: u32 = 128;
    let n: u32 = 1 << log_n;
    let blocks_per_ntt: u32 = (n + 32 * threads - 1) / (32 * threads);
    let blocks = blocks_per_ntt * num_ntts;
    // let inputs_ptr = inputs_matrix.as_ptr();
    // let outputs_ptr = outputs_matrix.as_mut_ptr();

    let config = CudaLaunchConfig::basic(blocks, threads, stream);
    let args = IntegrationArguments::new(
        inputs_ptr,
        outputs_ptr,
        log_n,
        num_ntts,
        stride_between_input_arrays,
        stride_between_output_arrays,
        rate_bits,
        inverse,
    );
    IntegrationFunction(pad_coset).launch(&config, &args)?;
    get_last_error().wrap()
}

pub fn batch_reverse_index_bits(
    inputs_ptr: *const GoldilocksField,
    outputs_ptr: *mut GoldilocksField,
    log_n: u32,
    num_ntts: u32,
    stride_between_input_arrays: u32,
    stride_between_output_arrays: u32,
    rate_bits: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let threads: u32 = 32;
    let elements_per_thread = threads * threads;
    let n: u32 = 1 << log_n;
    let blocks_per_ntt: u32 = (n + elements_per_thread - 1) / elements_per_thread;
    let blocks = blocks_per_ntt * num_ntts;
    // println!("{:?}, {:?}", blocks_per_ntt, blocks);
    // let inputs_ptr = inputs_matrix.as_ptr();
    // let outputs_ptr = outputs_matrix.as_mut_ptr();

    let config = CudaLaunchConfig::basic(blocks, threads, stream);
    let args = IntegrationArguments::new(
        inputs_ptr,
        outputs_ptr,
        log_n,
        num_ntts,
        stride_between_input_arrays,
        stride_between_output_arrays,
        rate_bits,
        false,
    );
    IntegrationFunction(reverse_index_bits).launch(&config, &args)?;

    get_last_error().wrap()
}

cuda_kernel!(RevId, reverse_index_bits_inplace(
    inputs_matrix_ptr: *mut GoldilocksField,
    log_n: u32,
    num_ntts: u32,
));

pub fn batch_reverse_index_bits_inplace(
    inputs_ptr: *mut GoldilocksField,
    log_n: u32,
    num_ntts: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let threads: u32 = 32;
    let elements_per_thread = threads * threads;
    let n: u32 = 1 << log_n;
    let blocks_per_ntt: u32 = (n + elements_per_thread - 1) / elements_per_thread;
    let blocks = blocks_per_ntt * num_ntts;

    let config = CudaLaunchConfig::basic(blocks, threads, stream);
    let args = RevIdArguments::new(inputs_ptr, log_n, num_ntts);
    RevIdFunction(reverse_index_bits_inplace).launch(&config, &args)?;

    get_last_error().wrap()
}

pub fn batch_transpose(
    inputs_ptr: *const GoldilocksField,
    outputs_ptr: *mut GoldilocksField,
    log_n: u32,
    num_ntts: u32,
    stride_between_input_arrays: u32,
    stride_between_output_arrays: u32,
    rate_bits: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let threads: u32 = 32;
    let blocks_row: u32 = (num_ntts + threads - 1) / threads;
    let blocks_col: u32 = (1 << log_n) / threads;
    let blocks = blocks_row * blocks_col;
    // println!("{:?}, {:?}", blocks_per_ntt, blocks);
    // let inputs_ptr = inputs_matrix.as_ptr();
    // let outputs_ptr = outputs_matrix.as_mut_ptr();

    let config = CudaLaunchConfig::basic(blocks, threads, stream);
    let args = IntegrationArguments::new(
        inputs_ptr,
        outputs_ptr,
        log_n,
        num_ntts,
        stride_between_input_arrays,
        stride_between_output_arrays,
        rate_bits,
        false,
    );
    IntegrationFunction(transpose_gpu).launch(&config, &args)?;

    get_last_error().wrap()
}

pub fn batch_copy(
    inputs_matrix: &DeviceSlice<GoldilocksField>,
    outputs_matrix: &mut DeviceSlice<GoldilocksField>,
    log_n: u32,
    num_polys: u32,
    offset_polys: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let threads: u32 = 128;
    let n: u32 = 1 << log_n;
    let blocks: u32 = (n + threads - 1) / threads;

    let inputs_ptr = inputs_matrix.as_ptr();
    let outputs_ptr = outputs_matrix.as_mut_ptr();

    let config = CudaLaunchConfig::basic(blocks, threads, stream);
    let args = IntegrationArguments::new(
        inputs_ptr,
        outputs_ptr,
        log_n,
        num_polys,
        offset_polys,
        n,
        0,
        false,
    );
    IntegrationFunction(copy_gpu).launch(&config, &args)?;

    get_last_error().wrap()
}

cuda_kernel!(NTTE, n2b_1_stage_e(
    inputs_matrix: *const GoldilocksField,
    root_pow: *const GoldilocksField,
    outputs_matrix: *mut GoldilocksField,
    log_n: u32,
    blocks_per_ntt: u32,
    start_stage: u32,
));

pub fn ntt_extension_inplace(
    inputs_matrix: &mut DeviceSlice<GoldilocksField>,
    log_n: u32,
    num_ntts: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let threads: u32 = 128;
    let n: u32 = 1 << log_n;
    let blocks_per_ntt: u32 = (n + 2 * threads - 1) / (2 * threads);
    let blocks = blocks_per_ntt * num_ntts;
    let config = CudaLaunchConfig::basic(blocks, threads, stream);
    for stage in 0..log_n {
        let args = NTTEArguments::new(
            inputs_matrix.as_ptr(),
            inputs_matrix.as_ptr(),
            inputs_matrix.as_mut_ptr(),
            log_n,
            blocks_per_ntt,
            stage,
        );
        NTTEFunction(n2b_1_stage_e).launch(&config, &args)?;
    }
    get_last_error().wrap()
}

cuda_kernel!(CosetExt, coset_e(
    inputs: *const GoldilocksField,
    outputs: *mut GoldilocksField,
    coset: GoldilocksField,
    degree_bits: u32,
    num_polys: u32,
    stride_between_input_arrays: u32,
    stride_between_output_arrays: u32,
));

pub fn batch_pad_coset_ext(
    inputs_ptr: *const GoldilocksField,
    outputs_ptr: *mut GoldilocksField,
    coset: GoldilocksField,
    log_n: u32,
    num_ntts: u32,
    stride_between_input_arrays: u32,
    stride_between_output_arrays: u32,
    inverse: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    let threads: u32 = 128;
    let n: u32 = 1 << log_n;
    let blocks_per_ntt: u32 = (n + 32 * threads - 1) / (32 * threads);
    let blocks = blocks_per_ntt;
    let config = CudaLaunchConfig::basic(blocks, threads, stream);
    let args = CosetExtArguments::new(
        inputs_ptr,
        outputs_ptr,
        coset,
        log_n,
        1,
        stride_between_input_arrays,
        stride_between_output_arrays,
    );
    CosetExtFunction(coset_e).launch(&config, &args)?;
    get_last_error().wrap()
}

cuda_kernel!(ReduceExt, reduce_e(
    inputs: *const GoldilocksField,
    beta: *const GoldilocksField,
    outputs: *mut GoldilocksField,
    num_chunks: u32,
    chunk_size: u32,
));

pub fn batch_reduce_ext(
    inputs: *const GoldilocksField,
    beta: *const GoldilocksField,
    outputs: *mut GoldilocksField,
    num_chunks: u32,
    chunk_size: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let threads: u32 = 128;
    // let n: u32 = 1 << log_n;
    // let blocks_per_ntt: u32 = (n + 32 * threads - 1) / (32 * threads);
    let blocks = (num_chunks + threads - 1) / threads;
    let config = CudaLaunchConfig::basic(blocks, threads, stream);
    let args = ReduceExtArguments::new(inputs, beta, outputs, num_chunks, chunk_size);
    ReduceExtFunction(reduce_e).launch(&config, &args)?;
    get_last_error().wrap()
}

cuda_kernel!(MulExp, mul_exp(
    inputs: *mut GoldilocksField,
    log_n: u32,
    num_ntts: u32,
    stride_between_input_arrays: u32,
    mul_val: GoldilocksField,
));

pub fn batch_mul_exp(
    inputs_ptr: *mut GoldilocksField,
    log_n: u32,
    num_ntts: u32,
    stride_between_input_arrays: u32,
    mul_val: GoldilocksField,
    stream: &CudaStream,
) -> CudaResult<()> {
    let threads: u32 = 128;
    let n: u32 = 1 << log_n;
    let point_per_thread = 32; // no longer 32!
    let blocks_per_ntt: u32 = (n + point_per_thread * threads - 1) / (point_per_thread * threads);
    let blocks = blocks_per_ntt * num_ntts;

    let config = CudaLaunchConfig::basic(blocks, threads, stream);
    let args = MulExpArguments::new(
        inputs_ptr,
        log_n,
        num_ntts,
        stride_between_input_arrays,
        mul_val,
    );
    MulExpFunction(mul_exp).launch(&config, &args)?;
    get_last_error().wrap()
}
