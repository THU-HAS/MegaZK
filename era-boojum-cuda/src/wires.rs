use cudart::stream::CudaStream;
use boojum::field::goldilocks::GoldilocksField;
use cudart::cuda_kernel;
use cudart::execution::{Dim3, CudaLaunchConfig, KernelFunction};
use cudart::slice::DeviceSlice;
use cudart::result::CudaResult;

cuda_kernel!(WPPP, wires_permutation_partial_products_kernel(
    witness_device: *const GoldilocksField,
    subgroup_device: *const GoldilocksField,
    k_is_device: *const GoldilocksField,
    s_sigmas_device: *const GoldilocksField,
    quotient_values_device: *mut GoldilocksField,
    degree: usize,
    degree2: usize,
    betas_device: *const GoldilocksField,
    gammas_device: *const GoldilocksField,
));

cuda_kernel!(WPPPT, wires_permutation_partial_products_kernel_trans(
    witness_device: *const GoldilocksField,
    subgroup_device: *const GoldilocksField,
    k_is_device: *const GoldilocksField,
    s_sigmas_device: *const GoldilocksField,
    quotient_values_device: *mut GoldilocksField,
    degree: usize,
    degree2: usize,
    betas_device: *const GoldilocksField,
    gammas_device: *const GoldilocksField,
));

pub fn wires_permutation_partial_products(
    witness_device: &DeviceSlice<GoldilocksField>,
    subgroup_device: &DeviceSlice<GoldilocksField>,
    k_is_device: &DeviceSlice<GoldilocksField>,
    s_sigmas_device: &DeviceSlice<GoldilocksField>,
    quotient_values_device: &mut DeviceSlice<GoldilocksField>,
    degree: usize,
    degree2:usize,
    betas_device: &DeviceSlice<GoldilocksField>,
    gammas_device: &DeviceSlice<GoldilocksField>,
    block_x: u32,
    block_y: u32,
    thread_x: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let witness_device = witness_device.as_ptr();
    let subgroup_device = subgroup_device.as_ptr();
    let k_is_device = k_is_device.as_ptr();
    let s_sigmas_device = s_sigmas_device.as_ptr();
    let betas_device = betas_device.as_ptr();
    let gammas_device = gammas_device.as_ptr();

    let quotient_values_device = quotient_values_device.as_mut_ptr();

    let kernel_function = WPPPFunction::default();

    let num_blocks = Dim3::from((block_x, block_y));
    let threads_per_block = Dim3::from(thread_x);
    let config = CudaLaunchConfig::basic(num_blocks, threads_per_block, &stream);
    let args = WPPPArguments {
        witness_device,
        subgroup_device,
        k_is_device,
        s_sigmas_device,
        quotient_values_device,
        degree,
        degree2,
        betas_device,
        gammas_device,
    };
    kernel_function.launch(&config, &args)
}

pub fn wires_permutation_partial_products_trans_ptr(
    witness_device: *const GoldilocksField,
    subgroup_device: *const GoldilocksField,
    k_is_device: *const GoldilocksField,
    s_sigmas_device: *const GoldilocksField,
    quotient_values_device: *mut GoldilocksField,
    degree: usize,
    degree2:usize,
    betas_device: *const GoldilocksField,
    gammas_device: *const GoldilocksField,
    block_x: u32,
    block_y: u32,
    thread_x: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let kernel_function = WPPPTFunction::default();

    let num_blocks = Dim3::from((block_x, block_y));
    let threads_per_block = Dim3::from(thread_x);
    let config = CudaLaunchConfig::basic(num_blocks, threads_per_block, &stream);
    let args = WPPPTArguments {
        witness_device,
        subgroup_device,
        k_is_device,
        s_sigmas_device,
        quotient_values_device,
        degree,
        degree2,
        betas_device,
        gammas_device,
    };
    kernel_function.launch(&config, &args)
}

cuda_kernel!(ScanPart, scan_part_kernel(
    quotient_values_device: *const GoldilocksField,
    part_device: *mut GoldilocksField,
    all_partial_products_and_zs_device: *mut GoldilocksField,
    length: usize
));

pub fn scan_part_gpu(
    quotient_values_device: &DeviceSlice<GoldilocksField>,
    part_device: &mut DeviceSlice<GoldilocksField>,
    all_partial_products_and_zs_device: &mut DeviceSlice<GoldilocksField>,
    length: usize,
    block_x: u32,
    block_y: u32,
    thread_x: u32,
    stream: &CudaStream,
) -> CudaResult<()> {

    let quotient_values_device = quotient_values_device.as_ptr();
    let all_partial_products_and_zs_device = all_partial_products_and_zs_device.as_mut_ptr();
    let part_device = part_device.as_mut_ptr();

    let kernel_function = ScanPartFunction::default();

    let num_blocks = Dim3::from((block_x, block_y));
    let threads_per_block = Dim3::from(thread_x);
    let config = CudaLaunchConfig::basic(num_blocks, threads_per_block, &stream);
    let args = ScanPartArguments {
        quotient_values_device,
        part_device,
        all_partial_products_and_zs_device,
        length
    };
    kernel_function.launch(&config, &args)
}

cuda_kernel!(ScanPartProduct, scan_part_product_kernel(
    part_device: *mut GoldilocksField,
    part_num: usize
));

pub fn scan_part_product_gpu(
    part_device: &mut DeviceSlice<GoldilocksField>,
    part_num: usize,
    block_x: u32,
    block_y: u32,
    thread_x: u32,
    stream: &CudaStream,
) -> CudaResult<()> {

    let part_device = part_device.as_mut_ptr();

    let kernel_function = ScanPartProductFunction::default();

    let num_blocks = Dim3::from((block_x, block_y));
    let threads_per_block = Dim3::from(thread_x);
    let config = CudaLaunchConfig::basic(num_blocks, threads_per_block, &stream);
    let args = ScanPartProductArguments {
        part_device,
        part_num
    };
    kernel_function.launch(&config, &args)
}

cuda_kernel!(MulPartProduct, mul_part_product_kernel(
    part_device: *const GoldilocksField,
    all_partial_products_device: *mut GoldilocksField,
    length: usize
));

pub fn mul_part_product_gpu(
    part_device: &DeviceSlice<GoldilocksField>,
    all_partial_products_device: &mut DeviceSlice<GoldilocksField>,
    length: usize,
    block_x: u32,
    block_y: u32,
    thread_x: u32,
    stream: &CudaStream,
) -> CudaResult<()> {

    let part_device = part_device.as_ptr();
    let all_partial_products_device = all_partial_products_device.as_mut_ptr();

    let kernel_function = MulPartProductFunction::default();

    let num_blocks = Dim3::from((block_x, block_y));
    let threads_per_block = Dim3::from(thread_x);
    let config = CudaLaunchConfig::basic(num_blocks, threads_per_block, &stream);
    let args = MulPartProductArguments {
        part_device,
        all_partial_products_device,
        length
    };
    kernel_function.launch(&config, &args)
}


cuda_kernel!(MatrixTrans, matrix_trans_kernel(
    all_partial_products_device: *const GoldilocksField,
    zs_partial_products_lookups_device: *mut GoldilocksField
));

pub fn matrix_trans_gpu(
    all_partial_products_device: &DeviceSlice<GoldilocksField>,
    zs_partial_products_lookups_device: &mut DeviceSlice<GoldilocksField>,
    zs_offset: usize,
    block_x: u32,
    block_y: u32,
    thread_x: u32,
    stream: &CudaStream,
) -> CudaResult<()> {

    let all_partial_products_device = all_partial_products_device.as_ptr();
    let zs_partial_products_lookups_device = unsafe { zs_partial_products_lookups_device.as_mut_ptr().add(zs_offset) };

    let kernel_function = MatrixTransFunction::default();

    let num_blocks = Dim3::from((block_x, block_y));
    let threads_per_block = Dim3::from(thread_x);
    let config = CudaLaunchConfig::basic(num_blocks, threads_per_block, &stream);
    let args = MatrixTransArguments {
        all_partial_products_device,
        zs_partial_products_lookups_device
    };
    kernel_function.launch(&config, &args)
}