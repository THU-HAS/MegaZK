use boojum::field::goldilocks::GoldilocksField;
use cudart::cuda_kernel;
use cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use cudart::result::CudaResult;
use cudart::slice::DeviceSlice;
use cudart::stream::CudaStream;
use phf::phf_map;

cuda_kernel!(
    GeneratorNew,
    generator_new_kernel,
    witness: *mut GoldilocksField,
    rep_map: *const u32,
    params: *const usize,
    write_start: usize,
    read_start: usize,
    param_start: usize,
    length: usize
);

generator_new_kernel!(arithmetic_base_generator_new_kernel);
generator_new_kernel!(random_value_generator_new_kernel);
generator_new_kernel!(constant_generator_new_kernel);
generator_new_kernel!(poseidon_generator_new_kernel);
generator_new_kernel!(u32_add_many_generator_new_kernel);
generator_new_kernel!(u32_arithmetic_generator_new_kernel);
generator_new_kernel!(comparison_generator_new_kernel);
generator_new_kernel!(u32_interleave_generator_new_kernel);
generator_new_kernel!(u32_range_check_generator_new_kernel);
generator_new_kernel!(u32_subtraction_generator_new_kernel);
generator_new_kernel!(uninterleave_to_u32_generator_new_kernel);
generator_new_kernel!(base_sum_generator_new_kernel);
generator_new_kernel!(base_split_generator_new_kernel);
generator_new_kernel!(random_access_generator_new_kernel);
generator_new_kernel!(curve_point_decompression_generator_new_kernel);
generator_new_kernel!(non_native_addition_generator_new_kernel);
generator_new_kernel!(non_native_subtraction_generator_new_kernel);
generator_new_kernel!(non_native_multiplication_generator_new_kernel);
generator_new_kernel!(non_native_inverse_generator_new_kernel);
generator_new_kernel!(equality_generator_new_kernel);
generator_new_kernel!(biguint_div_rem_generator_new_kernel);
generator_new_kernel!(glv_decomposition_generator_new_kernel);

pub static NEW_FUNC_MAP: phf::Map<
    &'static str,
    unsafe extern "C" fn(
        *mut GoldilocksField,
        *const u32,
        *const usize,
        usize,
        usize,
        usize,
        usize,
    ),
> = phf_map! {
    "ArithmeticBaseGenerator" => arithmetic_base_generator_new_kernel,
    "RandomValueGenerator" => random_value_generator_new_kernel,
    "ConstantGenerator" => constant_generator_new_kernel,
    "PoseidonGenerator" => poseidon_generator_new_kernel,
    "U32AddManyGenerator" => u32_add_many_generator_new_kernel,
    "U32ArithmeticGenerator" => u32_arithmetic_generator_new_kernel,
    "ComparisonGenerator" => comparison_generator_new_kernel,
    "U32InterleaveGenerator" => u32_interleave_generator_new_kernel,
    "U32RangeCheckGenerator" => u32_range_check_generator_new_kernel,
    "U32SubtractionGenerator" => u32_subtraction_generator_new_kernel,
    "UninterleaveToU32Generator" => uninterleave_to_u32_generator_new_kernel,
    "BaseSumGenerator" => base_sum_generator_new_kernel,
    "BaseSplitGenerator" => base_split_generator_new_kernel,
    "RandomAccessGenerator" => random_access_generator_new_kernel,
    "CurvePointDecompressionGenerator" => curve_point_decompression_generator_new_kernel,
    "NonNativeAdditionGenerator" => non_native_addition_generator_new_kernel,
    "NonNativeSubtractionGenerator" => non_native_subtraction_generator_new_kernel,
    "NonNativeMultiplicationGenerator" =>non_native_multiplication_generator_new_kernel,
    "NonNativeInverseGenerator" => non_native_inverse_generator_new_kernel,
    "EqualityGenerator" => equality_generator_new_kernel,
    "BigUintDivRemGenerator" => biguint_div_rem_generator_new_kernel,
    "GLVDecompositionGenerator" => glv_decomposition_generator_new_kernel,
    // "WireSplitGenerator" => wire_split_generator_new_kernel,
    // "MulExtensionGenerator" => mul_extension_generator_new_kernel,
    // "ArithmeticExtensionGenerator" => arithmetic_extension_generator_new_kernel,
    // "ReducingExtensionGenerator" => reducing_extension_generator_new_kernel,
    // "ReducingGenerator" => reducing_generator_new_kernel,
    // "InterpolationGenerator" => interpolation_generator_new_kernel,
    // "PoseidonMdsGenerator" => poseidon_mds_generator_new_kernel,
};

cuda_kernel!(GFW, generate_full_witness_kernel(
    full_witness: *mut GoldilocksField,
    witness: *const GoldilocksField,
    representative_map: *const u32,
));

pub fn generate_full_witness(
    full_witness_device: &mut DeviceSlice<GoldilocksField>,
    offset: usize,
    representative_map_device: &DeviceSlice<u32>,
    block_x: u32,
    thread_x: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let full_witness = unsafe { full_witness_device.as_mut_ptr().add(offset) };

    let witness = full_witness;
    let representative_map = representative_map_device.as_ptr();
    let kernel_function = GFWFunction::default();

    let num_blocks = Dim3::from(block_x);
    let threads_per_block = Dim3::from(thread_x);
    let config = CudaLaunchConfig::basic(num_blocks, threads_per_block, &stream);
    let args = GFWArguments {
        full_witness,
        witness,
        representative_map,
    };
    kernel_function.launch(&config, &args)
}
