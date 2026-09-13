#include "context.cuh"
#include "goldilocks.cuh"
#include "memory.cuh"
#include <iostream>
#define BLOCK_SIZE 80
#define SCAN_BLOCK_SIZE 128
using namespace goldilocks;

extern "C" __global__ void wires_permutation_partial_products_kernel(
    base_field *witness_device,
    base_field *subgroup_device,
    base_field *k_is_device,
    base_field *s_sigmas_device,
    base_field *quotient_values_device,
    const unsigned degree,
    const unsigned degree2,
    base_field *betas,
    base_field *gammas
    ) {

    int tid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    int i, j;
    base_field wire_value, k_i, s_id, s_sigma, numerator, denominator, quotient;

    const base_field x = subgroup_device[blockIdx.x];
    quotient_values_device[tid] = base_field::one();

    for(i = 0; i < degree; i++){
        j = threadIdx.x * degree + i;
        wire_value = witness_device[j * degree2 + blockIdx.x];
        k_i = k_is_device[j];
        s_id = base_field::mul(k_i, x);
        s_sigma = s_sigmas_device[blockIdx.x * blockDim.x * degree + j];
        numerator = base_field::add(base_field::add(wire_value, base_field::mul(betas[blockIdx.y], s_id)), gammas[blockIdx.y]);
        denominator = base_field::add(base_field::add(wire_value, base_field::mul(betas[blockIdx.y], s_sigma)), gammas[blockIdx.y]);
        quotient = base_field::mul(numerator, base_field::inv(denominator));
        quotient_values_device[tid] = base_field::mul(quotient, quotient_values_device[tid]);
    }
}

extern "C" __global__ void wires_permutation_partial_products_kernel_trans(
    base_field *witness_device,
    base_field *subgroup_device,
    base_field *k_is_device,
    base_field *s_sigmas_device,
    base_field *quotient_values_device,
    const unsigned degree,
    const unsigned degree2,
    base_field *betas,
    base_field *gammas
    ) {

    int tid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    int i, j;
    base_field wire_value, k_i, s_id, s_sigma, numerator, denominator, quotient;

    const base_field x = subgroup_device[blockIdx.x];
    quotient_values_device[tid] = base_field::one();

    for(i = 0; i < degree; i++){
        j = threadIdx.x * degree + i;
        wire_value = witness_device[j * degree2 + blockIdx.x];
        k_i = k_is_device[j];
        s_id = base_field::mul(k_i, x);
        s_sigma = s_sigmas_device[blockIdx.x + j * degree2];
        numerator = base_field::add(base_field::add(wire_value, base_field::mul(betas[blockIdx.y], s_id)), gammas[blockIdx.y]);
        denominator = base_field::add(base_field::add(wire_value, base_field::mul(betas[blockIdx.y], s_sigma)), gammas[blockIdx.y]);
        quotient = base_field::mul(numerator, base_field::inv(denominator));
        quotient_values_device[tid] = base_field::mul(quotient, quotient_values_device[tid]);
    }
}

// extern "C" __global__ void scan_part_kernel(
//     const base_field *quotient_values_device,
//     base_field *part_device,
//     base_field *all_partial_products_device,
//     const unsigned length
// ) {
//     int part_i = blockIdx.y * gridDim.x + blockIdx.x;
//     int part_begin = part_i * blockDim.x;
//     int part_end = min((part_i + 1) * blockDim.x, length);
//     if (threadIdx.x == 0) {
//       base_field prod = base_field::one();
//       for (int i = part_begin; i < part_end; i++) {
//         prod = base_field::mul(prod, quotient_values_device[i]);
//         all_partial_products_device[i] = prod;
//       }
//       part_device[part_i] = prod;
//     }
// }

__device__ void ScanBlock(base_field *shm) {
    if (threadIdx.x == 0) {
        base_field prod = base_field::one();
        for (int i = 0; i < blockDim.x; i++) {
            prod = base_field::mul(prod, shm[i]);;
            shm[i] = prod;
        }
    }
}

extern "C" __global__ void scan_part_kernel(
    const base_field *quotient_values_device,
    base_field *part_device,
    base_field *all_partial_products_device,
    const unsigned length
) {
    __shared__ base_field shm[SCAN_BLOCK_SIZE];
    int part_i = blockIdx.y * gridDim.x + blockIdx.x;
    int tid = part_i * blockDim.x + threadIdx.x;
    shm[threadIdx.x] = tid < length ? quotient_values_device[tid] : base_field::zero();
    __syncthreads();
    ScanBlock(shm);
    __syncthreads();
    if (tid < length) {
        all_partial_products_device[tid] = shm[threadIdx.x];
    }
    if (threadIdx.x == blockDim.x - 1) {
        part_device[part_i] = shm[threadIdx.x];
    }
}

extern "C" __global__ void scan_part_product_kernel(
    base_field *part_device,
    const unsigned part_num
) {
    base_field prod = base_field::one();
    for (int i = 0; i < part_num; i++) {
        prod = base_field::mul(prod, part_device[i]);
        part_device[i] = prod;
    }
    prod = base_field::from_u64(1);
    for (int i = part_num; i < 2 * part_num; i++) {
        prod = base_field::mul(prod, part_device[i]);
        part_device[i] = prod;
    }
}

extern "C" __global__ void mul_part_product_kernel(
    base_field *part_device,
    base_field *all_partial_products_device,
    const unsigned length
) {
    int part_i = blockIdx.y * gridDim.x + blockIdx.x;
    if (blockIdx.x != 0) {
        int tid = part_i * blockDim.x + threadIdx.x;
        if (tid < length) {
        all_partial_products_device[tid] = base_field::mul(all_partial_products_device[tid], part_device[part_i - 1]);
        }
    }
}

extern "C" __global__ void matrix_trans_kernel(
    base_field *all_partial_products_device,
    base_field *zs_partial_products_lookups_device
) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = blockIdx.y;

    if(j == blockDim.x - 1){
        if(i == gridDim.x -1){
            zs_partial_products_lookups_device[k * gridDim.x] = base_field::one();
        }
        else{
            zs_partial_products_lookups_device[k * gridDim.x + i + 1] = all_partial_products_device[k * gridDim.x * blockDim.x + i * blockDim.x + j];
        }
    }
    else{
        zs_partial_products_lookups_device[k * gridDim.x * blockDim.x + (j + 2 - k) * gridDim.x + i] = all_partial_products_device[k * gridDim.x * blockDim.x + i * blockDim.x + j];
    }
}