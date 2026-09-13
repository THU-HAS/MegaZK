#include "context.cuh"
#include "goldilocks.cuh"
#include "memory.cuh"
#include "goldilocks_extension.cuh"
#include <iostream>
#define TILE 128

using namespace goldilocks;

DEVICE_FORCEINLINE void mul_ext(const base_field *in1, const base_field *in2, base_field *out) {
    base_field c{0x7, 0x0};
    base_field tmp0 = base_field::add(base_field::mul(in1[0], in2[0]), base_field::mul(c, base_field::mul(in1[1], in2[1])));
    base_field tmp1 = base_field::add(base_field::mul(in1[0], in2[1]), base_field::mul(in1[1], in2[0]));
    out[0] = tmp0;
    out[1] = tmp1;
}

extern "C" __launch_bounds__(128, 8) __global__ 
void dsa(const base_field *inputs, const base_field *point, const base_field *alpha,
    base_field *outputs, const unsigned degree_bits, const unsigned num_polys, const unsigned order) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id == 0) {
        base_field alpha_pow[2] = {1, 0};
        base_field alpha_sqr[2] = {alpha[0], alpha[1]};
        unsigned c = 1, t = num_polys;
        while (c <= num_polys) {
            if (t % 2 == 1)
                mul_ext(alpha_pow, alpha_sqr, alpha_pow);
            mul_ext(alpha_sqr, alpha_sqr, alpha_sqr);
            c *= 2;
            t /= 2;
            // printf("alpha_pow[0]: %llu, t: %d, c: %d\n", base_field::to_u64(alpha_pow[0]), t, c);
        }
        // mul_ext(alpha, alpha, alpha_pow);
        base_field acc[2] = {0, 0};
        base_field tmp[2] = {0, 0};
        const unsigned degree = 1 << degree_bits;
        // printf("alpha: (%llu, %llu)\n", base_field::to_u64(alpha[0]), base_field::to_u64(alpha[1]));
        // printf("alpha_pow: (%llu, %llu), count: %d\n", base_field::to_u64(alpha_pow[0]), base_field::to_u64(alpha_pow[1]), num_polys);
        printf("\nHello!\n");
        if (order == 0) {
            printf("point_used: (%llu, %llu)\n", point[0], point[1]);
            for (unsigned j = degree; j > 0; j--) {
                mul_ext(acc, point, acc);
                if (j > degree - 4) printf("acc * r: (%llu, %llu)\n", acc[0], acc[1]);
                acc[0] = base_field::add(acc[0], inputs[2 * j]);
                acc[1] = base_field::add(acc[1], inputs[2 * j + 1]);
                if (j > degree - 4) printf("in: (%llu, %llu)\n", inputs[2 * j], inputs[2 * j + 1]);
                if (j > degree - 4) printf("acc: (%llu, %llu)\n", acc[0], acc[1]);
                tmp[0] = outputs[2 * j - 2];
                tmp[1] = outputs[2 * j - 1];
                mul_ext(tmp, alpha_pow, tmp);
                outputs[2 * j - 2] = base_field::add(acc[0], tmp[0]);
                outputs[2 * j - 1] = base_field::add(acc[1], tmp[1]);            }
        } else {
            base_field g_zeta[2];
            mul_ext(point, point + 2, g_zeta);
            for (unsigned j = degree; j > 0; j--) {
                mul_ext(acc, g_zeta, acc);
                acc[0] = base_field::add(acc[0], inputs[2 * j]);
                acc[1] = base_field::add(acc[1], inputs[2 * j + 1]);
                tmp[0] = outputs[2 * j - 2];
                tmp[1] = outputs[2 * j - 1];
                mul_ext(tmp, alpha_pow, tmp);
                outputs[2 * j - 2] = base_field::add(acc[0], tmp[0]);
                outputs[2 * j - 1] = base_field::add(acc[1], tmp[1]);
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__ 
void fp_divide(const base_field *inputs, const base_field *point, 
    base_field *outputs, const unsigned degree_bits, const unsigned order) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id < TILE) {
        base_field point_used[2] = {0, 0};
        if (order == 0) {
            point_used[0] = point[0];
            point_used[1] = point[1];
        } else {
            mul_ext(point, point + 2, point_used);
        }
        // printf("point_used: (%llu, %llu)\n", point_used[0], point_used[1]);
        // First Step
        const unsigned degree = 1 << degree_bits;
        const unsigned block_len = degree / TILE;
        const base_field *inputs_start = inputs + g_id * block_len * 2;
        base_field *outputs_start = outputs + g_id * block_len * 2;
        base_field acc[2] = {0, 0};
        for (unsigned i = block_len; i > 0;) {
            --i;
            mul_ext(acc, point_used, acc);
            acc[0] = base_field::add(acc[0], inputs_start[2 * i]);
            acc[1] = base_field::add(acc[1], inputs_start[2 * i + 1]);
            outputs_start[2 * i] = acc[0];
            outputs_start[2 * i + 1] = acc[1];
        }
        __syncthreads();
        // Second Step
        base_field point_len[2] = { point_used[0], point_used[1] };
        for (unsigned i = 1; i < block_len; i *= 2) {
            mul_ext(point_len, point_len, point_len);
        }
        acc[0] = base_field::from_u64(0);
        acc[1] = base_field::from_u64(0);
        for (unsigned i = TILE - 1; i > g_id; i--) {
            mul_ext(acc, point_len, acc);
            acc[0] = base_field::add(acc[0], outputs[i * block_len * 2]);
            acc[1] = base_field::add(acc[1], outputs[i * block_len * 2 + 1]);
        }
        __syncthreads();
        // Third Step
        for (unsigned i = block_len; i > 0;) {
            --i;
            mul_ext(acc, point_used, acc);
            outputs_start[2 * i] = base_field::add(outputs_start[2 * i], acc[0]);
            outputs_start[2 * i + 1] = base_field::add(outputs_start[2 * i + 1], acc[1]);
        }
    }
}
