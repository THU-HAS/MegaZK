#include "goldilocks.cuh"
using namespace goldilocks;

#define POINT_PER_THREAD 32

DEVICE_FORCEINLINE unsigned reverse_bits(unsigned in, unsigned log_n) {
  unsigned out = 0;
  for (unsigned i = 0; i < log_n; i++) {
    if (in & (1 << i)) {
      out |= 1 << (log_n - 1 - i);
    }
  }
  return out;
}

extern "C" __launch_bounds__(128, 8) __global__
    void reverse_index_bits_inplace(base_field *gmem_inputs_matrix, const unsigned log_n, const unsigned num_ntts) {
  static const unsigned array[] = {0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31};
  const unsigned tile = 32, log_tile = 5;
  const unsigned degree = 1 << log_n;
  const unsigned blocks_per_ntt = (degree + tile * tile - 1) / (tile * tile);
  // const unsigned ntt_id = (blockIdx.x + blocks_per_ntt - 1) / blocks_per_ntt;
  const unsigned ntt_id = blockIdx.x / blocks_per_ntt;
  const unsigned warp_id = blockIdx.x % blocks_per_ntt;
  const unsigned warp_id_r = reverse_bits(warp_id, log_n - 2 * log_tile);
  // if (threadIdx.x == 0) printf("warp: %d %d, ntt id: %d\n", warp_id, warp_id_r, ntt_id);
  const unsigned lane_id{threadIdx.x & 31};
  const unsigned lane_id_r = array[lane_id];
  base_field *gmem_inputs_start = gmem_inputs_matrix + ntt_id * degree + warp_id * tile;
  base_field *gmem_inputs_start_r = gmem_inputs_matrix + ntt_id * degree + warp_id_r * tile;
  const unsigned stride_within = degree / tile;
  __shared__ base_field smem[2 * tile][tile + 1];
  if (warp_id < warp_id_r) {
#pragma unroll
    for (unsigned i = 0; i < tile; i++) {
      smem[i][lane_id] = memory::load_cg(gmem_inputs_start + lane_id + i * stride_within);
      smem[i + tile][lane_id] = memory::load_cg(gmem_inputs_start_r + lane_id + i * stride_within);
    }
    __syncwarp();
#pragma unroll
    for (unsigned i = 0; i < tile; i++) {
      memory::store_cg(gmem_inputs_start_r + lane_id + i * stride_within, smem[lane_id_r][array[i]]);
      memory::store_cg(gmem_inputs_start + lane_id + i * stride_within, smem[lane_id_r + tile][array[i]]);
      // if (threadIdx.x == 0) printf("warp: %d %d\n", ntt_id * stride_between_input_arrays + warp_id_r * TILLE, lane_id + i * stride_within);
    }
  } else if (warp_id == warp_id_r) {
#pragma unroll
    for (unsigned i = 0; i < tile; i++) {
      smem[i][lane_id] = memory::load_cg(gmem_inputs_start + lane_id + i * stride_within);
    }
    __syncwarp();
#pragma unroll
    for (unsigned i = 0; i < tile; i++) {
      memory::store_cg(gmem_inputs_start_r + lane_id + i * stride_within, smem[lane_id_r][array[i]]);
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void mul_exp(base_field *gmem_inputs_matrix, const unsigned log_n, const unsigned num_ntts,
                   const unsigned stride_between_input_arrays, const base_field mul_val) {
  const unsigned blocks_per_ntt = (stride_between_input_arrays + blockDim.x * POINT_PER_THREAD - 1) / (blockDim.x * POINT_PER_THREAD);
  const unsigned ntt_id = blockIdx.x / blocks_per_ntt;
  const unsigned stride_within = stride_between_input_arrays / POINT_PER_THREAD;
  const unsigned g_id = blockIdx.x % blocks_per_ntt * blockDim.x + threadIdx.x;
  // printf("gid: %d, stride within: %d\n", g_id, stride_within);
  // if (threadIdx.x == 0) printf("gid: %d, ntt id: %d\n", g_id, ntt_id);
  if (g_id < stride_within) {
    base_field *gmem_inputs_start = gmem_inputs_matrix + ntt_id * stride_between_input_arrays + g_id;
    // base_field *gmem_outputs_start = gmem_outputs_matrix + ntt_id * stride_between_output_arrays + g_id;
    // if (threadIdx.x == 0) printf("warp: %d %d, ntt id: %d\n", warp_id, warp_id_r, ntt_id);
    base_field a{0x1, 0x0};
    unsigned c = 1, ggid = g_id;
    base_field b_used = mul_val;
    while (c != stride_within) {
      if (ggid % 2 == 1)
        a = base_field::mul(a, b_used);
      b_used = base_field::sqr(b_used);
      c *= 2;
      ggid /= 2;
      // printf("gid: %d, a: %lld, b: %lld, c: %d\n", g_id, base_field::to_u64(a), base_field::to_u64(b), c);
    }
    // printf("gid: %d, a: %lld\n", g_id, base_field::to_u64(a));

#pragma unroll
    for (unsigned i = 0; i < POINT_PER_THREAD; i++) {
      base_field coseted = base_field::mul(memory::load_cs(gmem_inputs_start + i * stride_within), a);
      memory::store_cs(gmem_inputs_start + i * stride_within, coseted);
      a = base_field::mul(a, b_used);
    }
  }
}