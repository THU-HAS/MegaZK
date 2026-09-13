#include "context.cuh"
#include "goldilocks.cuh"
#include "memory.cuh"
#include "goldilocks_extension.cuh"
#include <iostream>

#define TILE_O 64

using namespace goldilocks;

DEVICE_FORCEINLINE void mul_ext(const base_field *in1, const base_field *in2, base_field *out) {
  base_field c{0x7, 0x0};
  base_field tmp0 = base_field::add(base_field::mul(in1[0], in2[0]), base_field::mul(c, base_field::mul(in1[1], in2[1])));
  base_field tmp1 = base_field::add(base_field::mul(in1[0], in2[1]), base_field::mul(in1[1], in2[0]));
  out[0] = tmp0;
  out[1] = tmp1;
}

extern "C" __launch_bounds__(128, 8) __global__
    void open_polys(const base_field *polys, const base_field *zeta_g, base_field *openings, 
                  const unsigned degree_bits, const unsigned num_polys, const unsigned num_next, const unsigned next_start
) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (g_id < num_polys + num_next) {
    const unsigned degree = 1 << degree_bits;
    base_field acc[2] = {0, 0};
    if (g_id < num_polys) {
        const base_field *polys_start = polys + g_id * degree;
        // base_field coeff[2] = {0, 0};
        for (unsigned j = degree; j > 0;) {
            // coeff[0] = polys_start[j];
            mul_ext(acc, zeta_g, acc);
            acc[0] = base_field::add(acc[0], polys_start[--j]);
        }
    } else {
        base_field g_zeta[2];
        mul_ext(zeta_g, zeta_g + 2, g_zeta);
        const base_field *polys_start = polys + (g_id + next_start - num_polys) * degree;
        for (unsigned j = degree; j > 0;) {
            mul_ext(acc, g_zeta, acc);
            acc[0] = base_field::add(acc[0], polys_start[--j]);
        }
    }
    openings[2 * g_id] = acc[0];
    openings[2 * g_id + 1] = acc[1];
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void open_polys_new(const base_field *polys, const base_field *zeta_g, base_field *buffer, 
                  const unsigned degree_bits, const unsigned num_polys, const unsigned num_next, const unsigned next_start
) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id < (num_polys + num_next) * TILE_O) {
    const unsigned degree = 1 << degree_bits;
    const unsigned seg_len = degree / TILE_O;
    const unsigned p_id = g_id / TILE_O;
    const unsigned seg_id = g_id % TILE_O;
    base_field acc[2] = {0, 0};
    if (p_id < num_polys) {
        const base_field *polys_start = polys + p_id * degree + seg_id * seg_len;
        // base_field coeff[2] = {0, 0};
        for (unsigned j = seg_len; j > 0;) {
            // coeff[0] = polys_start[j];
            mul_ext(acc, zeta_g, acc);
            acc[0] = base_field::add(acc[0], polys_start[--j]);
        }
    } else {
        base_field g_zeta[2];
        mul_ext(zeta_g, zeta_g + 2, g_zeta);
        const base_field *polys_start = polys + (p_id + next_start - num_polys) * degree + seg_id * seg_len;
        for (unsigned j = seg_len; j > 0;) {
            mul_ext(acc, g_zeta, acc);
            acc[0] = base_field::add(acc[0], polys_start[--j]);
        }
    }
    buffer[2 * g_id] = acc[0];
    buffer[2 * g_id + 1] = acc[1];
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void open_polys_reduce(const base_field *buffer, const base_field *zeta_g, base_field *openings, 
                  const unsigned degree_bits, const unsigned num_polys, const unsigned num_next, const unsigned next_start
) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (g_id < num_polys + num_next) {
    const unsigned degree = 1 << degree_bits;
    const unsigned seg_len = degree / TILE_O;
    base_field acc[2] = {0, 0};
    base_field zeta_pow[2] = {0, 0};
    if (g_id < num_polys) {
      zeta_pow[0] = zeta_g[0];
      zeta_pow[1] = zeta_g[1];
    } else {
      mul_ext(zeta_g, zeta_g + 2, zeta_pow);
    }
    unsigned cnt = 1;
    while (cnt < seg_len)
    {
      mul_ext(zeta_pow ,zeta_pow, zeta_pow);
      cnt *= 2;
    }
    const base_field *buffer_start = buffer + 2 * g_id * TILE_O;
    for (unsigned j = TILE_O; j > 0;) {
      mul_ext(acc, zeta_pow, acc);
      j--;
      acc[0] = base_field::add(acc[0], buffer_start[j * 2]);
      acc[1] = base_field::add(acc[1], buffer_start[j * 2 + 1]);
    }
    openings[2 * g_id] = acc[0];
    openings[2 * g_id + 1] = acc[1];
  }
}

extern "C" __launch_bounds__(128, 8) __global__ 
void reduce(const base_field *gmem_inputs_matrix, const base_field *alpha_matrix,
    base_field *gmem_outputs_matrix, const unsigned log_n, const unsigned num_ntts, const unsigned offset) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  // base_field f0{0x1,0x0}, f1{0x1,0x0};
  base_field acc0{0x0, 0x0}, acc1{0x0, 0x0};
  // base_field current[2] = {acc0, acc1};
  base_field alpha_pow[2] = {1, 0};
  // if (g_id == 0) { printf("num_alpha: %lld\n", base_field::to_u64(alpha_matrix[0])); }
  const base_field *gmem_inputs_start = gmem_inputs_matrix + (offset << log_n) + g_id;
  for (unsigned i = 0; i < num_ntts; i++) {
    base_field tmp = memory::load_cs(gmem_inputs_start + (i << log_n));
    acc0 = base_field::add(acc0, base_field::mul(tmp, alpha_pow[0]));
    acc1 = base_field::add(acc1, base_field::mul(tmp, alpha_pow[1]));
    // f0 = base_field::mul(alpha_matrix[0], f0);
    // f1 = base_field::mul(alpha_matrix[1], f1);
    mul_ext(alpha_pow, alpha_matrix, alpha_pow);
    // if (g_id == 0) { printf("acc0: %lld\n", base_field::to_u64(acc0)); }
    // if (g_id == 0) { printf("acc1: %lld\n", base_field::to_u64(acc1)); }
    // if (g_id == 0) { printf("alpha_pow: %lld %lld\n", base_field::to_u64(alpha_pow[0]), base_field::to_u64(alpha_pow[1])); }
  }
  memory::store_cs(gmem_outputs_matrix + 2 * g_id, acc0);
  memory::store_cs(gmem_outputs_matrix + 2 * g_id + 1, acc1);
}

extern "C" __launch_bounds__(128, 8) __global__ 
void fp_shift_add(const base_field *inputs, const base_field *alpha, base_field *outputs, 
    const unsigned degree_bits, const unsigned num_polys) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned row_step = nrows / 128;
    if (g_id < row_step) {
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
        for (unsigned i = 0; i < 128; i++) {
          base_field *out = outputs + 2 * (i * row_step + g_id);
          const base_field *in = inputs + 2 * (i * row_step + g_id);
          mul_ext(out, alpha_pow, out);
          out[0] = base_field::add(out[0], in[0]);
          out[1] = base_field::add(out[1], in[1]);
        }
    }
}

__device__ __forceinline__ void exchg_dit_e(extension_field &a, extension_field &b, const extension_field &twiddle) {
  b = extension_field::mul(b, twiddle);
  const auto a_tmp = a;
  a = extension_field::add(a_tmp, b);
  b = extension_field::sub(a_tmp, b);
}

DEVICE_FORCEINLINE unsigned reverse_bits(unsigned in, unsigned log_n) {
  unsigned out = 0;
  for (unsigned i = 0; i < log_n; i++) {
    if (in & (1 << i)) {
      out |= 1 << (log_n - 1 - i);
    }
  }
  return out;
}

// Simple, non-optimized kernel used for log_n < 16, to unblock debugging small proofs.
extern "C" __launch_bounds__(512, 2) __global__
    void n2b_1_stage_e(const base_field *gmem_inputs_matrix, const base_field *root_pow, base_field *gmem_outputs_matrix, 
        const unsigned log_n, const unsigned blocks_per_ntt, const unsigned start_stage) {
  const unsigned ntt_idx = blockIdx.x / blocks_per_ntt; // n-th ntt
  const unsigned bid_in_ntt = blockIdx.x - ntt_idx * blocks_per_ntt; // block in this ntt
  const unsigned tid_in_ntt = threadIdx.x + bid_in_ntt * blockDim.x; // thread in this ntt
  const unsigned stride = 1 << log_n;
  if (tid_in_ntt >= (1 << (log_n - 1)))
    return;
  const unsigned log_exchg_region_sz = log_n - start_stage;
  const unsigned exchg_region = tid_in_ntt >> (log_exchg_region_sz - 1);
  const unsigned tid_in_exchg_region = tid_in_ntt - (exchg_region << (log_exchg_region_sz - 1));
  const unsigned exchg_stride = 1 << (log_exchg_region_sz - 1);
  const unsigned a_idx = tid_in_exchg_region + exchg_region * (1 << log_exchg_region_sz);
  const unsigned b_idx = a_idx + exchg_stride;
  const extension_field *gmem_input = ((extension_field *)gmem_inputs_matrix) + ntt_idx * stride;
  extension_field *gmem_output = ((extension_field *)gmem_outputs_matrix) + ntt_idx * stride;
  unsigned pow = exchg_region * exchg_stride;
  pow = reverse_bits(pow, log_n - 1) << (log_n - 1 - start_stage);
  // printf("tid: %d, aid: %d, bid: %d, region: %d, stride: %d, pow: %d\n", tid_in_ntt, a_idx, b_idx, exchg_region, exchg_stride, pow);

  // const auto twiddle = get_twiddle_e(false, exchg_region);
  base_field twiddle_array[2] = {1, 0};
  base_field root_unity[2] = {0, 0};
  root_unity[1] = base_field::from_u64(15659105665374529263ULL);
  for (unsigned i = 0; i < 33 - log_n; i++) {
    mul_ext(root_unity, root_unity, root_unity);
  }
  unsigned c = 1, t = pow;
  while (c <= pow) {
      if (t % 2 == 1)
          mul_ext(twiddle_array, root_unity, twiddle_array);
      mul_ext(root_unity, root_unity, root_unity);
      c *= 2;
      t /= 2;
      // printf("root_unity[0]: %llu, t: %d, c: %d\n", base_field::to_u64(root_unity[0]), t, c);
  }
  // if (tid_in_ntt == 0) printf("root: (%llu, %llu)\n", root_unity[0], root_unity[1]);
  extension_field twiddle = {twiddle_array[0], twiddle_array[1]};
  auto a = memory::load_cg(gmem_input + a_idx);
  auto b = memory::load_cg(gmem_input + b_idx);
  // printf("tid: %d, a: (%llu, %llu), b: (%llu, %llu), twiddle:(%llu, %llu)\n", tid_in_ntt, a[0], a[1], b[0], b[1], twiddle[0], twiddle[1]);
  exchg_dit_e(a, b, twiddle);
  // printf("tid: %d, a: (%llu, %llu), b: (%llu, %llu), twiddle:(%llu, %llu)\n", tid_in_ntt, a[0], a[1], b[0], b[1], twiddle[0], twiddle[1]);
  memory::store_cg(gmem_output + a_idx, a);
  memory::store_cg(gmem_output + b_idx, b);
}

#define TILE 32

extern "C" __launch_bounds__(128, 8) __global__
    void coset_e(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const base_field coset, const unsigned log_n, const unsigned num_ntts,
                   const unsigned stride_between_input_arrays, const unsigned stride_between_output_arrays) {
  const unsigned blocks_per_ntt = (stride_between_input_arrays + blockDim.x * TILE - 1) / (blockDim.x * TILE);
  const unsigned ntt_id = blockIdx.x / blocks_per_ntt;
  const unsigned stride_within = stride_between_input_arrays / TILE;
  const unsigned g_id = blockIdx.x % blocks_per_ntt * blockDim.x + threadIdx.x;
  // printf("gid: %d, stride within: %d\n", g_id, stride_within);
  // if (threadIdx.x == 0) printf("gid: %d, ntt id: %d\n", g_id, ntt_id);
  if (g_id < stride_within) {
    // const base_field *gmem_inputs_start = gmem_inputs_matrix + ntt_id * stride_between_input_arrays + g_id;
    // base_field *gmem_outputs_start = gmem_outputs_matrix + ntt_id * stride_between_output_arrays + g_id;
    const extension_field *gmem_input = (extension_field *)gmem_inputs_matrix + ntt_id * stride_between_input_arrays + g_id;
    extension_field *gmem_output = (extension_field *)gmem_outputs_matrix + ntt_id * stride_between_output_arrays + g_id;
    // if (threadIdx.x == 0) printf("warp: %d %d, ntt id: %d\n", warp_id, warp_id_r, ntt_id);
    base_field a{0x1, 0x0};//, b{0x7, 0x0}, b_inv{0x6db6db6e, 0x24924924};
    unsigned c = 1, ggid = g_id;
    // base_field b_used = inverse ? b_inv : b;
    base_field b_used = coset;
    while (c != stride_within) {
      if (ggid % 2 == 1)
        a = base_field::mul(a, b_used);
      b_used = base_field::sqr(b_used);
      c *= 2;
      ggid /= 2;
      // printf("gid: %d, a: %lld, b: %lld, c: %d\n", g_id, base_field::to_u64(a), base_field::to_u64(b), c);
    }
    // printf("gid: %d, a: %lld\n", g_id, base_field::to_u64(a));
    // extension_field a_e = {a, 0};

#pragma unroll
    for (unsigned i = 0; i < TILE; i++) {
      extension_field coseted = extension_field::mul(memory::load_cs(gmem_input + i * stride_within), a);
      memory::store_cs(gmem_output + i * stride_within, coseted);
      a = base_field::mul(a, b_used);
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__ 
  void reduce_e(const base_field *gmem_inputs_matrix, const base_field *beta_matrix,
    base_field *gmem_outputs_matrix, const unsigned num_chunks, const unsigned chunk_size) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (g_id < num_chunks) {
    extension_field acc = {0, 0};
    extension_field beta = {beta_matrix[0], beta_matrix[1]};
    // extension_field beta_pow = {1, 0};
    extension_field *inputs_start = (extension_field *)gmem_inputs_matrix + g_id * chunk_size;
    extension_field *outputs_start = (extension_field *)gmem_outputs_matrix + g_id;
    // if (g_id == 0) { printf("num_alpha: %lld\n", base_field::to_u64(alpha_matrix[0])); }
    for (unsigned i = chunk_size; i > 0;) {
      extension_field tmp = memory::load_cs(inputs_start + (--i));
      acc = extension_field::add(tmp, extension_field::mul(acc, beta));
      // beta_pow = extension_field::mul(beta, beta_pow);
      // if (g_id == 0) { printf("i: %d, beta_pow: (%llu, %llu), tmp: (%llu, %llu), acc: (%llu, %llu)\n", 
      //   i, beta[0], beta[1], tmp[0], tmp[1], acc[0], acc[1]); }
    }
    memory::store_cs(outputs_start, acc);
  }
}

extern "C" __launch_bounds__(128, 8) __global__ 
  void fri_initial(const base_field *leaves, const base_field *digests,
    base_field *leave, base_field *siblings, base_field *x_indexes,
    const unsigned num_querys, const unsigned num_polys, const unsigned log_n,
    const unsigned cap_height, const unsigned num_layers) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (g_id < num_querys) {
    uint64_t degree = 1 << log_n;
    uint64_t x_index = base_field::to_u64(x_indexes[g_id]) % degree;
    x_indexes[g_id] = base_field::from_u64(x_index);
    // printf("gid: %d, x_index: %llu\n", g_id, x_index);
    // copy leave
    base_field *leave_start = leave + g_id * num_polys;
    const base_field *leaves_start = leaves + x_index;
    // const base_field *leaves_start = leaves + reverse_bits(x_index, log_n);
    for (unsigned i = 0; i < num_polys; i++) {
      leave_start[i] = leaves_start[i << log_n];
    }
    // generate merkle path
    uint64_t tree_index = x_index >> num_layers;
    unsigned full_layers = num_layers + cap_height + 1;
    base_field *siblings_start = siblings + g_id * (4 * 4 * num_layers);
    uint64_t pair_index = x_index & ((1 << num_layers) - 1);
    for (unsigned i = 0; i < num_layers; i++) {
      uint64_t parity = pair_index & 1;
      pair_index = pair_index >> 1;
      uint64_t siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
      uint64_t sibling_index = 2 * siblings_index + (1 - parity);
      // let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
      // let sibling_index = 2 * siblings_index + (1 - parity);
      uint64_t id = sibling_index;
      unsigned layer = 0;
      while (true) {
        id = id >> 1;
        if (id & 1 == 1) {
          layer += 1;
        } else {
          break;
        }
      }
      uint64_t id_in_layer = (sibling_index + 2 - (1 << (layer + 1))) / (1 << (layer + 2)) * 2 + sibling_index % 2;
      uint64_t index = id_in_layer + (1 << full_layers) - (1 << (full_layers - layer));
      index += (tree_index << (num_layers - layer));
      for (unsigned j = 0; j < 4; j++) {
        siblings_start[4 * i + j] = digests[4 * index + j];
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__ 
  void fri_initial_leaves_seg(const base_field *inputs, base_field *buffer, base_field *x_indexes,
    const unsigned num_querys, const unsigned num_polys, const unsigned log_n, const unsigned rate_bits) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned query_id = blockIdx.x % num_querys;
  const unsigned p_id = blockIdx.x / num_querys;
  const unsigned seg_len = (1 << log_n) / TILE_O;
  // const unsigned p_id = threadIdx.x / TILE_O;
  const unsigned seg_id = threadIdx.x;
  if (blockIdx.x < num_querys * num_polys && threadIdx.x < TILE_O) {
    uint64_t degree = 1 << log_n;
    uint64_t lde_degree = 1 << (log_n + rate_bits);
    uint64_t x_index = base_field::to_u64(x_indexes[query_id]) % lde_degree;
    if (p_id == 0 && seg_id == 0) {
      x_indexes[query_id] = base_field::from_u64(x_index);
    }
    // get omega and omega^j
    base_field root_unity = base_field::from_u64(0x185629dcda58878c); // What value?
    for (unsigned i = 0; i < 32 - (log_n + rate_bits); i++) {
      root_unity = root_unity * root_unity;
    }
    x_index = reverse_bits(x_index, log_n + rate_bits);
    base_field omega_j {0x1, 0x0};
    for (unsigned i = 0; i < log_n + rate_bits; i++) {
      if (x_index % 2 == 1) {
        omega_j = omega_j * root_unity;
      }
      root_unity = root_unity * root_unity;
      x_index = x_index >> 1;
    }
    // use coset * omega^j to reduce segs into buffer
    omega_j = omega_j * base_field::from_u64(0x7); // coset
    const base_field *polys_start = inputs + p_id * degree + seg_id * seg_len;
    base_field acc {0x0, 0x0};
    for (unsigned j = seg_len; j > 0;) {
        acc = acc * omega_j;
        acc = acc + polys_start[--j];
    }
    buffer[g_id] = acc;
  }
}

extern "C" __launch_bounds__(128, 8) __global__ 
  void fri_initial_leaves_reduce(const base_field *buffer, base_field *leave, const base_field *x_indexes,
    const unsigned num_querys, const unsigned num_polys, const unsigned log_n, const unsigned rate_bits) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned query_id = g_id % num_querys;
  const unsigned p_id = g_id / num_querys;
  const unsigned seg_len = (1 << log_n) / TILE_O;
  if (g_id < num_querys * num_polys) {
    uint64_t degree = 1 << log_n;
    uint64_t lde_degree = 1 << (log_n + rate_bits);
    uint64_t x_index = base_field::to_u64(x_indexes[query_id]) % lde_degree;
    // x_indexes[query_id] = base_field::from_u64(x_index);
    // get omega and omega^(j * seg_len)
    base_field root_unity = base_field::from_u64(0x185629dcda58878c); // What value?
    for (unsigned i = 0; i < 32 - (log_n + rate_bits); i++) {
      root_unity = root_unity * root_unity;
    }
    x_index = reverse_bits(x_index, log_n + rate_bits);
    base_field omega_j {0x1, 0x0};
    for (unsigned i = 0; i < log_n + rate_bits; i++) {
      if (x_index % 2 == 1) {
        omega_j = omega_j * root_unity;
      }
      root_unity = root_unity * root_unity;
      x_index = x_index >> 1;
    }
    omega_j = omega_j * base_field::from_u64(0x7); // coset
    unsigned cnt = 1;
    while (cnt < seg_len) {
      omega_j = omega_j * omega_j;
      cnt *= 2;
    }
    // use omega^(j * seg_len) to reduce segs into leave
    const base_field *buffer_start = buffer + g_id * TILE_O;
    base_field acc {0x0, 0x0};
    for (unsigned j = TILE_O; j > 0;) {
        acc = acc * omega_j;
        acc = acc + buffer_start[--j];
    }
    leave[query_id * num_polys + p_id] = acc;
  }
}

extern "C" __launch_bounds__(128, 8) __global__ 
  void fri_initial_leaves_reduce_partial(const base_field *ldes, const base_field *buffer, base_field *leave, base_field *x_indexes,
    const unsigned num_querys, const unsigned num_polys, const unsigned num_ldes, const unsigned log_n, const unsigned rate_bits) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned query_id = g_id % num_querys;
  const unsigned p_id = g_id / num_querys;
  const unsigned seg_len = (1 << log_n) / TILE_O;
  if (g_id < num_querys * num_polys) {
    uint64_t degree = 1 << log_n;
    uint64_t lde_degree = 1 << (log_n + rate_bits);
    uint64_t x_index = base_field::to_u64(x_indexes[query_id]) % lde_degree;
    x_indexes[query_id] = base_field::from_u64(x_index);
    if (p_id < num_ldes) {
      leave[query_id * num_polys + p_id] = ldes[(p_id * lde_degree) + x_index];
      return;
    }
    // get omega and omega^(j * seg_len)
    base_field root_unity = base_field::from_u64(0x185629dcda58878c); // What value?
    for (unsigned i = 0; i < 32 - (log_n + rate_bits); i++) {
      root_unity = root_unity * root_unity;
    }
    x_index = reverse_bits(x_index, log_n + rate_bits);
    base_field omega_j {0x1, 0x0};
    for (unsigned i = 0; i < log_n + rate_bits; i++) {
      if (x_index % 2 == 1) {
        omega_j = omega_j * root_unity;
      }
      root_unity = root_unity * root_unity;
      x_index = x_index >> 1;
    }
    omega_j = omega_j * base_field::from_u64(0x7); // coset
    unsigned cnt = 1;
    while (cnt < seg_len) {
      omega_j = omega_j * omega_j;
      cnt *= 2;
    }
    // use omega^(j * seg_len) to reduce segs into leave
    const base_field *buffer_start = buffer + (g_id - num_ldes * num_querys) * TILE_O;
    base_field acc {0x0, 0x0};
    for (unsigned j = TILE_O; j > 0;) {
        acc = acc * omega_j;
        acc = acc + buffer_start[--j];
    }
    leave[query_id * num_polys + p_id] = acc;
  }
}

extern "C" __launch_bounds__(128, 8) __global__ 
  void fri_initial_siblings(const base_field *digests, base_field *siblings, const base_field *x_indexes,
    const unsigned num_querys, const unsigned num_polys, const unsigned log_n,
    const unsigned cap_height, const unsigned num_layers) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (g_id < num_querys) {
    uint64_t degree = 1 << log_n;
    uint64_t x_index = base_field::to_u64(x_indexes[g_id]) % degree;
    // x_indexes[g_id] = base_field::from_u64(x_index);
    // printf("gid: %d, x_index: %llu\n", g_id, x_index);
    // generate merkle path
    uint64_t tree_index = x_index >> num_layers;
    unsigned full_layers = num_layers + cap_height + 1;
    base_field *siblings_start = siblings + g_id * (4 * 4 * num_layers);
    uint64_t pair_index = x_index & ((1 << num_layers) - 1);
    for (unsigned i = 0; i < num_layers; i++) {
      uint64_t parity = pair_index & 1;
      pair_index = pair_index >> 1;
      uint64_t siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
      uint64_t sibling_index = 2 * siblings_index + (1 - parity);
      // let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
      // let sibling_index = 2 * siblings_index + (1 - parity);
      uint64_t id = sibling_index;
      unsigned layer = 0;
      while (true) {
        id = id >> 1;
        if (id & 1 == 1) {
          layer += 1;
        } else {
          break;
        }
      }
      uint64_t id_in_layer = (sibling_index + 2 - (1 << (layer + 1))) / (1 << (layer + 2)) * 2 + sibling_index % 2;
      uint64_t index = id_in_layer + (1 << full_layers) - (1 << (full_layers - layer));
      index += (tree_index << (num_layers - layer));
      for (unsigned j = 0; j < 4; j++) {
        siblings_start[4 * i + j] = digests[4 * index + j];
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__ 
  void fri_step(const base_field *leaves, const base_field *digests,
    base_field *leave, base_field *siblings, base_field *x_indexes, const unsigned num_querys,
    const unsigned arity_bits, const unsigned arity_sum, const unsigned arity_acc,
    const unsigned total_step_siblings, const unsigned acc_step_siblings, 
    const unsigned log_n, const unsigned cap_height) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id < num_querys) {
      // uint64_t degree = 1 << log_n;
      unsigned num_polys = 2 << arity_bits;
      uint64_t x_index = base_field::to_u64(x_indexes[g_id]) >> arity_bits;
      x_indexes[g_id] = base_field::from_u64(x_index);
      // printf("gid: %d, x_index: %llu, bits: %d, sum: %d, acc: %d\n", g_id, x_index, arity_bits, arity_sum, arity_acc);
      // copy leave
      base_field *leave_start = leave + g_id * arity_sum * 2 + arity_acc * 2; // D = 2
      const base_field *leaves_start = leaves + x_index * num_polys;
      for (unsigned i = 0; i < num_polys; i++) {
        leave_start[i] = leaves_start[i];
      }
      unsigned num_layers = log_n - cap_height;
      // generate merkle path
      uint64_t tree_index = x_index >> num_layers;
      unsigned full_layers = num_layers + cap_height + 1;
      base_field *siblings_start = siblings + g_id * total_step_siblings * 4 + acc_step_siblings * 4;
      uint64_t pair_index = x_index & ((1 << num_layers) - 1);
      for (unsigned i = 0; i < num_layers; i++) {
        uint64_t parity = pair_index & 1;
        pair_index = pair_index >> 1;
        uint64_t siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
        uint64_t sibling_index = 2 * siblings_index + (1 - parity);
        // let siblings_index = (pair_index << (i + 1)) + (1 << i) - 1;
        // let sibling_index = 2 * siblings_index + (1 - parity);
        uint64_t id = sibling_index;
        unsigned layer = 0;
        while (true) {
          id = id >> 1;
          if (id & 1 == 1) {
            layer += 1;
          } else {
            break;
          }
        }
        uint64_t id_in_layer = (sibling_index + 2 - (1 << (layer + 1))) / (1 << (layer + 2)) * 2 + sibling_index % 2;
        uint64_t index = id_in_layer + (1 << full_layers) - (1 << (full_layers - layer));
        index += (tree_index << (num_layers - layer));
        for (unsigned j = 0; j < 4; j++) {
          siblings_start[4 * i + j] = digests[4 * index + j];
        }
      }
    }

}

extern "C" __launch_bounds__(128, 8) __global__ 
  void copy_w_offset(const base_field *tree, base_field *cap, const unsigned offset, const unsigned num) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id < num) {
      cap[g_id] = tree[offset + g_id];
    }
}