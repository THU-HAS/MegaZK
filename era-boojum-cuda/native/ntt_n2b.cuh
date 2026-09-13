#pragma once // also, this file should only be compiled in one compile unit because it has __global__ definitions

#define TILE_Q 16
#define TILLE 32
#define LOG_TILE 5
#define TILE_ARRAY {0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31}
// #include "poseidon_single_thread.cuh"

// typedef field<3> poseidon_state[12];
// using namespace poseidon;

// extern "C" __launch_bounds__(128, 8) __global__
//     void compute_filter(bool inv) {
//       const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
//       if (g_id == 0)  printf("Hello, World! %d\n", inv);
//     }
#define N_CLGS 2

// DEVICE_FORCEINLINE void mul_pow_add(base_field in, base_field *acc, const base_field *alphas, base_field *alpha_pows) {
// #pragma unroll
//     for (unsigned i = 0; i < N_CLGS; i++) { // TODO: check if constant is faster
//         acc[i] = acc[i] + alpha_pows[i] * in;
//         alpha_pows[i] = alpha_pows[i] * alphas[i];
//     }
// }

// DEVICE_FORCEINLINE base_field compute_filter(const unsigned index, const base_field *constants_ptr, const unsigned row,
//         const unsigned selector_index, const unsigned start, const unsigned end, const unsigned nrows,
//         const bool verbose=false) {
//     base_field s = memory::load_cs(constants_ptr + index + selector_index * nrows);
//     base_field filter{0x1, 0x0};
//     const base_field UNUSED_SELECTOR{0xffffffff, 0x0};
//     if (verbose) printf("s: %llu, start: %d, end: %d, row: %d\n", s, start, end, row);
//     for (unsigned i = start; i < end; i++) {
//       if (i != row) {
//           filter = base_field::mul(filter, base_field::sub(base_field::from_u64(i), s));
//       }
//       if (verbose) printf("i: %d, filter: %llu\n", i, filter);
//     }
//     if (true) { // many selectors
//       filter = base_field::mul(filter, base_field::sub(UNUSED_SELECTOR, s));
//     }
//     if (verbose) printf("filter: %llu\n", filter);
//     return filter;
// }

DEVICE_FORCEINLINE unsigned reverse_bits(unsigned in, unsigned log_n) {
  unsigned out = 0;
  for (unsigned i = 0; i < log_n; i++) {
    if (in & (1 << i)) {
      out |= 1 << (log_n - 1 - i);
    }
  }
  return out;
}

// static DEVICE_FORCEINLINE void display(poseidon_state &state) {
// #pragma unroll 1
//   for (unsigned i = 0; i < 12; i++) {
//     field<2> value = field<3>::field3_to_field2(state[i]);
//     printf("(0x%x%x), ", value[1], value[0]);
//   }
//   printf("\n");
// }

// DEVICE_FORCEINLINE unsigned permute_state(poseidon_state &state) {
//     // display(state);
//     for (unsigned round = 0; round < 30; round++) {
//       if (round < 4 || round >= 26) {
//         poseidon::apply_round_constants(state, round);
//         poseidon::apply_non_linearity(state);
//         poseidon::apply_mds_matrix_new(state);
//       } else {
//         if (round == 4) {
//           poseidon::partial_rounds_init(state);
//         }
//         poseidon::partial_round_optimized(state, round);
//       }
//     }
//     // printf("state after permute:\n");
//     // display(state);
// }

// DEVICE_FORCEINLINE unsigned duplexing(poseidon_state &state,
//     base_field *input_buffer, const unsigned in_idx) {
//     for (unsigned j = 0; j < in_idx; j++) {
//         state[j] = base_field::into<3>(input_buffer[j]);
//     }
//     // display(state);
//     // poseidon::permutation(state);
//     for (unsigned round = 0; round < 30; round++) {
//       if (round < 4 || round >= 26) {
//         poseidon::apply_round_constants(state, round);
//         poseidon::apply_non_linearity(state);
//         poseidon::apply_mds_matrix_new(state);
//       } else {
//         if (round == 4) {
//           poseidon::partial_rounds_init(state);
//         }
//         poseidon::partial_round_optimized(state, round);
//       }
//     }
//     for (unsigned j = 0; j < 8; j++) {
//       input_buffer[j] = base_field::from_u64(0);
//     }
//     // printf("state after duplexing:\n");
//     // display(state);
// }

// DEVICE_FORCEINLINE unsigned set_output(poseidon_state &state,
//     base_field *output_buffer) {
//     for (unsigned j = 0; j < 8; j++) {
//         output_buffer[j] = base_field::field3_to_field2(state[j]);
//     }
// }

// DEVICE_FORCEINLINE unsigned set_sponge(poseidon_state &state,
//     base_field *sponge) {
//     for (unsigned j = 0; j < 12; j++) {
//         sponge[j] = base_field::field3_to_field2(state[j]);
//     }
// }

// extern "C" __launch_bounds__(128, 8) __global__
// void observe(base_field *elements, base_field *sponge,
//     base_field *input_buffer, base_field *output_buffer,
//     const unsigned num_elements,
//     const unsigned in_idx, const unsigned out_idx, const unsigned offset
// ) {
//     const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (g_id == 0) {
//         poseidon_state state{};
//         for (unsigned j = 0; j < 12; j++) {
//             state[j] = base_field::into<3>(sponge[j]);
//         }
//         unsigned current = in_idx;
//         for (unsigned i = 0; i < num_elements; i++) {
//             input_buffer[current] = elements[i + offset];
//             current = (current + 1) % 8;
//             if (current == 0) {
//                 duplexing(state, input_buffer, 8);
//                 if (num_elements - i <= 8) {
//                     set_output(state, output_buffer);
//                 }
//             }
//         }
//         set_sponge(state, sponge);
//     }
// }

// extern "C" __launch_bounds__(128, 8) __global__
// void get(base_field *challenges, base_field *sponge,
//     base_field *input_buffer, base_field *output_buffer,
//     const unsigned num_challenges,
//     const unsigned in_idx, const unsigned out_idx, const unsigned offset
// ) {
//     const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (g_id == 0) {
//         poseidon_state state{};
//         for (unsigned j = 0; j < 12; j++) {
//             state[j] = base_field::into<3>(sponge[j]);
//         }
//         unsigned current = out_idx;
//         if (in_idx != 0 || out_idx == 0) {
//             duplexing(state, input_buffer, in_idx);
//             set_output(state, output_buffer);
//             current = 8;
//         }
//         for (unsigned i = 0; i < num_challenges; i++) {
//             if (current == 0) {
//                 duplexing(state, input_buffer, 0);
//                 set_output(state, output_buffer);
//                 current = 8;
//             }
//             challenges[i + offset] = output_buffer[current - 1];
//             // printf("%x\n", base_field::to_u64(challenges[i]));
//             current = (current + 7) % 8;
//         }
//         set_sponge(state, sponge);
//     }
// }

// extern "C" __launch_bounds__(128, 8) __global__
//   void fri_pow(const base_field *input_buffer, const base_field *sponge, base_field *witness,
//   unsigned *found, const base_field stride, const unsigned in_idx, const unsigned min_leading_zeros) {
//   const uint64_t g_id = blockIdx.x * blockDim.x + threadIdx.x;
//   // uint64_t threshold = 1 << (63 - min_leading_zeros);
//   // threshold = threshold * 2;
//   // if (g_id == 0) printf("start pow, threshold:%llu\n", threshold);
//   base_field current = base_field::from_u64(g_id);
//   __shared__ field<3> init_state[12];
//   if (threadIdx.x == 0) {
//     for (unsigned j = 0; j < 12; j++) {
//       init_state[j] = base_field::into<3>(sponge[j]);
//     }
//     for (unsigned j = 0; j < in_idx; j++) {
//       init_state[j] = base_field::into<3>(input_buffer[j]);
//     }
//   }
//   // if (g_id == 0) display(init_state);
//   uint64_t best_more = 1 << (min_leading_zeros + 2);
//   while ((*found) != 1) {
//     for (unsigned i = 0; i < (1 << 4); i++) {
//       poseidon_state state{};
//       for (unsigned j = 0; j < 12; j++) {
//         state[j] = init_state[j];
//       }
//       state[in_idx] = base_field::into<3>(current);
//       permute_state(state);
//       uint64_t clg = base_field::to_u64(base_field::field3_to_field2(state[7]));
//       uint64_t more = clg >> (63 - min_leading_zeros);
//       // if (g_id == 0) printf("clg: %llu, more: %llu\n", clg, more);
//       if (more < best_more) best_more = more;
//       // if (clg < threshold) {
//       if (more == 0) {
//         if ((*found) != 1) {
//           *found = 1;
//           *witness = current;
//           // printf("found! gid: %llu, w: %llu, clg: %llu\n", g_id, current, clg);
//         }
//       }
//       current = base_field::add(current, stride);
//     }
//     // break;
//     // if (g_id == 0) printf("best more: %u, another loop...\n", best_more);
//   }
//   // base_field w = *witness;
//   // poseidon_state state{};
//   // for (unsigned j = 0; j < 12; j++) {
//   //   state[j] = init_state[j];
//   // }
//   // state[in_idx] = base_field::into<3>(w);
//   // permute_state(state);
//   // uint64_t clg = base_field::to_u64(base_field::field3_to_field2(state[7]));
//   // if (g_id == 0) printf("final found! w: %llu, clg: %llu\n", w, clg);
// }

// extern "C" __launch_bounds__(128, 8) __global__
//   void fri_pow_new(const base_field *input_buffer, const base_field *sponge, base_field *witness, unsigned *found, unsigned *found_buf,
//   base_field *buf, const base_field stride, const unsigned in_idx, const unsigned min_leading_zeros, const unsigned offset) {
//   const uint64_t g_id = blockIdx.x * blockDim.x + threadIdx.x;
//   base_field current = base_field::from_u64(g_id + offset);
//   __shared__ field<3> init_state[12];
//   if (threadIdx.x == 0) {
// #pragma unroll
//     for (unsigned j = 0; j < 12; j++) {
//       init_state[j] = base_field::into<3>(sponge[j]);
//     }
//     for (unsigned j = 0; j < in_idx; j++) {
//       init_state[j] = base_field::into<3>(input_buffer[j]);
//     }
//   }
// #pragma unroll
//   for (unsigned i = 0; i < (1 << 3); i++) {
//     poseidon_state state{};
// #pragma unroll
//     for (unsigned j = 0; j < 12; j++) {
//       state[j] = init_state[j];
//     }
//     state[in_idx] = base_field::into<3>(current);
//     permute_state(state);
//     uint64_t clg = base_field::to_u64(base_field::field3_to_field2(state[7]));
//     uint64_t more = clg >> (63 - min_leading_zeros);
//     if (more == 0) {
//         buf[g_id] = current;
//         found_buf[g_id] = 1;
//         break;
//         // printf("found! gid: %llu, w: %llu, clg: %llu\n", g_id, current, clg);
//       }
//     current = base_field::add(current, stride);
//   }
//   if (g_id == 0) {
//     const unsigned num_threads = base_field::to_u64(stride);
//     for (unsigned i = 0; i < num_threads; i++) {
//       if (found_buf[i] == 1) {
//         *witness = buf[i];
//         *found = 1;
//         break;
//       }
//     }
//   }
// }

// extern "C" __launch_bounds__(128, 8) __global__
// void hash_pi(const base_field *witness, const size_t *pi_index,
//     base_field *pi, base_field *pi_hash, const size_t num_public_inputs
// ) {
//     const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (g_id == 0) {
//         // printf("Hello, PI!\n");
//         poseidon_state state{};
//         const unsigned num_blocks = num_public_inputs >> 3;
//         const unsigned num_left = num_public_inputs - (num_blocks << 3);
//         for (unsigned i = 0; i < num_blocks; i++) {
//             for (unsigned j = 0; j < 8; j++) {
//               base_field p = witness[pi_index[j + 8 * i]];
//               pi[j + 8 * i] = p;
//               state[j] = base_field::into<3>(p);
//             }
//             permute_state(state);
//         }
//         for (unsigned j = 0; j < num_left; j++) {
//             base_field p = witness[pi_index[j + 8 * num_blocks]];
//             state[j] = base_field::into<3>(p);
//             pi[j + 8 * num_blocks] = p;
//         }
//         if (num_left != 0) {
//             permute_state(state);
//         }
//         for (unsigned j = 0; j < 4; j++) {
//             pi_hash[j] = base_field::field3_to_field2(state[j]);
//         }
//     }
// }

extern "C" __launch_bounds__(128, 8) __global__
    void z_partial(const base_field *points_ptr, const base_field *z_h_coset_ptr, const base_field *k_is_ptr,
      const base_field *wires_ptr, const base_field *zs_partial_products_ptr, const base_field *constants_ptr,
      const base_field *betas, const base_field *gammas, base_field *output, const unsigned degree_bits,
      const unsigned quotient_degree_bits, const unsigned num_wires, const unsigned num_const_sigmas, unsigned num_challenges,
      const unsigned num_routed_wires, const unsigned num_partial_products, const unsigned num_gate_constraints
    ) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned nrows = 1 << degree_bits;
  const unsigned row_step = nrows / TILE_Q;
  const unsigned num_zpp = (num_routed_wires >> quotient_degree_bits) * num_challenges;
  const unsigned rate = 1 << quotient_degree_bits;
  // const unsigned stride = num_partial_products + 1;
  const unsigned const_offset = num_const_sigmas - num_routed_wires;
  if (g_id < row_step) {
#pragma unroll
    for (unsigned j = 0; j < TILE_Q; j++) { // TODO!
      const unsigned index = g_id + j * row_step;
      // const unsigned lde_started = reverse_bits(index, degree_bits);
      const unsigned lde_started = index;
      // const unsigned next_lde_started = reverse_bits((index + rate) % nrows, degree_bits);
      const unsigned next_lde_started = (index + rate) % nrows;
      const base_field *wire_start = wires_ptr + lde_started;
      const base_field *const_start = constants_ptr + lde_started + const_offset * nrows;
      unsigned cons_offset = num_challenges * nrows;
      base_field *out_start = output + index;
      const base_field x = base_field::mul(base_field::from_u64(7), memory::load_cs(points_ptr + index));
      const base_field *zpp_start = zs_partial_products_ptr + lde_started;
      const base_field *next_zpp_start = zs_partial_products_ptr + next_lde_started;
      const base_field l_0_x = base_field::mul(memory::load_cs(z_h_coset_ptr + (index % rate)),
                                               base_field::inv(base_field::mul(base_field::from_u64(nrows >> quotient_degree_bits), // WARNING: rate bits!
                                                                               base_field::add(x, base_field::minus_one()))));
      // if (g_id == 0) printf("gid: %d, l0x: %lld, x: %lld, index: %d\n", g_id, base_field::to_u64(l_0_x), base_field::to_u64(x), index);
      base_field con;
      for (unsigned k = 0; k < num_challenges; k++) {
        // base_field *start = out_start + k * stride;
        const base_field beta = memory::load_cs(betas + k);
        const base_field gamma = memory::load_cs(gammas + k);
        // z_1_terms
        const base_field z_x = memory::load_cs(zpp_start + k * nrows);
        const base_field z_gx = memory::load_cs(next_zpp_start + k * nrows);
        con = base_field::mul(l_0_x, base_field::add(z_x, base_field::minus_one()));
        // if (g_id == 0 && j == 3) printf("cons: %llu\n", con);
        memory::store_cs(out_start + k * nrows, con);
        // pp: zpp_start + num_chalenges + k * num_partial_products

        const base_field *pp_start = zpp_start + (num_challenges + k * num_partial_products) * nrows;
        base_field numerator_acc{0x1, 0x0}, denominator_acc{0x1, 0x0};
        for (unsigned r = 0; r < rate; r++) {
          const base_field wire = memory::load_cs(wire_start + r * nrows);
          numerator_acc = base_field::mul(
              numerator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, base_field::mul(x, memory::load_cs(k_is_ptr + r))), gamma)));
          denominator_acc =
              base_field::mul(denominator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, memory::load_cs(const_start + r * nrows)), gamma)));
        }
        base_field pp = memory::load_cs(pp_start);
        con = base_field::sub(base_field::mul(z_x, numerator_acc), base_field::mul(pp, denominator_acc));
        // if (g_id == 0 && j == 3) printf("cons: %llu\n", con);
        memory::store_cs(out_start + cons_offset, con);
        cons_offset += nrows;
        for (unsigned i = 1; i < num_partial_products; i++) {
          base_field numerator_acc{0x1, 0x0}, denominator_acc{0x1, 0x0};
          for (unsigned r = 0; r < rate; r++) {
            unsigned col = i * rate + r;
            const base_field wire = memory::load_cs(wire_start + col * nrows);
            numerator_acc = base_field::mul(
                numerator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, base_field::mul(x, memory::load_cs(k_is_ptr + col))), gamma)));
            denominator_acc =
                base_field::mul(denominator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, memory::load_cs(const_start + col * nrows)), gamma)));
          }
          base_field pp_prev = pp;
          pp = memory::load_cs(pp_start + i * nrows);
          // base_field pp_prev = memory::load_cs(pp_start + i - 1);
          con = base_field::sub(base_field::mul(pp_prev, numerator_acc), base_field::mul(pp, denominator_acc));
          // if (g_id == 0 && j == 3) printf("cons: %llu\n", con);
          memory::store_cs(out_start + cons_offset, con);
          cons_offset += nrows;
        }
        numerator_acc = base_field::from_u64(1);
        denominator_acc = base_field::from_u64(1);
        for (unsigned r = 0; r < rate; r++) {
          unsigned col = num_partial_products * rate + r;
          const base_field wire = memory::load_cs(wire_start + col * nrows);
          numerator_acc = base_field::mul(
              numerator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, base_field::mul(x, memory::load_cs(k_is_ptr + col))), gamma)));
          denominator_acc =
              base_field::mul(denominator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, memory::load_cs(const_start + col * nrows)), gamma)));
        }
        con = base_field::sub(base_field::mul(pp, numerator_acc), base_field::mul(z_gx, denominator_acc));
        // if (g_id == 0 && j == 3) printf("cons: %llu\n", con);
        memory::store_cs(out_start + cons_offset, con);
        cons_offset += nrows;
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void reduce_q(const base_field *alphas, const base_field *z_h_coset, const base_field *constraints_ptr, base_field *quotient_ptr,
                  const unsigned degree_bits, const unsigned quotient_degree_bits, const unsigned constraints_per_point, const unsigned num_challenges) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned nrows = 1 << (degree_bits + quotient_degree_bits);
  const unsigned row_step = nrows / TILE_Q;
  const unsigned rate = 1 << quotient_degree_bits;
  // if (g_id == 0) printf("gid: %d, alpha: %lld\n", g_id, base_field::to_u64(memory::load_cs(alphas + 1)));
  // Calculate constraints
  if (g_id < row_step) {
#pragma unroll
    for (unsigned k = 0; k < num_challenges; k++) {
      base_field alpha = memory::load_cs(alphas + k);
      for (unsigned j = 0; j < TILE_Q; j++) {
        base_field acc{0x0, 0x0};
        // base_field a = alpha;
        unsigned index = g_id + j * row_step;
        for (int i = (constraints_per_point - 1) * nrows + index; i >= 0; i -= nrows) {
          base_field constraint = memory::load_cs(constraints_ptr + i);
          acc = base_field::add(constraint, base_field::mul(alpha, acc));
          // if (g_id == 0 && j == 3 && k == 0 && i < 30 * nrows) printf("cons_filtered: %llu, rev_acc[0]: %llu\n", constraint, acc);
          // if (g_id == 0 && j == 0 && k == 0) {
          //   printf("cons: %llu, acc: %llu, i: %d\n", base_field::to_u64(constraint), base_field::to_u64(acc), i / nrows);
          // }
        }
        acc = base_field::mul(acc, memory::load_cs(z_h_coset + rate + (g_id % rate)));
        // if (g_id == 0 && j == 3 && k == 0) printf("q_value: %llu, z_h_inv: %llu\n", acc, memory::load_cs(z_h_coset + rate + (g_id % rate)));
        memory::store_cs(quotient_ptr + k * nrows + index, acc);
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void noop_gate(const base_field *wires_ptr, const base_field *constants_ptr, const base_field *public_inputs_hash, base_field *output,
                   const unsigned degree_bits, const unsigned num_wires, const unsigned num_const_sigmas, const unsigned num_gate_constraints,
                   const unsigned num_selectors, const unsigned num_constants,
                           const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants) {}

extern "C" __launch_bounds__(128, 8) __global__
    void constant_gate(const base_field *wires_ptr, const base_field *constants_ptr, const base_field *public_inputs_hash, base_field *output,
                       const unsigned degree_bits, const unsigned num_wires, const unsigned num_const_sigmas, const unsigned num_gate_constraints,
                       const unsigned num_selectors, const unsigned num_constants,
                               const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned nrows = 1 << degree_bits;
  const unsigned row_step = nrows / TILE_Q;
  // Calculate constraints
  if (g_id < row_step) {
#pragma unroll
    for (unsigned j = 0; j < TILE_Q; j++) {
      // const unsigned lde_started = reverse_bits(g_id + j * row_step, degree_bits);
      const unsigned lde_started = g_id + j * row_step;
      for (unsigned i = 0; i < num_constants; i++) {
        // base_field constraint = base_field::sub(memory::load_cs(constants_ptr + lde_started * num_const_sigmas + (i + num_selectors)),
        //                                         memory::load_cs(wires_ptr + lde_started * num_wires + i) //
        // );
        base_field constraint = base_field::sub(memory::load_cs(constants_ptr + lde_started + (i + num_selectors) * nrows),
                                                memory::load_cs(wires_ptr + lde_started + i * nrows) //
        );
        // if (g_id == 0 && j == 3) printf("cons: %llu\n", constraint);
        memory::store_cs(output + g_id + j * row_step + i * nrows, constraint);
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void public_input_gate(const base_field *wires_ptr, const base_field *constants_ptr, const base_field *public_inputs_hash, base_field *output,
                           const unsigned degree_bits, const unsigned num_wires, const unsigned num_const_sigmas, const unsigned num_gate_constraints,
                           const unsigned num_selectors, const unsigned num_constants,
                                   const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned nrows = 1 << degree_bits;
  const unsigned row_step = nrows / TILE_Q;
  // Calculate constraints
  if (g_id < row_step) {
#pragma unroll
    for (unsigned j = 0; j < TILE_Q; j++) {
      // const unsigned lde_started = reverse_bits(g_id + j * row_step, degree_bits);
      const unsigned lde_started = g_id + j * row_step;
#pragma unroll
      for (unsigned i = 0; i < 4; i++) {
        // base_field constraint = base_field::sub(memory::load_cs(wires_ptr + lde_started * num_wires + i), memory::load_cs(public_inputs_hash + i));
        base_field constraint = base_field::sub(memory::load_cs(wires_ptr + lde_started + i * nrows), memory::load_cs(public_inputs_hash + i));
        memory::store_cs(output + g_id + j * row_step + i * nrows, constraint);
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void arithmetic_gate(const base_field *wires_ptr, const base_field *constants_ptr, const base_field *public_inputs_hash, base_field *output,
                         const unsigned degree_bits, const unsigned num_wires, const unsigned num_const_sigmas, const unsigned num_gate_constraints,
                         const unsigned num_selectors, const unsigned num_constants,
                                 const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned nrows = 1 << degree_bits;
  const unsigned row_step = nrows / TILE_Q;
  // Calculate constraints
  if (g_id < row_step) {
#pragma unroll
    for (unsigned j = 0; j < TILE_Q; j++) {
      // const unsigned lde_started = reverse_bits(g_id + j * row_step, degree_bits);
      const unsigned lde_started = g_id + j * row_step;
      const base_field c0 = memory::load_cs(constants_ptr + lde_started + num_selectors * nrows);
      const base_field c1 = memory::load_cs(constants_ptr + lde_started + (num_selectors + 1) * nrows);
#pragma unroll
      for (unsigned i = 0; i < num_ops; i++) {
        base_field m0 = memory::load_cs(wires_ptr + lde_started + (4 * i) * nrows);
        base_field m1 = memory::load_cs(wires_ptr + lde_started + (4 * i + 1) * nrows);
        base_field ad = memory::load_cs(wires_ptr + lde_started + (4 * i + 2) * nrows);
        base_field oput = memory::load_cs(wires_ptr + lde_started + (4 * i + 3) * nrows);
        base_field computed = base_field::add(base_field::mul(base_field::mul(m0, m1), c0), base_field::mul(ad, c1));
        base_field constraint = base_field::sub(oput, computed);
        memory::store_cs(output + g_id + j * row_step + i * nrows, constraint);
      }
    }
  }
}

// extern "C" __launch_bounds__(128, 8) __global__
//     void poseidon_gate(const base_field *wires_ptr, const base_field *constants_ptr, const base_field *public_inputs_hash, base_field *output,
//                        const unsigned degree_bits, const unsigned num_wires, const unsigned num_const_sigmas, const unsigned num_gate_constraints,
//                        const unsigned num_selectors, const unsigned num_constants,
//                                const unsigned num_ops,
//         const unsigned num_consts,
//         const unsigned num_limbs,
//         const unsigned num_addends,
//         const unsigned num_bits,
//         const unsigned num_chunks,
//         const unsigned num_input_limbs,
//         const unsigned B,
//         const unsigned bits,
//         const unsigned num_copies,
//         const unsigned num_extra_constants) {
//   const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
//   const unsigned nrows = 1 << degree_bits;
//   const unsigned row_step = nrows / TILE_Q;
//   // Calculate constraints
//   if (g_id < row_step) {
// #pragma unroll
//     for (unsigned j = 0; j < TILE_Q; j++) {
//       // Init, 5
//       // const unsigned lde_started = reverse_bits(g_id + j * row_step, degree_bits);
//       const unsigned lde_started = g_id + j * row_step;
//       const base_field *wire_start = wires_ptr + lde_started;
//       base_field *cons_start = output + g_id + j * row_step;
//       unsigned cons_offset = 0;
//       const base_field swap = memory::load_cs(wire_start + 24 * nrows);
//       memory::store_cs(cons_start + cons_offset, base_field::mul(swap, base_field::add(swap, base_field::minus_one())));
//       cons_offset += nrows;
//       // base_field state[12];
//       poseidon_state state{};
// #pragma unroll
//       for (unsigned i = 0; i < 4; i++) {
//         base_field input_lhs = memory::load_cs(wire_start + i * nrows);
//         base_field input_rhs = memory::load_cs(wire_start + (i + 4) * nrows);
//         base_field delta_i = memory::load_cs(wire_start + (i + 25) * nrows);
//         memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::mul(swap, base_field::sub(input_rhs, input_lhs)), delta_i));
//         cons_offset += nrows;
//         state[i] = base_field::into<3>(base_field::add(input_lhs, delta_i));
//         state[i + 4] = base_field::into<3>(base_field::sub(input_rhs, delta_i));
//       }
// #pragma unroll
//       for (unsigned i = 8; i < 12; i++) {
//         state[i] = base_field::into<3>(memory::load_cs(wire_start + i * nrows));
//       }

//       unsigned round_ctr = 0;
//       // First set of full rounds, 36
// #pragma unroll
//       for (unsigned r = 0; r < 4; r++) {
//         poseidon::apply_round_constants(state, round_ctr);
//         if (r != 0) {
// #pragma unroll
//           for (unsigned i = 0; i < 12; i++) {
//             base_field sbox_in = memory::load_cs(wire_start + (29 + 12 * (r - 1) + i) * nrows);
//             memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), sbox_in));
//             cons_offset += nrows;
//             state[i] = base_field::into<3>(sbox_in);
//           }
//         }
//         poseidon::apply_non_linearity(state);
//         poseidon::apply_mds_matrix_new(state);
//         round_ctr += 1;
//       }

//       // Partial rounds, 22
//       poseidon::partial_rounds_init(state);
// #pragma unroll
//       for (unsigned r = 0; r < 22; r++) {
//         base_field sbox_in = memory::load_cs(wire_start + (29 + 12 * 3 + r) * nrows); // START_FULL_0 + SPONGE_WIDTH * (poseidon::HALF_N_FULL_ROUNDS - 1) + round
//         memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[0]), sbox_in));
//         cons_offset += nrows;
//         state[0] = base_field::into<3>(sbox_in);
//         poseidon::partial_round_optimized(state, r + 4); // + HALF_NUM_FULL_ROUNDS
//       }
//       round_ctr += 22;

//       // Second set of full rounds, 48
// #pragma unroll
//       for (unsigned r = 0; r < 4; r++) {
//         poseidon::apply_round_constants(state, round_ctr);
// #pragma unroll
//         for (unsigned i = 0; i < 12; i++) {
//           base_field sbox_in = memory::load_cs(wire_start + (87 + 12 * r + i) * nrows);
//           memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), sbox_in));
//           cons_offset += nrows;
//           state[i] = base_field::into<3>(sbox_in);
//         }
//         poseidon::apply_non_linearity(state);
//         poseidon::apply_mds_matrix_new(state);
//         round_ctr += 1;
//       }

//       // Output, 12
// #pragma unroll
//       for (unsigned i = 0; i < 12; i++) {
//         base_field output = memory::load_cs(wire_start + (i + 12) * nrows);
//         memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), output));
//         cons_offset += nrows;
//       }
//     }
//   }
// }

// extern "C" __launch_bounds__(128, 8) __global__
//     void poseidon_gate_in_place(const base_field *wires_ptr, const base_field *constants_ptr, const base_field *public_inputs_hash, base_field *output, const base_field *alphas,
//                        const unsigned degree_bits, const unsigned num_wires, const unsigned num_const_sigmas, const unsigned num_gate_constraints,
//                        const unsigned num_selectors, const unsigned num_constants,
//         const unsigned param1,
//         const unsigned param2,
//         const unsigned param3,
//         const unsigned row,
//         const unsigned selector_index, const unsigned start, const unsigned end) {
//   const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
//   const unsigned nrows = 1 << degree_bits;
//   const unsigned row_step = nrows / TILE_Q;
//   // Calculate constraints
//   if (g_id < row_step) {
// #pragma unroll
//     for (unsigned j = 0; j < TILE_Q; j++) {
//       // Init, 5
//       // const unsigned lde_started = reverse_bits(g_id + j * row_step, degree_bits);
//       const unsigned index = g_id + j * row_step;
//       // Calculate filter
//       base_field filter = compute_filter(index, constants_ptr, row, selector_index, start, end, nrows);

//       base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
//       base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

//       const base_field *wire_start = wires_ptr + index;
//       base_field *cons_start = output + index;
//       unsigned cons_offset = 0;
//       const base_field swap = memory::load_cs(wire_start + 24 * nrows);
//     //   memory::store_cs(cons_start + cons_offset, base_field::mul(swap, base_field::add(swap, base_field::minus_one())));
//     //   cons_offset += nrows;
//       base_field cons = base_field::mul(swap, base_field::add(swap, base_field::minus_one()));
//       mul_pow_add(cons, acc, alphas, alpha_pows);
//       // base_field state[12];
//       poseidon_state state{};
// #pragma unroll
//       for (unsigned i = 0; i < 4; i++) {
//         base_field input_lhs = memory::load_cs(wire_start + i * nrows);
//         base_field input_rhs = memory::load_cs(wire_start + (i + 4) * nrows);
//         base_field delta_i = memory::load_cs(wire_start + (i + 25) * nrows);
//         // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::mul(swap, base_field::sub(input_rhs, input_lhs)), delta_i));
//         // cons_offset += nrows;
//         cons = base_field::sub(base_field::mul(swap, base_field::sub(input_rhs, input_lhs)), delta_i);
//         mul_pow_add(cons, acc, alphas, alpha_pows);
//         state[i] = base_field::into<3>(base_field::add(input_lhs, delta_i));
//         state[i + 4] = base_field::into<3>(base_field::sub(input_rhs, delta_i));
//       }
// #pragma unroll
//       for (unsigned i = 8; i < 12; i++) {
//         state[i] = base_field::into<3>(memory::load_cs(wire_start + i * nrows));
//       }

//       unsigned round_ctr = 0;
//       // First set of full rounds, 36
// #pragma unroll
//       for (unsigned r = 0; r < 4; r++) {
//         poseidon::apply_round_constants(state, round_ctr);
//         if (r != 0) {
// #pragma unroll
//           for (unsigned i = 0; i < 12; i++) {
//             base_field sbox_in = memory::load_cs(wire_start + (29 + 12 * (r - 1) + i) * nrows);
//             // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), sbox_in));
//             // cons_offset += nrows;
//             cons = base_field::sub(base_field::field3_to_field2(state[i]), sbox_in);
//             mul_pow_add(cons, acc, alphas, alpha_pows);
//             state[i] = base_field::into<3>(sbox_in);
//           }
//         }
//         poseidon::apply_non_linearity(state);
//         poseidon::apply_mds_matrix_new(state);
//         round_ctr += 1;
//       }

//       // Partial rounds, 22
//       poseidon::partial_rounds_init(state);
// #pragma unroll
//       for (unsigned r = 0; r < 22; r++) {
//         base_field sbox_in = memory::load_cs(wire_start + (29 + 12 * 3 + r) * nrows); // START_FULL_0 + SPONGE_WIDTH * (poseidon::HALF_N_FULL_ROUNDS - 1) + round
//         // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[0]), sbox_in));
//         // cons_offset += nrows;
//         cons = base_field::sub(base_field::field3_to_field2(state[0]), sbox_in);
//         mul_pow_add(cons, acc, alphas, alpha_pows);
//         state[0] = base_field::into<3>(sbox_in);
//         poseidon::partial_round_optimized(state, r + 4); // + HALF_NUM_FULL_ROUNDS
//       }
//       round_ctr += 22;

//       // Second set of full rounds, 48
// #pragma unroll
//       for (unsigned r = 0; r < 4; r++) {
//         poseidon::apply_round_constants(state, round_ctr);
// #pragma unroll
//         for (unsigned i = 0; i < 12; i++) {
//           base_field sbox_in = memory::load_cs(wire_start + (87 + 12 * r + i) * nrows);
//         //   memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), sbox_in));
//         //   cons_offset += nrows;
//           cons = base_field::sub(base_field::field3_to_field2(state[i]), sbox_in);
//           mul_pow_add(cons, acc, alphas, alpha_pows);
//           state[i] = base_field::into<3>(sbox_in);
//         }
//         poseidon::apply_non_linearity(state);
//         poseidon::apply_mds_matrix_new(state);
//         round_ctr += 1;
//       }

//       // Output, 12
// #pragma unroll
//       for (unsigned i = 0; i < 12; i++) {
//         base_field output = memory::load_cs(wire_start + (i + 12) * nrows);
//         // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), output));
//         // cons_offset += nrows;
//         cons = base_field::sub(base_field::field3_to_field2(state[i]), output);
//         mul_pow_add(cons, acc, alphas, alpha_pows);
//       }
//       // Store to quotient
// #pragma unroll
//       for (unsigned k = 0; k < N_CLGS; k++) {
//         base_field quotient_acc = output[index + k * nrows * 8];
//         output[index + k * nrows * 8] = filter * acc[k] + quotient_acc;
//       }
//     }
//   }
// }

extern "C" __launch_bounds__(128, 8) __global__
    void filter(const base_field *constants_ptr, base_field *buffer_ptr, base_field *constraint_ptr, const unsigned degree_bits, const unsigned row,
                const unsigned selector_index, const unsigned start, const unsigned end, const unsigned num_const_sigmas, const unsigned num_gate_constraints,
                const unsigned offset) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned nrows = 1 << degree_bits;
  const unsigned row_step = nrows / TILE_Q;
  const base_field UNUSED_SELECTOR{0xffffffff, 0x0};
  // if (g_id == 0) printf("Hello, World, row step:%d!\n\n", row_step);
  // Calculate constraints
  if (g_id < row_step) {
#pragma unroll
    for (unsigned j = 0; j < TILE_Q; j++) {
      // const unsigned lde_started = reverse_bits(g_id + j * row_step, degree_bits);
      const unsigned lde_started = g_id + j * row_step;
      base_field s = memory::load_cs(constants_ptr + lde_started + selector_index * nrows);
      base_field acc{0x1, 0x0};
      // if (g_id == 0 && j == 3) printf("s: %llu, start: %d, end: %d, row: %d\n", s, start, end, row);
      for (unsigned i = start; i < end; i++) {
        if (i != row) {
          acc = base_field::mul(acc, base_field::sub(base_field::from_u64(i), s));
        }
        // if (g_id == 0 && j == 3) printf("filter: %llu\n", acc);
      }
      if (true) { // many selectors
        acc = base_field::mul(acc, base_field::sub(UNUSED_SELECTOR, s));
      }
      // if (g_id == 0 && j == 3) printf("filter: %llu\n", acc);
      // if (g_id == 0) printf("gid: %d, acc: %lld, s: %lld, c: %d\n", g_id, base_field::to_u64(acc), base_field::to_u64(s), lde_started);
      for (unsigned i = 0; i < num_gate_constraints; i++) {
        unsigned index = g_id + j * row_step + i * nrows;
        base_field former = memory::load_cs(constraint_ptr + index + offset);
        base_field unfiltered = memory::load_cs(buffer_ptr + index);
        // memory::store_cs(buffer_ptr + index, base_field::zero());
        base_field filtered = base_field::add(former, base_field::mul(acc, unfiltered));
        // if (g_id == 0 && j == 3 && row == 1 && i < 10) printf("former: %llu, unfiltered: %llu, filtered: %llu, i: %d\n", base_field::to_u64(former),
        // base_field::to_u64(unfiltered), base_field::to_u64(filtered), i);
        memory::store_cs(constraint_ptr + index + offset, filtered);
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void init_q(base_field *points_ptr, base_field *z_h_coset_ptr, base_field *k_is_ptr,
                  const unsigned degree_bits, const unsigned quotient_degree_bits,
                  const unsigned num_routed_wires) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  // const unsigned root = 1753635133440165772;
  const unsigned exp_of_2 = 32 - degree_bits - quotient_degree_bits; // TWO_ADICITY - degree_bits
  const unsigned degree = 1 << (degree_bits + quotient_degree_bits);
  const unsigned stride_within = degree / TILLE;
  base_field *points_start = points_ptr + g_id;
  if (g_id < stride_within) {
    // if (g_id == 0) printf("stride: %d, exp_of_2: %d\n", stride_within, exp_of_2);
    base_field root{0xda58878c,0x185629dc}, a{0x1,0x0};
    for (unsigned i = 0; i < exp_of_2; i++) {
      root = base_field::sqr(root);
    }
    unsigned c = 1, ggid = g_id;
    base_field b_used = root;
    while (c != stride_within) {
      if (ggid % 2 == 1)
        a = base_field::mul(a, b_used);
      b_used = base_field::sqr(b_used);
      c *= 2;
      ggid /= 2;
      // // if (g_id == 0) printf("gid: %d, a: %lld, b: %lld, c: %d\n", g_id, base_field::to_u64(a), base_field::to_u64(b_used), c);
      // printf("gid: %d, a: %lld, b: %lld, c: %d\n", g_id, base_field::to_u64(a), base_field::to_u64(b_used), c);
    }
    // printf("gid: %d, a: %lld\n", g_id, base_field::to_u64(a));

#pragma unroll
    for (unsigned i = 0; i < TILLE; i++) {
      // base_field coseted = base_field::mul(memory::load_cs(gmem_inputs_start + i * stride_within), a);
      memory::store_cs(points_start + i * stride_within, a);
      a = base_field::mul(a, b_used);
      // if (g_id == 0) printf("gid: %d, a: %lld, b: %lld, c: %d\n", g_id, base_field::to_u64(a), base_field::to_u64(b_used), c);
    }
  }
  const unsigned z_h_length = 1 << quotient_degree_bits;
  if (g_id < z_h_length) {
    base_field g_pow_n{0x7, 0x0};
    for (unsigned i = 0; i < degree_bits; i++) {
      g_pow_n = base_field::sqr(g_pow_n);
    }
    base_field root_z{0xda58878c, 0x185629dc};
    const unsigned exp_of_2_z = 32 - quotient_degree_bits;
    for (unsigned i = 1; i < exp_of_2_z; i++) {
      root_z = base_field::sqr(root_z);
    }
    base_field x{0x1, 0x0};
    for (unsigned i = 0; i < quotient_degree_bits; i++) {
      root_z = base_field::sqr(root_z);
      if ((g_id >> i) & 1) {
        x = base_field::mul(x, root_z);
      }
    }
    // printf("gid: %d, x: %lld, exp_of_2_z: %d\n", g_id, base_field::to_u64(x), exp_of_2_z);
    base_field eval = base_field::add(base_field::mul(g_pow_n, x), base_field::minus_one());
    base_field inv = base_field::inv(eval);
    memory::store_cs(z_h_coset_ptr + g_id, eval);
    memory::store_cs(z_h_coset_ptr + g_id + z_h_length, inv);
  }
  if (g_id < num_routed_wires) {
    base_field k{0x7,0x0}, k_i{0x1,0x0};
    unsigned c = 1, ggid = g_id;
    while (c <= g_id) {
      if (ggid % 2 == 1) {
        k_i = base_field::mul(k_i, k);
      }
      k = base_field::sqr(k);
      c *= 2;
      ggid /= 2;
    }
    memory::store_cs(k_is_ptr + g_id, k_i);
  }
}



DEVICE_FORCEINLINE void mul_ext(const base_field *in1, const base_field *in2, base_field *out) {
  base_field c{0x7, 0x0};
  base_field tmp0 = base_field::add(base_field::mul(in1[0], in2[0]), base_field::mul(c, base_field::mul(in1[1], in2[1])));
  base_field tmp1 = base_field::add(base_field::mul(in1[0], in2[1]), base_field::mul(in1[1], in2[0]));
  out[0] = tmp0;
  out[1] = tmp1;
}

extern "C" __launch_bounds__(128, 8) __global__
    void copy_gpu(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned log_n, const unsigned num_ntts,
                  const unsigned stride_between_input_arrays, const unsigned stride_between_output_arrays, const unsigned rate_bits, const bool inverse) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  base_field *gmem_outputs_start = gmem_outputs_matrix + (stride_between_input_arrays << log_n);
  for (unsigned i = 0; i < num_ntts; i++) {
    base_field tmp = memory::load_cs(gmem_inputs_matrix + g_id + (i << log_n));
    memory::store_cs(gmem_outputs_start + g_id + (i << log_n), tmp);
    // if (g_id == 0 && i == 0) printf("gid: %d, tmp: %lld\n", g_id, base_field::to_u64(tmp));
  }
}

extern "C" __launch_bounds__(128, 8) __global__ void final_poly(const base_field *gmem_inputs_matrix, const base_field *alpha_matrix,
                                                                base_field *gmem_outputs_matrix, const unsigned log_n, const unsigned num_ntts) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  // base_field f0{0x1,0x0}, f1{0x1,0x0};
  base_field acc0{0x0, 0x0}, acc1{0x0, 0x0};
  // base_field current[2] = {acc0, acc1};
  base_field alpha_pow[2] = {1, 0};
  // if (g_id == 0) { printf("num_alpha: %lld\n", base_field::to_u64(alpha_matrix[0])); }
  for (unsigned i = 0; i < num_ntts; i++) {
    base_field tmp = memory::load_cs(gmem_inputs_matrix + g_id + (i << log_n));
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
    void reverse_index_bits(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned log_n, const unsigned num_ntts,
                            const unsigned stride_between_input_arrays, const unsigned stride_between_output_arrays, const unsigned rate_bits,
                            const bool inverse) {
  static const unsigned array[] = {0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31};
  const unsigned tile = 32, log_tile = 5;
  const unsigned blocks_per_ntt = (stride_between_input_arrays + tile * tile - 1) / (tile * tile);
  // const unsigned ntt_id = (blockIdx.x + blocks_per_ntt - 1) / blocks_per_ntt;
  const unsigned ntt_id = blockIdx.x / blocks_per_ntt;
  const unsigned warp_id = blockIdx.x % blocks_per_ntt;
  const unsigned warp_id_r = reverse_bits(warp_id, log_n - 2 * log_tile);
  // if (threadIdx.x == 0) printf("warp: %d %d, ntt id: %d\n", warp_id, warp_id_r, ntt_id);
  const unsigned lane_id{threadIdx.x & 31};
  const unsigned lane_id_r = array[lane_id];
  const base_field *gmem_inputs_start = gmem_inputs_matrix + ntt_id * stride_between_input_arrays + warp_id * tile;
  base_field *gmem_outputs_start = gmem_outputs_matrix + ntt_id * stride_between_input_arrays + warp_id_r * tile;
  const unsigned stride_within = stride_between_input_arrays / tile;
  __shared__ base_field smem[tile][tile + 1];

#pragma unroll
  for (unsigned i = 0; i < tile; i++) {
    smem[i][lane_id] = memory::load_cs(gmem_inputs_start + lane_id + i * stride_within);
  }
  __syncwarp();
#pragma unroll
  for (unsigned i = 0; i < tile; i++) {
    memory::store_cs(gmem_outputs_start + lane_id + i * stride_within, smem[lane_id_r][array[i]]);
    // if (threadIdx.x == 0) printf("warp: %d %d\n", ntt_id * stride_between_input_arrays + warp_id_r * TILLE, lane_id + i * stride_within);
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void pad_coset(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned log_n, const unsigned num_ntts,
                   const unsigned stride_between_input_arrays, const unsigned stride_between_output_arrays, const unsigned rate_bits, const bool inverse) {
  const unsigned blocks_per_ntt = (stride_between_input_arrays + blockDim.x * TILLE - 1) / (blockDim.x * TILLE);
  const unsigned ntt_id = blockIdx.x / blocks_per_ntt;
  const unsigned stride_within = stride_between_input_arrays / TILLE;
  const unsigned g_id = blockIdx.x % blocks_per_ntt * blockDim.x + threadIdx.x;
  // printf("gid: %d, stride within: %d\n", g_id, stride_within);
  // if (threadIdx.x == 0) printf("gid: %d, ntt id: %d\n", g_id, ntt_id);
  if (g_id < stride_within) {
    const base_field *gmem_inputs_start = gmem_inputs_matrix + ntt_id * stride_between_input_arrays + g_id;
    base_field *gmem_outputs_start = gmem_outputs_matrix + ntt_id * stride_between_output_arrays + g_id;
    // if (threadIdx.x == 0) printf("warp: %d %d, ntt id: %d\n", warp_id, warp_id_r, ntt_id);
    base_field a{0x1, 0x0}, b{0x7, 0x0}, b_inv{0x6db6db6e, 0x24924924};
    unsigned c = 1, ggid = g_id;
    base_field b_used = inverse ? b_inv : b;
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
    for (unsigned i = 0; i < TILLE; i++) {
      base_field coseted = base_field::mul(memory::load_cs(gmem_inputs_start + i * stride_within), a);
      memory::store_cs(gmem_outputs_start + i * stride_within, coseted);
      a = base_field::mul(a, b_used);
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void transpose_gpu(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned log_n, const unsigned num_ntts,
                       const unsigned stride_between_input_arrays, const unsigned stride_between_output_arrays, const unsigned rate_bits, const bool inverse) {
  const unsigned blocks_col = 1 << (log_n - LOG_TILE);
  const unsigned row = (blockIdx.x / blocks_col) * TILLE;
  const unsigned col = (blockIdx.x % blocks_col) * TILLE;
  if (row + threadIdx.x >= num_ntts)
    return;
  // printf("thread %d remains\n", threadIdx.x);
  const base_field *gmem_inputs_start = gmem_inputs_matrix + ((row + threadIdx.x) << log_n) + col;
  base_field *gmem_outputs_start = gmem_outputs_matrix + col * num_ntts + row + threadIdx.x;
#pragma unroll
  for (unsigned i = 0; i < TILLE; i++) {
    base_field ld = memory::load_cs(gmem_inputs_start + i);
    // printf("read %lld\n", base_field::to_u64(ld));
    memory::store_cs(gmem_outputs_start + i * num_ntts, ld);
  }
}

// This kernel basically reverses the pattern of the b2n_initial_stages_warp kernel.
template <unsigned LOG_VALS_PER_THREAD>
DEVICE_FORCEINLINE void n2b_final_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                              const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                              const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                                              const unsigned coset_idx, const bool transpose) {
  constexpr unsigned VALS_PER_THREAD = 1 << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned LOG_VALS_PER_BLOCK = 5 + LOG_VALS_PER_THREAD + 2;
  constexpr unsigned VALS_PER_BLOCK = 1 << LOG_VALS_PER_BLOCK;

  __shared__ base_field smem[VALS_PER_BLOCK];

  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned gmem_offset = VALS_PER_BLOCK * blockIdx.x + VALS_PER_WARP * warp_id;
  const base_field *gmem_in = gmem_inputs_matrix + gmem_offset + NTTS_PER_BLOCK * stride_between_input_arrays * blockIdx.y;
  base_field *gmem_out = gmem_outputs_matrix + gmem_offset + NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  base_field vals[VALS_PER_THREAD];

  load_initial_twiddles_warp<VALS_PER_WARP, LOG_VALS_PER_THREAD>(twiddle_cache, lane_id, gmem_offset, inverse);

  const unsigned bound = std::min(NTTS_PER_BLOCK, num_ntts - NTTS_PER_BLOCK * blockIdx.y);
  for (unsigned ntt_idx = 0; ntt_idx < bound; ntt_idx++, gmem_in += stride_between_input_arrays, gmem_out += stride_between_output_arrays) {
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      vals[2 * i] = memory::load_cs(gmem_in + 64 * i + lane_id);
      vals[2 * i + 1] = memory::load_cs(gmem_in + 64 * i + lane_id + 32);
    }

    base_field *twiddles_this_stage = twiddle_cache + VALS_PER_WARP - 2;
    unsigned num_twiddles_this_stage = 1;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
#pragma unroll
      for (unsigned j = 0; j < (1 << i); j++) {
        const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
        const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
        const auto twiddle = twiddles_this_stage[j];
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++) {
          exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
        }
      }
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    unsigned lane_mask = 16;
    for (unsigned stage = 0, s = 5; stage < 6; stage++, s--) {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const auto twiddle = twiddles_this_stage[(32 * i + lane_id) >> s];
        exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
        if (stage < 5)
          shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask >>= 1;
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    if (inverse) {
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        vals[i] = base_field::mul(vals[i], inv_sizes[log_n]);
    }

    if (inverse && log_extension_degree) {
      __syncwarp();
      base_field tmp[VALS_PER_THREAD];
      base_field *scratch = twiddle_cache + VALS_PER_THREAD * lane_id;
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++) {
        tmp[i] = scratch[i];
        scratch[i] = vals[i];
      }
      apply_lde_factors<VALS_PER_THREAD, true>(scratch, gmem_offset, lane_id, log_n, log_extension_degree, coset_idx);
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const uint4 out{scratch[2 * i][0], scratch[2 * i][1], scratch[2 * i + 1][0], scratch[2 * i + 1][1]};
        // unsigned out_offset = gmem_offset + 64 * i + 2 * lane_id;
        // memory::store_cs(reinterpret_cast<uint4 *>(gmem_outputs_matrix + NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y +
        // reverse_bits<LOG_COUNT>(out_offset)), out);
        memory::store_cs(reinterpret_cast<uint4 *>(gmem_out + 64 * i + 2 * lane_id), out);
      }
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        scratch[i] = tmp[i];
      __syncwarp();
    } else {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
        memory::store_cs(reinterpret_cast<uint4 *>(gmem_out + 64 * i + 2 * lane_id), out);
        // base_field *gmem_out_base = gmem_outputs_matrix + NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y;
        // unsigned out_offset = gmem_offset + 64 * i + 2 * lane_id;
        // const uint2 out0{vals[2 * i][0], vals[2 * i][1]};
        // const uint2 out1{vals[2 * i + 1][0], vals[2 * i + 1][1]};
        // memory::store_cs(reinterpret_cast<uint2 *>(gmem_out_base + reverse_bits(out_offset, log_n)), out0);
        // memory::store_cs(reinterpret_cast<uint2 *>(gmem_out_base + reverse_bits(out_offset + 1, log_n)), out1);

        // memory::store_cs((gmem_out + 64 * i + 2 * lane_id), out);
        // memory::store_cs(reinterpret_cast<uint1 *>(gmem_out_base + reverse_bits(out_offset, log_n)), uint1{vals[2 * i][0]});
        // memory::store_cs(reinterpret_cast<uint1 *>(gmem_out_base + reverse_bits(out_offset + 1, log_n)), uint1{vals[2 * i][1]});
        // memory::store_cs(reinterpret_cast<uint1 *>(gmem_out_base + reverse_bits(out_offset + 2, log_n)), uint1{vals[2 * i + 1][0]});
        // memory::store_cs(reinterpret_cast<uint1 *>(gmem_out_base + reverse_bits(out_offset + 3, log_n)), uint1{vals[2 * i + 1][1]});
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void n2b_final_8_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                 const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                 const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                                 const unsigned coset_idx, const bool transpose) {
  n2b_final_stages_warp<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage, stages_this_launch,
                           log_n, inverse, num_ntts, log_extension_degree, coset_idx, transpose);
}

extern "C" __launch_bounds__(128, 8) __global__
    void n2b_final_7_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                 const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                 const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                                 const unsigned coset_idx, const bool transpose) {
  n2b_final_stages_warp<2>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage, stages_this_launch,
                           log_n, inverse, num_ntts, log_extension_degree, coset_idx, transpose);
}

// This kernel basically reverses the pattern of the b2n_initial_stages_block kernel.
template <unsigned LOG_VALS_PER_THREAD>
DEVICE_FORCEINLINE void n2b_final_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix,
                                               const unsigned stride_between_input_arrays, const unsigned stride_between_output_arrays,
                                               const unsigned start_stage, const unsigned stages_this_launch, const unsigned log_n, const bool inverse,
                                               const unsigned num_ntts, const unsigned log_extension_degree, const unsigned coset_idx, const bool transpose) {
  constexpr unsigned VALS_PER_THREAD = 1 << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned WARPS_PER_BLOCK = VALS_PER_WARP >> 4;
  constexpr unsigned VALS_PER_BLOCK = 32 * VALS_PER_THREAD * WARPS_PER_BLOCK;
  constexpr unsigned MAX_STAGES_THIS_LAUNCH = 2 * (LOG_VALS_PER_THREAD + 5) - 4;

  __shared__ base_field smem[VALS_PER_BLOCK];

  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned gmem_block_offset = VALS_PER_BLOCK * blockIdx.x;
  const unsigned gmem_offset = gmem_block_offset + VALS_PER_WARP * warp_id;
  // annoyingly scrambled, but should be coalesced overall
  const unsigned gmem_in_thread_offset = 16 * warp_id + VALS_PER_WARP * (lane_id >> 4) + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
  const base_field *gmem_in = gmem_inputs_matrix + gmem_block_offset + gmem_in_thread_offset + NTTS_PER_BLOCK * stride_between_input_arrays * blockIdx.y;
  base_field *gmem_out = gmem_outputs_matrix + gmem_offset + NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  base_field vals[VALS_PER_THREAD];

  const unsigned bound = std::min(NTTS_PER_BLOCK, num_ntts - NTTS_PER_BLOCK * blockIdx.y);
  for (unsigned ntt_idx = 0; ntt_idx < bound; ntt_idx++, gmem_in += stride_between_input_arrays, gmem_out += stride_between_output_arrays) {
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      vals[2 * i] = memory::load_cs(gmem_in + 4 * i * VALS_PER_WARP);
      vals[2 * i + 1] = memory::load_cs(gmem_in + (4 * i + 2) * VALS_PER_WARP);
    }

    const unsigned stages_to_skip = MAX_STAGES_THIS_LAUNCH - stages_this_launch;
    unsigned exchg_region_offset = blockIdx.x;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
      if (i >= stages_to_skip) {
#pragma unroll
        for (unsigned j = 0; j < (1 << i); j++) {
          const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
          const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
          const auto twiddle = get_twiddle(inverse, exchg_region_offset + j);
#pragma unroll
          for (unsigned k = 0; k < half_exchg_tile_sz; k++)
            exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
        }
      }
      exchg_region_offset <<= 1;
    }

    unsigned lane_mask = 16;
    unsigned halfwarp_id = lane_id >> 4;
    for (unsigned s = 0; s < 2; s++) {
      if ((s + LOG_VALS_PER_THREAD - 1) >= stages_to_skip) {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          // TODO: Handle these cooperatively?
          const auto twiddle = get_twiddle(inverse, exchg_region_offset + ((2 * i + halfwarp_id) >> (1 - s)));
          exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
          shfl_xor_bf(vals, i, lane_id, lane_mask);
        }
      } else {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++)
          shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask >>= 1;
      exchg_region_offset <<= 1;
    }

    __syncwarp(); // maybe unnecessary but can't hurt

    {
      base_field tmp[VALS_PER_THREAD];
      auto pair_addr = smem + 16 * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2 * (threadIdx.x & 7);
      if (ntt_idx > 0) {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          tmp[2 * i] = twiddle_cache[64 * i + lane_id];
          tmp[2 * i + 1] = twiddle_cache[64 * i + lane_id + 32];
        }

        __syncthreads();

#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++, pair_addr += 4 * VALS_PER_WARP) {
          uint4 *pair = reinterpret_cast<uint4 *>(pair_addr);
          const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
          *pair = out;
        }
      } else {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++, pair_addr += 4 * VALS_PER_WARP) {
          uint4 *pair = reinterpret_cast<uint4 *>(pair_addr);
          const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
          *pair = out;
        }
      }

      __syncthreads();

      if (ntt_idx > 0) {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          vals[2 * i] = twiddle_cache[64 * i + lane_id];
          vals[2 * i + 1] = twiddle_cache[64 * i + lane_id + 32];
          twiddle_cache[64 * i + lane_id] = tmp[2 * i];
          twiddle_cache[64 * i + lane_id + 32] = tmp[2 * i + 1];
        }

        __syncwarp();
      } else {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          vals[2 * i] = twiddle_cache[64 * i + lane_id];
          vals[2 * i + 1] = twiddle_cache[64 * i + lane_id + 32];
        }

        __syncwarp();

        load_initial_twiddles_warp<VALS_PER_WARP, LOG_VALS_PER_THREAD>(twiddle_cache, lane_id, gmem_offset, inverse);
      }
    }

    base_field *twiddles_this_stage = twiddle_cache + VALS_PER_WARP - 2;
    unsigned num_twiddles_this_stage = 1;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
#pragma unroll
      for (unsigned j = 0; j < (1 << i); j++) {
        const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
        const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
        const auto twiddle = twiddles_this_stage[j];
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++) {
          exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
        }
      }
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    lane_mask = 16;
    for (unsigned stage = 0, s = 5; stage < 6; stage++, s--) {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const auto twiddle = twiddles_this_stage[(32 * i + lane_id) >> s];
        exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
        if (stage < 5)
          shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask >>= 1;
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    if (inverse) {
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        vals[i] = base_field::mul(vals[i], inv_sizes[log_n]);
    }

    if (inverse && log_extension_degree) {
      __syncwarp();
      base_field tmp[VALS_PER_THREAD];
      base_field *scratch = twiddle_cache + VALS_PER_THREAD * lane_id;
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++) {
        tmp[i] = scratch[i];
        scratch[i] = vals[i];
      }
      apply_lde_factors<VALS_PER_THREAD, true>(scratch, gmem_offset, lane_id, log_n, log_extension_degree, coset_idx);
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const uint4 out{scratch[2 * i][0], scratch[2 * i][1], scratch[2 * i + 1][0], scratch[2 * i + 1][1]};
        memory::store_cs(reinterpret_cast<uint4 *>(gmem_out + 64 * i + 2 * lane_id), out);
        // unsigned out_offset = gmem_offset + 64 * i + 2 * lane_id;
        // memory::store_cs(reinterpret_cast<uint4 *>(gmem_outputs_matrix + NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y +
        // reverse_bits<LOG_COUNT>(out_offset)), out);
      }
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        scratch[i] = tmp[i];
      __syncwarp();
    } else {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
        memory::store_cs(reinterpret_cast<uint4 *>(gmem_out + 64 * i + 2 * lane_id), out);
        // base_field *gmem_out_base = gmem_outputs_matrix + NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y;
        // unsigned out_offset = gmem_offset + 64 * i + 2 * lane_id;
        // const uint2 out0{vals[2 * i][0], vals[2 * i][1]};
        // const uint2 out1{vals[2 * i + 1][0], vals[2 * i + 1][1]};
        // memory::store_cs(reinterpret_cast<uint2 *>(gmem_out_base + reverse_bits(out_offset, log_n)), out0);
        // memory::store_cs(reinterpret_cast<uint2 *>(gmem_out_base + reverse_bits(out_offset + 1, log_n)), out1);
      }
    }
  }
}

extern "C" __launch_bounds__(512, 2) __global__
    void n2b_final_9_to_12_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                        const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                        const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                                        const unsigned coset_idx, const bool transpose) {
  n2b_final_stages_block<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage, stages_this_launch,
                            log_n, inverse, num_ntts, log_extension_degree, coset_idx, transpose);
}

// This kernel basically reverses the pattern of the b2n_noninitial_stages_block kernel.
template <unsigned LOG_VALS_PER_THREAD>
DEVICE_FORCEINLINE void n2b_nonfinal_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix,
                                                  const unsigned stride_between_input_arrays, const unsigned stride_between_output_arrays,
                                                  const unsigned start_stage, const bool skip_last_stage, const unsigned log_n, const bool inverse,
                                                  const unsigned num_ntts, const unsigned log_extension_degree, const unsigned coset_idx) {
  constexpr unsigned VALS_PER_THREAD = 1 << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned TILES_PER_WARP = VALS_PER_WARP >> 4;
  constexpr unsigned WARPS_PER_BLOCK = VALS_PER_WARP >> 4;
  constexpr unsigned VALS_PER_BLOCK = VALS_PER_WARP * WARPS_PER_BLOCK;
  constexpr unsigned TILES_PER_BLOCK = VALS_PER_BLOCK >> 4;
  constexpr unsigned EXCHG_REGIONS_PER_BLOCK = TILES_PER_BLOCK >> 1;
  constexpr unsigned MAX_STAGES_THIS_LAUNCH = 2 * (LOG_VALS_PER_THREAD + 5) - 8;

  __shared__ base_field smem[VALS_PER_BLOCK];

  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned log_tile_stride = log_n - start_stage - MAX_STAGES_THIS_LAUNCH;
  const unsigned tile_stride = 1 << log_tile_stride;
  const unsigned log_blocks_per_region = log_tile_stride - 4; // tile size is always 16
  const unsigned block_bfly_region_size = TILES_PER_BLOCK * tile_stride;
  const unsigned block_bfly_region = blockIdx.x >> log_blocks_per_region;
  const unsigned block_bfly_region_start = block_bfly_region * block_bfly_region_size;
  const unsigned block_start_in_bfly_region = 16 * (blockIdx.x & ((1 << log_blocks_per_region) - 1));
  // annoyingly scrambled, but should be coalesced overall
  const unsigned gmem_in_thread_offset = tile_stride * warp_id + tile_stride * WARPS_PER_BLOCK * (lane_id >> 4) + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
  const unsigned gmem_in_offset = block_bfly_region_start + block_start_in_bfly_region + gmem_in_thread_offset;
  const base_field *gmem_in = gmem_inputs_matrix + gmem_in_offset + NTTS_PER_BLOCK * stride_between_input_arrays * blockIdx.y;
  base_field *gmem_out =
      gmem_outputs_matrix + block_bfly_region_start + block_start_in_bfly_region + NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  base_field vals[VALS_PER_THREAD];

  const unsigned bound = std::min(NTTS_PER_BLOCK, num_ntts - NTTS_PER_BLOCK * blockIdx.y);
  for (unsigned ntt_idx = 0; ntt_idx < bound; ntt_idx++, gmem_in += stride_between_input_arrays, gmem_out += stride_between_output_arrays) {
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      vals[2 * i] = memory::load_cs(gmem_in + 4 * i * tile_stride * WARPS_PER_BLOCK);
      vals[2 * i + 1] = memory::load_cs(gmem_in + (4 * i + 2) * tile_stride * WARPS_PER_BLOCK);
    }

    if ((start_stage == 0) && log_extension_degree && !inverse) {
      __syncwarp();
      base_field *scratch = twiddle_cache + lane_id;
      base_field tmp = *scratch;
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        scratch[32 * i] = vals[i];
#pragma unroll 1
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        base_field val0 = scratch[64 * i];
        base_field val1 = scratch[64 * i + 32];
        const unsigned idx0 = gmem_in_offset + 4 * i * tile_stride * WARPS_PER_BLOCK;
        const unsigned idx1 = gmem_in_offset + (4 * i + 2) * tile_stride * WARPS_PER_BLOCK;
        if (coset_idx) {
          const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
          const unsigned offset = coset_idx << shift;
          auto power_of_w0 = get_power_of_w(idx0 * offset, false);
          auto power_of_w1 = get_power_of_w(idx1 * offset, false);
          val0 = base_field::mul(val0, power_of_w0);
          val1 = base_field::mul(val1, power_of_w1);
        }
        auto power_of_g0 = get_power_of_g(idx0, false);
        auto power_of_g1 = get_power_of_g(idx1, false);
        scratch[64 * i] = base_field::mul(val0, power_of_g0);
        scratch[64 * i + 32] = base_field::mul(val1, power_of_g1);
      }
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        vals[i] = scratch[32 * i];
      *scratch = tmp;
      __syncwarp();
    }

    unsigned block_exchg_region_offset = block_bfly_region;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
#pragma unroll
      for (unsigned j = 0; j < (1 << i); j++) {
        const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
        const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
        const auto twiddle = get_twiddle(inverse, block_exchg_region_offset + j);
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++)
          exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
      }
      block_exchg_region_offset <<= 1;
    }

    unsigned lane_mask = 16;
    unsigned halfwarp_id = lane_id >> 4;
    for (unsigned s = 0; s < 2; s++) {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        // TODO: Handle these cooperatively?
        const auto twiddle = get_twiddle(inverse, block_exchg_region_offset + ((2 * i + halfwarp_id) >> (1 - s)));
        exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
        shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask >>= 1;
      block_exchg_region_offset <<= 1;
    }

    __syncwarp(); // maybe unnecessary but can't hurt

    // there are at most 31 per-warp twiddles, so we only need 1 temporary per thread to stash them
    base_field tmp;
    if ((ntt_idx > 0) || ((start_stage == 0) && log_extension_degree && !inverse)) {
      tmp = twiddle_cache[lane_id];
      __syncthreads();
    }

    auto smem_pair_addr = smem + 16 * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2 * (threadIdx.x & 7);
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++, smem_pair_addr += 4 * VALS_PER_WARP) {
      uint4 *pair = reinterpret_cast<uint4 *>(smem_pair_addr);
      const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
      *pair = out;
    }

    __syncthreads();

    // annoyingly scrambled but should be bank-conflict-free
    const unsigned smem_thread_offset = 16 * (lane_id >> 4) + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      vals[2 * i] = twiddle_cache[64 * i + smem_thread_offset];
      vals[2 * i + 1] = twiddle_cache[64 * i + smem_thread_offset + 32];
    }

    __syncwarp();

    if (ntt_idx > 0) {
      twiddle_cache[lane_id] = tmp;
      __syncwarp();
    } else {
      load_noninitial_twiddles_warp<LOG_VALS_PER_THREAD>(twiddle_cache, lane_id, warp_id, block_bfly_region * EXCHG_REGIONS_PER_BLOCK, inverse);
    }

    base_field *twiddles_this_stage = twiddle_cache + 2 * VALS_PER_THREAD - 2;
    unsigned num_twiddles_this_stage = 1;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
#pragma unroll
      for (unsigned j = 0; j < (1 << i); j++) {
        const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
        const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
        const auto twiddle = twiddles_this_stage[j];
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++) {
          exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
        }
      }
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    lane_mask = 16;
    for (unsigned s = 0; s < 2; s++) {
      if (!skip_last_stage || s < 1) {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          // TODO: Handle these cooperatively?
          const auto twiddle = twiddles_this_stage[(2 * i + halfwarp_id) >> (1 - s)];
          exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
          shfl_xor_bf(vals, i, lane_id, lane_mask);
        }
        lane_mask >>= 1;
        num_twiddles_this_stage <<= 1;
        twiddles_this_stage -= num_twiddles_this_stage;
      }
    }

    if (skip_last_stage) {
      auto val0_addr = gmem_out + TILES_PER_WARP * tile_stride * warp_id + 2 * tile_stride * (lane_id >> 4) + 2 * (threadIdx.x & 7) + (lane_id >> 3 & 1);
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        memory::store_cs(val0_addr, vals[2 * i]);
        memory::store_cs(val0_addr + tile_stride, vals[2 * i + 1]);
        val0_addr += 4 * tile_stride;
      }
    } else {
      auto pair_addr = gmem_out + TILES_PER_WARP * tile_stride * warp_id + tile_stride * (lane_id >> 3) + 2 * (threadIdx.x & 7);
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
        memory::store_cs(reinterpret_cast<uint4 *>(pair_addr), out);
        pair_addr += 4 * tile_stride;
      }
    }
  }
}

extern "C" __launch_bounds__(512, 2) __global__
    void n2b_nonfinal_7_or_8_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                          const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                          const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                                          const unsigned coset_idx, const bool transpose) {
  n2b_nonfinal_stages_block<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                               stages_this_launch == 7, log_n, inverse, num_ntts, log_extension_degree, coset_idx);
}

// Simple, non-optimized kernel used for log_n < 16, to unblock debugging small proofs.
extern "C" __launch_bounds__(512, 2) __global__
    void n2b_1_stage(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                     const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch, const unsigned log_n,
                     const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree, const unsigned coset_idx) {
  const unsigned ntt_idx = blockIdx.x / blocks_per_ntt;
  const unsigned bid_in_ntt = blockIdx.x - ntt_idx * blocks_per_ntt;
  const unsigned tid_in_ntt = threadIdx.x + bid_in_ntt * blockDim.x;
  if (tid_in_ntt >= (1 << (log_n - 1)))
    return;
  const unsigned log_exchg_region_sz = log_n - start_stage;
  const unsigned exchg_region = tid_in_ntt >> (log_exchg_region_sz - 1);
  const unsigned tid_in_exchg_region = tid_in_ntt - (exchg_region << (log_exchg_region_sz - 1));
  const unsigned exchg_stride = 1 << (log_exchg_region_sz - 1);
  const unsigned a_idx = tid_in_exchg_region + exchg_region * (1 << log_exchg_region_sz);
  const unsigned b_idx = a_idx + exchg_stride;
  const base_field *gmem_input = gmem_inputs_matrix + ntt_idx * stride_between_input_arrays;
  base_field *gmem_output = gmem_outputs_matrix + ntt_idx * stride_between_output_arrays;

  const auto twiddle = get_twiddle(inverse, exchg_region);
  auto a = memory::load_cs(gmem_input + a_idx);
  auto b = memory::load_cs(gmem_input + b_idx);

  if ((start_stage == 0) && log_extension_degree && !inverse) {
    if (coset_idx) {
      const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
      const unsigned offset = coset_idx << shift;
      a = base_field::mul(a, get_power_of_w(a_idx * offset, false));
      b = base_field::mul(b, get_power_of_w(b_idx * offset, false));
    }
    a = base_field::mul(a, get_power_of_g(a_idx, false));
    b = base_field::mul(b, get_power_of_g(b_idx, false));
  }

  exchg_dit(a, b, twiddle);

  if (inverse && (start_stage + stages_this_launch == log_n)) {
    a = base_field::mul(a, inv_sizes[log_n]);
    b = base_field::mul(b, inv_sizes[log_n]);
    if (log_extension_degree) {
      const unsigned a_idx_brev = __brev(a_idx) >> (32 - log_n);
      const unsigned b_idx_brev = __brev(b_idx) >> (32 - log_n);
      if (coset_idx) {
        const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
        const unsigned offset = coset_idx << shift;
        a = base_field::mul(a, get_power_of_w(a_idx_brev * offset, true));
        b = base_field::mul(b, get_power_of_w(b_idx_brev * offset, true));
      }
      a = base_field::mul(a, get_power_of_g(a_idx_brev, true));
      b = base_field::mul(b, get_power_of_g(b_idx_brev, true));
    }
  }

  memory::store_cs(gmem_output + a_idx, a);
  memory::store_cs(gmem_output + b_idx, b);
}

// const unsigned SPONGE_RATE = 8;
// const unsigned SPONGE_CAPACITY = 4;
// const unsigned SPONGE_WIDTH = SPONGE_RATE + SPONGE_CAPACITY;
// const unsigned HALF_N_FULL_ROUNDS = 4;
// const unsigned N_PARTIAL_ROUNDS = 22;

// extern "C" __global__ void poseidon_generator_kernel(
//     base_field *witness,
//     const unsigned *representative_map,
//     const size_t *u_params,
//     const base_field *f_params,
//     const unsigned u_start,
//     const unsigned f_start,
//     const unsigned length,
//     const unsigned num
// ) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < length) {
//         for (int i = 0; i < num; i++) {
//         poseidon::poseidon_state state{};
//         for (int i = 0; i < SPONGE_WIDTH; i++) {
//             state[i] = base_field::into<3>(witness[representative_map[u_params[u_start + tid] + i]]);
//         }

//         base_field swap_value = witness[representative_map[u_params[u_start + tid] + 2 * SPONGE_WIDTH]];

//         for (int i = 0; i < 4; i++) {
//             base_field delta_i = base_field::mul(swap_value, base_field::sub(base_field::field3_to_field2(state[i + 4]), base_field::field3_to_field2(state[i])));
//             witness[representative_map[u_params[u_start + tid] + (2 * SPONGE_WIDTH + 1 + i)]] = delta_i;
//         }

//         if (base_field::to_u64(swap_value) == 1) {
//             for (int i = 0; i < 4; i++) {
//             field<3> temp = state[i];
//             state[i] = state[i + 4];
//             state[i + 4] = temp;
//             }
//         }

//         int round_ctr = 0;

//         for (int r = 0; r < HALF_N_FULL_ROUNDS; r++) {
//             poseidon::apply_round_constants(state, round_ctr);
//             if (r != 0) {
//                 for (int i = 0; i < SPONGE_WIDTH; i++) {
//                     witness[representative_map[u_params[u_start + tid] + 2 * SPONGE_WIDTH + 5 + (r - 1) * SPONGE_WIDTH + i]] = base_field::field3_to_field2(state[i]);
//                 }
//             }
//             poseidon::apply_non_linearity(state);
//             poseidon::apply_mds_matrix_new(state);
//             round_ctr++;
//         }
//         poseidon::partial_rounds_init(state);

//         for (int r = 0; r < N_PARTIAL_ROUNDS; r++) {
//             witness[representative_map[u_params[u_start + tid] + 2 * SPONGE_WIDTH + 5 + (HALF_N_FULL_ROUNDS - 1) * SPONGE_WIDTH + r]] = base_field::field3_to_field2(state[0]);
//             poseidon::partial_round_optimized(state, r + HALF_N_FULL_ROUNDS);
//         }

//         round_ctr += N_PARTIAL_ROUNDS;

//         for (int r = 0; r < HALF_N_FULL_ROUNDS; r++) {
//             poseidon::apply_round_constants(state, round_ctr);
//             for (int i = 0; i < SPONGE_WIDTH; i++) {
//             witness[representative_map[u_params[u_start + tid] + 2 * SPONGE_WIDTH + 1 + 4 + SPONGE_WIDTH * (HALF_N_FULL_ROUNDS - 1) + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH + i]] =
//                 base_field::field3_to_field2(state[i]);
//             }
//             poseidon::apply_non_linearity(state);
//             poseidon::apply_mds_matrix_new(state);
//             round_ctr++;
//         }
//         for (int i = 0; i < SPONGE_WIDTH; i++) {
//             witness[representative_map[u_params[u_start + tid] + SPONGE_WIDTH + i]] = base_field::field3_to_field2(state[i]);
//         }
//         tid++;
//         }
//     }
// }

extern "C" __launch_bounds__(128, 8) __global__
    void u32_add_many_gate(
        const base_field *wires_ptr,
        const base_field *constants_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const unsigned degree_bits,
        const unsigned num_wires,
        const unsigned num_const_sigmas,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned row_step = nrows / TILE_Q;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            unsigned cons_offset = 0;
            const unsigned lde_started = g_id + t * row_step;
            for (unsigned i = 0; i < num_ops; i++) {
                base_field carry = memory::load_cs(wires_ptr + lde_started + ((num_addends + 3) * i + num_addends) * nrows);
                base_field computed_output = carry;
                for (unsigned j = 0; j < num_addends; j++) {
                    base_field addend = memory::load_cs(wires_ptr + lde_started + ((num_addends + 3) * i + j) * nrows);
                    computed_output = base_field::add(computed_output, addend);
                }
                base_field output_result = memory::load_cs(wires_ptr + lde_started + ((num_addends + 3) * i + num_addends + 1) * nrows);
                base_field output_carry = memory::load_cs(wires_ptr + lde_started + ((num_addends + 3) * i + num_addends + 2) * nrows);

                base_field base = base_field::from_u64(1ULL << 32);
                base_field combined_output = base_field::add(base_field::mul(output_carry, base), output_result);

                base_field constraint = base_field::sub(combined_output, computed_output);
                memory::store_cs(output + g_id + t * row_step + cons_offset, constraint);
                cons_offset += nrows;

                base_field combined_result_limbs = base_field::zero();
                base_field combined_carry_limbs = base_field::zero();

                base = base_field::from_u64(1ULL << 2); // 2 is the self.limb_bits()
                for (int j = 18 - 1; j >= 0; j--) { // 18 is the self.num_limbs()
                    base_field this_limb = memory::load_cs(wires_ptr + lde_started + ((num_addends + 3) * num_ops + 18 * i + j) * nrows);
                    uint64_t max_limb = 1ULL << 2;
                    base_field product = base_field::one();
                    for (unsigned k = 0; k < max_limb; k++) {
                        product = base_field::mul(product, base_field::sub(this_limb, base_field::from_u64(k)));
                    }
                    memory::store_cs(output + g_id + t * row_step + cons_offset, product);
                    cons_offset += nrows;

                    if (j < 16) { // 16 is the self.num_result_limbs()
                        combined_result_limbs = base_field::add(base_field::mul(base, combined_result_limbs), this_limb);
                    } else {
                        combined_carry_limbs = base_field::add(base_field::mul(base, combined_carry_limbs), this_limb);
                    }
                }
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(combined_result_limbs, output_result));
                cons_offset += nrows;
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(combined_carry_limbs, output_carry));
                cons_offset += nrows;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void u32_arithmetic_gate(
        const base_field *wires_ptr,
        const base_field *constants_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const unsigned degree_bits,
        const unsigned num_wires,
        const unsigned num_const_sigmas,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned row_step = nrows / TILE_Q;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            unsigned cons_offset = 0;
            const unsigned lde_started = g_id + t * row_step;
            for (unsigned i = 0; i < num_ops; i++) {
                base_field multiplicand_0 = memory::load_cs(wires_ptr + lde_started + (6 * i) * nrows);
                base_field multiplicand_1 = memory::load_cs(wires_ptr + lde_started + (6 * i + 1) * nrows);
                base_field addend = memory::load_cs(wires_ptr + lde_started + (6 * i + 2) * nrows);

                base_field computed_output = base_field::add(base_field::mul(multiplicand_0, multiplicand_1), addend);

                base_field output_low = memory::load_cs(wires_ptr + lde_started + (6 * i + 3) * nrows);
                base_field output_high = memory::load_cs(wires_ptr + lde_started + (6 * i + 4) * nrows);
                base_field inverse = memory::load_cs(wires_ptr + lde_started + (6 * i + 5) * nrows);

                base_field base = base_field::from_u64(1ULL << 32);
                base_field one = base_field::one();
                base_field u32_max = base_field::from_u64(UINT_MAX);

                base_field diff = base_field::sub(u32_max, output_high);
                base_field hi_not_max = base_field::sub(base_field::mul(inverse, diff), one);
                base_field hi_not_max_or_lo_zero = base_field::mul(hi_not_max, output_low);

                memory::store_cs(output + g_id + t * row_step + cons_offset, hi_not_max_or_lo_zero);
                cons_offset += nrows;

                base_field combined_output = base_field::add(base_field::mul(output_high, base), output_low);

                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(combined_output, computed_output));
                cons_offset += nrows;

                base_field combined_low_limbs = base_field::zero();
                base_field combined_high_limbs = base_field::zero();

                size_t midpoint = 16;
                base = base_field::from_u64(1ULL << 2); // 2 is the self.limb_bits()
                for(int j = 32 - 1; j >= 0; j--) { // 32 is the self.num_limbs()
                    base_field this_limb = memory::load_cs(wires_ptr + lde_started + (6 * num_ops + 32 * i + j) * nrows);
                    uint64_t max_limb = 1ULL << 2; // 2 is the self.limb_bits()
                    base_field product = base_field::one();
                    for (unsigned k = 0; k < max_limb; k++) {
                        product = base_field::mul(product, base_field::sub(this_limb, base_field::from_u64(k)));
                    }
                    memory::store_cs(output + g_id + t * row_step + cons_offset, product);
                    cons_offset += nrows;

                    if (j < midpoint) {
                        combined_low_limbs = base_field::add(base_field::mul(base, combined_low_limbs), this_limb);
                    } else {
                        combined_high_limbs = base_field::add(base_field::mul(base, combined_high_limbs), this_limb);
                    }
                }
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(combined_low_limbs, output_low));
                cons_offset += nrows;
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(combined_high_limbs, output_high));
                cons_offset += nrows;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void comparison_gate(
        const base_field *wires_ptr,
        const base_field *constants_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const unsigned degree_bits,
        const unsigned num_wires,
        const unsigned num_const_sigmas,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned row_step = nrows / TILE_Q;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            unsigned cons_offset = 0;
            const unsigned lde_started = g_id + t * row_step;

            base_field first_input = memory::load_cs(wires_ptr + lde_started);
            base_field second_input = memory::load_cs(wires_ptr + lde_started + nrows);

            uint64_t chunk_bits = (num_bits + num_chunks - 1) / num_chunks;
            uint64_t chunk_size = 1ULL << chunk_bits;

            base_field first_chunks_combined = base_field::zero();
            base_field second_chunks_combined = base_field::zero();
            for(int i = num_chunks - 1; i >= 0; i--) {
                base_field first_chunk = memory::load_cs(wires_ptr + lde_started + (4 + i) * nrows);
                base_field second_chunk = memory::load_cs(wires_ptr + lde_started + (4 + num_chunks + i) * nrows);

                first_chunks_combined = base_field::add(base_field::mul(first_chunks_combined, base_field::from_u64(chunk_size)), first_chunk);
                second_chunks_combined = base_field::add(base_field::mul(second_chunks_combined, base_field::from_u64(chunk_size)), second_chunk);
            }

            memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(first_chunks_combined, first_input));
            cons_offset += nrows;
            memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(second_chunks_combined, second_input));
            cons_offset += nrows;

            base_field most_significant_diff_so_far = base_field::zero();

            for(unsigned i = 0; i < num_chunks; i++) {
                base_field first_chunk = memory::load_cs(wires_ptr + lde_started + (4 + i) * nrows);
                base_field second_chunk = memory::load_cs(wires_ptr + lde_started + (4 + num_chunks + i) * nrows);

                base_field first_product = base_field::one();
                base_field second_product = base_field::one();
                for(unsigned x = 0; x < chunk_size; x++) {
                    first_product = base_field::mul(first_product, base_field::sub(first_chunk, base_field::from_u64(x)));
                    second_product = base_field::mul(second_product, base_field::sub(second_chunk, base_field::from_u64(x)));
                }
                memory::store_cs(output + g_id + t * row_step + cons_offset, first_product);
                cons_offset += nrows;
                memory::store_cs(output + g_id + t * row_step + cons_offset, second_product);
                cons_offset += nrows;

                base_field difference = base_field::sub(second_chunk, first_chunk);
                base_field equality_dummy = memory::load_cs(wires_ptr + lde_started + (4 + 2 * num_chunks + i) * nrows);
                base_field chunks_equal = memory::load_cs(wires_ptr + lde_started + (4 + 3 * num_chunks + i) * nrows);

                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(base_field::mul(difference, equality_dummy), base_field::sub(base_field::one(), chunks_equal)));
                cons_offset += nrows;
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::mul(chunks_equal, difference));
                cons_offset += nrows;

                base_field intermediate_value = memory::load_cs(wires_ptr + lde_started + (4 + 4 * num_chunks + i) * nrows);
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(intermediate_value, base_field::mul(chunks_equal, most_significant_diff_so_far)));
                cons_offset += nrows;
                most_significant_diff_so_far = base_field::add(intermediate_value, base_field::mul(base_field::sub(base_field::one(), chunks_equal), difference));
            }

            base_field most_significant_diff = memory::load_cs(wires_ptr + lde_started + 3 * nrows);
            memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(most_significant_diff, most_significant_diff_so_far));
            cons_offset += nrows;

            for(unsigned i = 0; i < chunk_bits + 1; i++){
                base_field bit = memory::load_cs(wires_ptr + lde_started + (4 + 5 * num_chunks + i) * nrows);
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::mul(bit, base_field::sub(base_field::one(), bit)));
                cons_offset += nrows;
            }

            base_field bits_combined = base_field::zero();
            for(int i = chunk_bits; i >= 0; i--) {
                base_field bit = memory::load_cs(wires_ptr + lde_started + (4 + 5 * num_chunks + i) * nrows);
                bits_combined = base_field::add(base_field::mul(bits_combined, base_field::from_u64(2)), bit);
            }
            base_field two_n = base_field::from_u64(1ULL << chunk_bits);
            memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(base_field::add(two_n, most_significant_diff), bits_combined));
            cons_offset += nrows;

            base_field result_bool = memory::load_cs(wires_ptr + lde_started + 2 * nrows);
            base_field bit = memory::load_cs(wires_ptr + lde_started + (4 + 5 * num_chunks + chunk_bits) * nrows);
            memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(result_bool, bit));
            cons_offset += nrows;
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void u32_interleave_gate(
        const base_field *wires_ptr,
        const base_field *constants_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const unsigned degree_bits,
        const unsigned num_wires,
        const unsigned num_const_sigmas,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned row_step = nrows / TILE_Q;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            unsigned cons_offset = 0;
            const unsigned lde_started = g_id + t * row_step;

            for(unsigned i = 0; i < num_ops; i++) {
                base_field x = memory::load_cs(wires_ptr + lde_started + (2 * i) * nrows);
                base_field x_interleaved = memory::load_cs(wires_ptr + lde_started + (2 * i + 1) * nrows);

                base_field computed_x = base_field::zero();
                base_field computed_x_interleaved = base_field::zero();
                for(unsigned j = 0; j < 32; j++){
                    base_field bit = memory::load_cs(wires_ptr + lde_started + (2 * num_ops + 32 * i + j) * nrows);
                    computed_x = base_field::add(base_field::mul(computed_x, base_field::from_u64(2)), bit);
                    computed_x_interleaved = base_field::add(base_field::mul(computed_x_interleaved, base_field::from_u64(4)), bit);
                }
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(computed_x, x));
                cons_offset += nrows;
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(computed_x_interleaved, x_interleaved));
                cons_offset += nrows;

                for(unsigned j = 0; j < 32; j++){
                    base_field bit = memory::load_cs(wires_ptr + lde_started + (2 * num_ops + 32 * i + j) * nrows);
                    base_field product = base_field::one();
                    for(unsigned k = 0; k < 2; k++){
                        product = base_field::mul(product, base_field::sub(bit, base_field::from_u64(k)));
                    }
                    memory::store_cs(output + g_id + t * row_step + cons_offset, product);
                    cons_offset += nrows;
                }
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void u32_range_check_gate(
        const base_field *wires_ptr,
        const base_field *constants_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const unsigned degree_bits,
        const unsigned num_wires,
        const unsigned num_const_sigmas,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned row_step = nrows / TILE_Q;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            unsigned cons_offset = 0;
            const unsigned lde_started = g_id + t * row_step;
            base_field base = base_field::from_u64(1ULL << 2);
            for(unsigned i = 0; i < num_input_limbs; i++) {
                base_field input_limb = memory::load_cs(wires_ptr + lde_started + i * nrows);

                base_field computed_sum = base_field::zero();
                for(int j = 15; j >= 0; j--) {
                    base_field aux_limb = memory::load_cs(wires_ptr + lde_started + (num_input_limbs + 16 * i + j) * nrows);
                    computed_sum = base_field::add(base_field::mul(base, computed_sum), aux_limb);
                }
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(computed_sum, input_limb));
                cons_offset += nrows;

                for(unsigned j = 0; j < 16; j++) {
                    base_field aux_limb = memory::load_cs(wires_ptr + lde_started + (num_input_limbs + 16 * i + j) * nrows);
                    base_field product = base_field::one();
                    for(unsigned k = 0; k < 4; k++) { // 1 << 2
                        product = base_field::mul(product, base_field::sub(aux_limb, base_field::from_u64(k)));
                    }
                    memory::store_cs(output + g_id + t * row_step + cons_offset, product);
                    cons_offset += nrows;
                }
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void u32_subtraction_gate(
        const base_field *wires_ptr,
        const base_field *constants_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const unsigned degree_bits,
        const unsigned num_wires,
        const unsigned num_const_sigmas,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned row_step = nrows / TILE_Q;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            unsigned cons_offset = 0;
            const unsigned lde_started = g_id + t * row_step;

            for(unsigned i = 0; i < num_ops; i++) {
                base_field input_x = memory::load_cs(wires_ptr + lde_started + (5 * i) * nrows);
                base_field input_y = memory::load_cs(wires_ptr + lde_started + (5 * i + 1) * nrows);
                base_field input_borrow = memory::load_cs(wires_ptr + lde_started + (5 * i + 2) * nrows);

                base_field result_initial = base_field::sub(base_field::sub(input_x, input_y), input_borrow);
                base_field base = base_field::from_u64(1ULL << 32);

                base_field output_result = memory::load_cs(wires_ptr + lde_started + (5 * i + 3) * nrows);
                base_field output_borrow = memory::load_cs(wires_ptr + lde_started + (5 * i + 4) * nrows);

                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(output_result, base_field::add(result_initial, base_field::mul(output_borrow, base))));
                cons_offset += nrows;

                base_field combined_limbs = base_field::zero();
                base_field limb_base = base_field::from_u64(1ULL << 2);
                for(int j = 15; j >= 0; j--) {
                    base_field this_limb = memory::load_cs(wires_ptr + lde_started + (5 * num_ops + 16 * i + j) * nrows);
                    uint64_t max_limb = 1ULL << 2;
                    base_field product = base_field::one();
                    for(unsigned k = 0; k < max_limb; k++) {
                        product = base_field::mul(product, base_field::sub(this_limb, base_field::from_u64(k)));
                    }
                    memory::store_cs(output + g_id + t * row_step + cons_offset, product);
                    cons_offset += nrows;

                    combined_limbs = base_field::add(base_field::mul(limb_base, combined_limbs), this_limb);
                }
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(combined_limbs, output_result));
                cons_offset += nrows;

                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::mul(output_borrow, base_field::sub(base_field::one(), output_borrow)));
                cons_offset += nrows;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void uninterleave_to_u32_gate(
        const base_field *wires_ptr,
        const base_field *constants_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const unsigned degree_bits,
        const unsigned num_wires,
        const unsigned num_const_sigmas,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned row_step = nrows / TILE_Q;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            unsigned cons_offset = 0;
            const unsigned lde_started = g_id + t * row_step;

            for(unsigned i = 0; i < num_ops; i++) {
                base_field x_interleaved = memory::load_cs(wires_ptr + lde_started + (3 * i) * nrows);

                base_field computed_x_interleaved = base_field::zero();
                for(unsigned j = 0; j < 64; j++) {
                    base_field bit = memory::load_cs(wires_ptr + lde_started + (3 * num_ops + 64 * i + j) * nrows);
                    computed_x_interleaved = base_field::add(base_field::mul(computed_x_interleaved, base_field::from_u64(2)), bit);
                }
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(computed_x_interleaved, x_interleaved));
                cons_offset += nrows;

                base_field x_evens = memory::load_cs(wires_ptr + lde_started + (3 * i + 1) * nrows);
                base_field x_odds = memory::load_cs(wires_ptr + lde_started + (3 * i + 2) * nrows);

                base_field computed_x_evens = base_field::zero();
                base_field computed_x_odds = base_field::zero();

                for(unsigned j = 0; j < 32; j++) {
                    base_field jth_even = memory::load_cs(wires_ptr + lde_started + (3 * num_ops + 64 * i + 2 * j) * nrows);
                    base_field jth_odd = memory::load_cs(wires_ptr + lde_started + (3 * num_ops + 64 * i + 2 * j + 1) * nrows);

                    base_field coeff = base_field::from_u64(1ULL << (32 - j - 1));
                    computed_x_evens = base_field::add(computed_x_evens, base_field::mul(jth_even, coeff));
                    computed_x_odds = base_field::add(computed_x_odds, base_field::mul(jth_odd, coeff));
                }

                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(computed_x_evens, x_evens));
                cons_offset += nrows;
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(computed_x_odds, x_odds));
                cons_offset += nrows;

                for(unsigned j = 0; j < 64; j++) {
                    base_field bit = memory::load_cs(wires_ptr + lde_started + (3 * num_ops + 64 * i + j) * nrows);
                    base_field product = base_field::one();
                    for(unsigned k = 0; k < 2; k++) {
                        product = base_field::mul(product, base_field::sub(bit, base_field::from_u64(k)));
                    }
                    memory::store_cs(output + g_id + t * row_step + cons_offset, product);
                    cons_offset += nrows;
                }
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void base_sum_gate(
        const base_field *wires_ptr,
        const base_field *constants_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const unsigned degree_bits,
        const unsigned num_wires,
        const unsigned num_const_sigmas,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned row_step = nrows / TILE_Q;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            unsigned cons_offset = 0;
            const unsigned lde_started = g_id + t * row_step;

            base_field sum = memory::load_cs(wires_ptr + lde_started);
            base_field computed_sum = base_field::zero();
            for(int i = num_limbs; i > 0; i--) {
                base_field limb = memory::load_cs(wires_ptr + lde_started + i * nrows);
                computed_sum = base_field::add(base_field::mul(computed_sum, base_field::from_u64(B)), limb);
            }
            memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(computed_sum, sum));
            cons_offset += nrows;

            for(unsigned i = 0; i < num_limbs; i++) {
                base_field limb = memory::load_cs(wires_ptr + lde_started + (1 + i) * nrows);
                base_field product = base_field::one();
                for(unsigned j = 0; j < B; j++) {
                    product = base_field::mul(product, base_field::sub(limb, base_field::from_u64(j)));
                }
                memory::store_cs(output + g_id + t * row_step + cons_offset, product);
                cons_offset += nrows;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void random_access_gate(
        const base_field *wires_ptr,
        const base_field *constants_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const unsigned degree_bits,
        const unsigned num_wires,
        const unsigned num_const_sigmas,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned num_ops,
        const unsigned num_consts,
        const unsigned num_limbs,
        const unsigned num_addends,
        const unsigned num_bits,
        const unsigned num_chunks,
        const unsigned num_input_limbs,
        const unsigned B,
        const unsigned bits,
        const unsigned num_copies,
        const unsigned num_extra_constants
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned row_step = nrows / TILE_Q;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            unsigned cons_offset = 0;
            const unsigned lde_started = g_id + t * row_step;

            uint64_t vec_size = 1ULL << bits;
            for(unsigned copy = 0; copy < num_copies; copy++){
                base_field access_index = memory::load_cs(wires_ptr + lde_started + ((2 + vec_size) * copy) * nrows);
                base_field claimed_element = memory::load_cs(wires_ptr + lde_started + ((2 + vec_size) * copy + 1) * nrows);

                for(unsigned i = 0; i < bits; i++){
                    base_field bit = memory::load_cs(wires_ptr + lde_started + ((2 + vec_size) * num_copies + num_extra_constants + bits * copy + i) * nrows);
                    memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::mul(bit, base_field::sub(bit, base_field::one())));
                    cons_offset += nrows;
                }

                base_field reconstructed_index = base_field::zero();
                for(int i = bits - 1; i >= 0; i--){
                    base_field bit = memory::load_cs(wires_ptr + lde_started + ((2 + vec_size) * num_copies + num_extra_constants + bits * copy + i) * nrows);
                    reconstructed_index = base_field::add(base_field::add(reconstructed_index, reconstructed_index), bit);
                }
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(reconstructed_index, access_index));
                cons_offset += nrows;

                // base_field* list_items = (base_field*)malloc(vec_size * sizeof(base_field));
                base_field list_items[16]; // TODO: variable size
                for(unsigned i = 0; i < vec_size; i++){
                    list_items[i] = memory::load_cs(wires_ptr + lde_started + ((2 + vec_size) * copy + 2 + i) * nrows);
                }
                size_t new_size = vec_size;
                for(unsigned i = 0; i < bits; i++){
                    new_size = new_size / 2;
                    // base_field* new_list_items = (base_field*)malloc(new_size * sizeof(base_field));
                    base_field bit = memory::load_cs(wires_ptr + lde_started + ((2 + vec_size) * num_copies + num_extra_constants + bits * copy + i) * nrows);
                    for(unsigned j = 0; j < new_size; j++){
                        base_field x = list_items[2 * j];
                        base_field y = list_items[2 * j + 1];
                        // new_list_items[j] = base_field::add(x, base_field::mul(bit, base_field::sub(y, x)));
                        list_items[j] = base_field::add(x, base_field::mul(bit, base_field::sub(y, x)));
                    }
                    // free(list_items);
                    // list_items = new_list_items;
                }

                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(list_items[0], claimed_element));
                cons_offset += nrows;
            }
            for(unsigned i = 0; i < num_extra_constants; i++){
                base_field constant = memory::load_cs(constants_ptr + lde_started + (i + num_selectors) * nrows);
                base_field claimed_constant = memory::load_cs(wires_ptr + lde_started + ((2 + vec_size) * num_copies + i) * nrows);
                memory::store_cs(output + g_id + t * row_step + cons_offset, base_field::sub(constant, claimed_constant));
                cons_offset += nrows;
            }
        }
    }
}