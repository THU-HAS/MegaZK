#pragma once

#include "goldilocks_extension.cuh"

using namespace goldilocks;

#include "poseidon_single_thread.cuh"

typedef field<3> poseidon_state[12];
#define TILE_Q 16
#define N_CLGS 2

DEVICE_FORCEINLINE void mul_pow_add(base_field in, base_field *acc, const base_field *alphas, base_field *alpha_pows) {
    #pragma unroll
        for (unsigned i = 0; i < N_CLGS; i++) { // TODO: check if constant is faster
            acc[i] = acc[i] + alpha_pows[i] * in;
            alpha_pows[i] = alpha_pows[i] * alphas[i];
        }
    }

static DEVICE_FORCEINLINE void display(poseidon_state &state) {
#pragma unroll 1
    for (unsigned i = 0; i < 12; i++) {
    field<2> value = field<3>::field3_to_field2(state[i]);
    printf("(0x%x%x), ", value[1], value[0]);
    }
    printf("\n");
}

DEVICE_FORCEINLINE base_field compute_filter(const unsigned index, const base_field *constants_ptr, const unsigned row,
        const unsigned selector_index, const unsigned start, const unsigned end, const unsigned nrows,
        const bool verbose=false) {
    base_field s = memory::load_cs(constants_ptr + index + selector_index * nrows);
    base_field filter{0x1, 0x0};
    const base_field UNUSED_SELECTOR{0xffffffff, 0x0};
    if (verbose) printf("s: %llu, start: %d, end: %d, row: %d\n", s, start, end, row);
    for (unsigned i = start; i < end; i++) {
      if (i != row) {
          filter = base_field::mul(filter, base_field::sub(base_field::from_u64(i), s));
      }
      if (verbose) printf("i: %d, filter: %llu\n", i, filter);
    }
    if (true) { // many selectors
      filter = base_field::mul(filter, base_field::sub(UNUSED_SELECTOR, s));
    }
    if (verbose) printf("filter: %llu\n", filter);
    return filter;
}
    
DEVICE_FORCEINLINE unsigned permute_state(poseidon_state &state) {
    // display(state);
    for (unsigned round = 0; round < 30; round++) {
        if (round < 4 || round >= 26) {
        poseidon::apply_round_constants(state, round);
        poseidon::apply_non_linearity(state);
        poseidon::apply_mds_matrix_new(state);
        } else {
        if (round == 4) {
            poseidon::partial_rounds_init(state);
        }
        poseidon::partial_round_optimized(state, round);
        }
    }
    // printf("state after permute:\n");
    // display(state);
}

DEVICE_FORCEINLINE unsigned duplexing(poseidon_state &state,
    base_field *input_buffer, const unsigned in_idx) {
    for (unsigned j = 0; j < in_idx; j++) {
        state[j] = base_field::into<3>(input_buffer[j]);
    }
    // display(state);
    // poseidon::permutation(state);
    for (unsigned round = 0; round < 30; round++) {
        if (round < 4 || round >= 26) {
        poseidon::apply_round_constants(state, round);
        poseidon::apply_non_linearity(state);
        poseidon::apply_mds_matrix_new(state);
        } else {
        if (round == 4) {
            poseidon::partial_rounds_init(state);
        }
        poseidon::partial_round_optimized(state, round);
        }
    }
    for (unsigned j = 0; j < 8; j++) {
        input_buffer[j] = base_field::from_u64(0);
    }
    // printf("state after duplexing:\n");
    // display(state);
}

DEVICE_FORCEINLINE unsigned set_output(poseidon_state &state,
    base_field *output_buffer) {
    for (unsigned j = 0; j < 8; j++) {
        output_buffer[j] = base_field::field3_to_field2(state[j]);
    }
}

DEVICE_FORCEINLINE unsigned set_sponge(poseidon_state &state,
    base_field *sponge) {
    for (unsigned j = 0; j < 12; j++) {
        sponge[j] = base_field::field3_to_field2(state[j]);
    }
}

extern "C" __launch_bounds__(128, 8) __global__
void observe(base_field *elements, base_field *sponge,
    base_field *input_buffer, base_field *output_buffer,
    const unsigned num_elements,
    const unsigned in_idx, const unsigned out_idx, const unsigned offset
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id == 0) {
        poseidon_state state{};
        for (unsigned j = 0; j < 12; j++) {
            state[j] = base_field::into<3>(sponge[j]);
        }
        unsigned current = in_idx;
        for (unsigned i = 0; i < num_elements; i++) {
            input_buffer[current] = elements[i + offset];
            current = (current + 1) % 8;
            if (current == 0) {
                duplexing(state, input_buffer, 8);
                if (num_elements - i <= 8) {
                    set_output(state, output_buffer);
                }
            }
        }
        set_sponge(state, sponge);
    }
}

extern "C" __launch_bounds__(128, 8) __global__
void get(base_field *challenges, base_field *sponge,
    base_field *input_buffer, base_field *output_buffer,
    const unsigned num_challenges,
    const unsigned in_idx, const unsigned out_idx, const unsigned offset
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id == 0) {
        poseidon_state state{};
        for (unsigned j = 0; j < 12; j++) {
            state[j] = base_field::into<3>(sponge[j]);
        }
        unsigned current = out_idx;
        if (in_idx != 0 || out_idx == 0) {
            duplexing(state, input_buffer, in_idx);
            set_output(state, output_buffer);
            current = 8;
        }
        for (unsigned i = 0; i < num_challenges; i++) {
            if (current == 0) {
                duplexing(state, input_buffer, 0);
                set_output(state, output_buffer);
                current = 8;
            }
            challenges[i + offset] = output_buffer[current - 1];
            // printf("%x\n", base_field::to_u64(challenges[i]));
            current = (current + 7) % 8;
        }
        set_sponge(state, sponge);
    }
}

extern "C" __launch_bounds__(128, 8) __global__
void fri_pow_new(const base_field *input_buffer, const base_field *sponge, base_field *witness, unsigned *found, unsigned *found_buf,
base_field *buf, const base_field stride, const unsigned in_idx, const unsigned min_leading_zeros, const unsigned offset) {
const uint64_t g_id = blockIdx.x * blockDim.x + threadIdx.x;
base_field current = base_field::from_u64(g_id + offset);
__shared__ field<3> init_state[12];
if (threadIdx.x == 0) {
#pragma unroll
    for (unsigned j = 0; j < 12; j++) {
    init_state[j] = base_field::into<3>(sponge[j]);
    }
    for (unsigned j = 0; j < in_idx; j++) {
    init_state[j] = base_field::into<3>(input_buffer[j]);
    }
}
#pragma unroll
for (unsigned i = 0; i < (1 << 3); i++) {
    poseidon_state state{};
#pragma unroll
    for (unsigned j = 0; j < 12; j++) {
    state[j] = init_state[j];
    }
    state[in_idx] = base_field::into<3>(current);
    permute_state(state);
    uint64_t clg = base_field::to_u64(base_field::field3_to_field2(state[7]));
    uint64_t more = clg >> (63 - min_leading_zeros);
    if (more == 0) {
        buf[g_id] = current;
        found_buf[g_id] = 1;
        break;
        // printf("found! gid: %llu, w: %llu, clg: %llu\n", g_id, current, clg);
    }
    current = base_field::add(current, stride);
}
if (g_id == 0) {
    const unsigned num_threads = base_field::to_u64(stride);
    for (unsigned i = 0; i < num_threads; i++) {
    if (found_buf[i] == 1) {
        *witness = buf[i];
        *found = 1;
        break;
    }
    }
}
}

extern "C" __launch_bounds__(128, 8) __global__
void hash_pi(const base_field *witness, const size_t *pi_index,
    base_field *pi, base_field *pi_hash, const size_t num_public_inputs
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_id == 0) {
        // printf("Hello, PI!\n");
        poseidon_state state{};
        const unsigned num_blocks = num_public_inputs >> 3;
        const unsigned num_left = num_public_inputs - (num_blocks << 3);
        for (unsigned i = 0; i < num_blocks; i++) {
            for (unsigned j = 0; j < 8; j++) {
            base_field p = witness[pi_index[j + 8 * i]];
            pi[j + 8 * i] = p;
            state[j] = base_field::into<3>(p);
            }
            permute_state(state);
        }
        for (unsigned j = 0; j < num_left; j++) {
            base_field p = witness[pi_index[j + 8 * num_blocks]];
            state[j] = base_field::into<3>(p);
            pi[j + 8 * num_blocks] = p;
        }
        if (num_left != 0) {
            permute_state(state);
        }
        for (unsigned j = 0; j < 4; j++) {
            pi_hash[j] = base_field::field3_to_field2(state[j]);
        }
    }
}

  extern "C" __launch_bounds__(128, 8) __global__
  void poseidon_gate(const base_field *wires_ptr, const base_field *constants_ptr, const base_field *public_inputs_hash, base_field *output,
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
    // Init, 5
    // const unsigned lde_started = reverse_bits(g_id + j * row_step, degree_bits);
    const unsigned lde_started = g_id + j * row_step;
    const base_field *wire_start = wires_ptr + lde_started;
    base_field *cons_start = output + g_id + j * row_step;
    unsigned cons_offset = 0;
    const base_field swap = memory::load_cs(wire_start + 24 * nrows);
    memory::store_cs(cons_start + cons_offset, base_field::mul(swap, base_field::add(swap, base_field::minus_one())));
    cons_offset += nrows;
    // base_field state[12];
    poseidon_state state{};
#pragma unroll
    for (unsigned i = 0; i < 4; i++) {
      base_field input_lhs = memory::load_cs(wire_start + i * nrows);
      base_field input_rhs = memory::load_cs(wire_start + (i + 4) * nrows);
      base_field delta_i = memory::load_cs(wire_start + (i + 25) * nrows);
      memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::mul(swap, base_field::sub(input_rhs, input_lhs)), delta_i));
      cons_offset += nrows;
      state[i] = base_field::into<3>(base_field::add(input_lhs, delta_i));
      state[i + 4] = base_field::into<3>(base_field::sub(input_rhs, delta_i));
    }
#pragma unroll
    for (unsigned i = 8; i < 12; i++) {
      state[i] = base_field::into<3>(memory::load_cs(wire_start + i * nrows));
    }

    unsigned round_ctr = 0;
    // First set of full rounds, 36
#pragma unroll
    for (unsigned r = 0; r < 4; r++) {
      poseidon::apply_round_constants(state, round_ctr);
      if (r != 0) {
#pragma unroll
        for (unsigned i = 0; i < 12; i++) {
          base_field sbox_in = memory::load_cs(wire_start + (29 + 12 * (r - 1) + i) * nrows);
          memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), sbox_in));
          cons_offset += nrows;
          state[i] = base_field::into<3>(sbox_in);
        }
      }
      poseidon::apply_non_linearity(state);
      poseidon::apply_mds_matrix_new(state);
      round_ctr += 1;
    }

    // Partial rounds, 22
    poseidon::partial_rounds_init(state);
#pragma unroll
    for (unsigned r = 0; r < 22; r++) {
      base_field sbox_in = memory::load_cs(wire_start + (29 + 12 * 3 + r) * nrows); // START_FULL_0 + SPONGE_WIDTH * (poseidon::HALF_N_FULL_ROUNDS - 1) + round
      memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[0]), sbox_in));
      cons_offset += nrows;
      state[0] = base_field::into<3>(sbox_in);
      poseidon::partial_round_optimized(state, r + 4); // + HALF_NUM_FULL_ROUNDS
    }
    round_ctr += 22;

    // Second set of full rounds, 48
#pragma unroll
    for (unsigned r = 0; r < 4; r++) {
      poseidon::apply_round_constants(state, round_ctr);
#pragma unroll
      for (unsigned i = 0; i < 12; i++) {
        base_field sbox_in = memory::load_cs(wire_start + (87 + 12 * r + i) * nrows);
        memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), sbox_in));
        cons_offset += nrows;
        state[i] = base_field::into<3>(sbox_in);
      }
      poseidon::apply_non_linearity(state);
      poseidon::apply_mds_matrix_new(state);
      round_ctr += 1;
    }

    // Output, 12
#pragma unroll
    for (unsigned i = 0; i < 12; i++) {
      base_field output = memory::load_cs(wire_start + (i + 12) * nrows);
      memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), output));
      cons_offset += nrows;
    }
  }
}
}

extern "C" __launch_bounds__(128, 8) __global__
  void poseidon_gate_in_place(const base_field *wires_ptr, const base_field *constants_ptr, const base_field *public_inputs_hash, base_field *output, const base_field *alphas,
                     const unsigned degree_bits, const unsigned num_wires, const unsigned num_const_sigmas, const unsigned num_gate_constraints,
                     const unsigned num_selectors, const unsigned num_constants,
      const unsigned param1,
      const unsigned param2,
      const unsigned param3,
      const unsigned row,
      const unsigned selector_index, const unsigned start, const unsigned end) {
const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
const unsigned nrows = 1 << degree_bits;
const unsigned row_step = nrows / TILE_Q;
// Calculate constraints
if (g_id < row_step) {
#pragma unroll
  for (unsigned j = 0; j < TILE_Q; j++) {
    // Init, 5
    // const unsigned lde_started = reverse_bits(g_id + j * row_step, degree_bits);
    const unsigned index = g_id + j * row_step;
    // Calculate filter
    base_field filter = compute_filter(index, constants_ptr, row, selector_index, start, end, nrows);

    base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
    base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

    const base_field *wire_start = wires_ptr + index;
    base_field *cons_start = output + index;
    unsigned cons_offset = 0;
    const base_field swap = memory::load_cs(wire_start + 24 * nrows);
  //   memory::store_cs(cons_start + cons_offset, base_field::mul(swap, base_field::add(swap, base_field::minus_one())));
  //   cons_offset += nrows;
    base_field cons = base_field::mul(swap, base_field::add(swap, base_field::minus_one()));
    mul_pow_add(cons, acc, alphas, alpha_pows);
    // base_field state[12];
    poseidon_state state{};
#pragma unroll
    for (unsigned i = 0; i < 4; i++) {
      base_field input_lhs = memory::load_cs(wire_start + i * nrows);
      base_field input_rhs = memory::load_cs(wire_start + (i + 4) * nrows);
      base_field delta_i = memory::load_cs(wire_start + (i + 25) * nrows);
      // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::mul(swap, base_field::sub(input_rhs, input_lhs)), delta_i));
      // cons_offset += nrows;
      cons = base_field::sub(base_field::mul(swap, base_field::sub(input_rhs, input_lhs)), delta_i);
      mul_pow_add(cons, acc, alphas, alpha_pows);
      state[i] = base_field::into<3>(base_field::add(input_lhs, delta_i));
      state[i + 4] = base_field::into<3>(base_field::sub(input_rhs, delta_i));
    }
#pragma unroll
    for (unsigned i = 8; i < 12; i++) {
      state[i] = base_field::into<3>(memory::load_cs(wire_start + i * nrows));
    }

    unsigned round_ctr = 0;
    // First set of full rounds, 36
#pragma unroll
    for (unsigned r = 0; r < 4; r++) {
      poseidon::apply_round_constants(state, round_ctr);
      if (r != 0) {
#pragma unroll
        for (unsigned i = 0; i < 12; i++) {
          base_field sbox_in = memory::load_cs(wire_start + (29 + 12 * (r - 1) + i) * nrows);
          // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), sbox_in));
          // cons_offset += nrows;
          cons = base_field::sub(base_field::field3_to_field2(state[i]), sbox_in);
          mul_pow_add(cons, acc, alphas, alpha_pows);
          state[i] = base_field::into<3>(sbox_in);
        }
      }
      poseidon::apply_non_linearity(state);
      poseidon::apply_mds_matrix_new(state);
      round_ctr += 1;
    }

    // Partial rounds, 22
    poseidon::partial_rounds_init(state);
#pragma unroll
    for (unsigned r = 0; r < 22; r++) {
      base_field sbox_in = memory::load_cs(wire_start + (29 + 12 * 3 + r) * nrows); // START_FULL_0 + SPONGE_WIDTH * (poseidon::HALF_N_FULL_ROUNDS - 1) + round
      // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[0]), sbox_in));
      // cons_offset += nrows;
      cons = base_field::sub(base_field::field3_to_field2(state[0]), sbox_in);
      mul_pow_add(cons, acc, alphas, alpha_pows);
      state[0] = base_field::into<3>(sbox_in);
      poseidon::partial_round_optimized(state, r + 4); // + HALF_NUM_FULL_ROUNDS
    }
    round_ctr += 22;

    // Second set of full rounds, 48
#pragma unroll
    for (unsigned r = 0; r < 4; r++) {
      poseidon::apply_round_constants(state, round_ctr);
#pragma unroll
      for (unsigned i = 0; i < 12; i++) {
        base_field sbox_in = memory::load_cs(wire_start + (87 + 12 * r + i) * nrows);
      //   memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), sbox_in));
      //   cons_offset += nrows;
        cons = base_field::sub(base_field::field3_to_field2(state[i]), sbox_in);
        mul_pow_add(cons, acc, alphas, alpha_pows);
        state[i] = base_field::into<3>(sbox_in);
      }
      poseidon::apply_non_linearity(state);
      poseidon::apply_mds_matrix_new(state);
      round_ctr += 1;
    }

    // Output, 12
#pragma unroll
    for (unsigned i = 0; i < 12; i++) {
      base_field output = memory::load_cs(wire_start + (i + 12) * nrows);
      // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), output));
      // cons_offset += nrows;
      cons = base_field::sub(base_field::field3_to_field2(state[i]), output);
      mul_pow_add(cons, acc, alphas, alpha_pows);
    }
    // Store to quotient
#pragma unroll
    for (unsigned k = 0; k < N_CLGS; k++) {
      base_field quotient_acc = output[index + k * nrows * 8];
      output[index + k * nrows * 8] = filter * acc[k] + quotient_acc;
    }
  }
}
}

const unsigned SPONGE_RATE = 8;
const unsigned SPONGE_CAPACITY = 4;
const unsigned SPONGE_WIDTH = SPONGE_RATE + SPONGE_CAPACITY;
const unsigned HALF_N_FULL_ROUNDS = 4;
const unsigned N_PARTIAL_ROUNDS = 22;

extern "C" __global__ void poseidon_generator_kernel(
    base_field *witness,
    const unsigned *representative_map,
    const size_t *u_params,
    const base_field *f_params,
    const unsigned u_start,
    const unsigned f_start,
    const unsigned length
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        poseidon::poseidon_state state{};
        for (int i = 0; i < SPONGE_WIDTH; i++) {
            state[i] = base_field::into<3>(witness[representative_map[u_params[u_start + tid] + i]]);
        }

        base_field swap_value = witness[representative_map[u_params[u_start + tid] + 2 * SPONGE_WIDTH]];

        for (int i = 0; i < 4; i++) {
            base_field delta_i = base_field::mul(swap_value, base_field::sub(base_field::field3_to_field2(state[i + 4]), base_field::field3_to_field2(state[i])));
            witness[representative_map[u_params[u_start + tid] + (2 * SPONGE_WIDTH + 1 + i)]] = delta_i;
        }

        if (base_field::to_u64(swap_value) == 1) {
            for (int i = 0; i < 4; i++) {
            field<3> temp = state[i];
            state[i] = state[i + 4];
            state[i + 4] = temp;
            }
        }

        int round_ctr = 0;

        for (int r = 0; r < HALF_N_FULL_ROUNDS; r++) {
            poseidon::apply_round_constants(state, round_ctr);
            if (r != 0) {
                for (int i = 0; i < SPONGE_WIDTH; i++) {
                    witness[representative_map[u_params[u_start + tid] + 2 * SPONGE_WIDTH + 5 + (r - 1) * SPONGE_WIDTH + i]] = base_field::field3_to_field2(state[i]);
                }
            }
            poseidon::apply_non_linearity(state);
            poseidon::apply_mds_matrix_new(state);
            round_ctr++;
        }
        poseidon::partial_rounds_init(state);

        for (int r = 0; r < N_PARTIAL_ROUNDS; r++) {
            witness[representative_map[u_params[u_start + tid] + 2 * SPONGE_WIDTH + 5 + (HALF_N_FULL_ROUNDS - 1) * SPONGE_WIDTH + r]] = base_field::field3_to_field2(state[0]);
            poseidon::partial_round_optimized(state, r + HALF_N_FULL_ROUNDS);
        }

        round_ctr += N_PARTIAL_ROUNDS;

        for (int r = 0; r < HALF_N_FULL_ROUNDS; r++) {
            poseidon::apply_round_constants(state, round_ctr);
            for (int i = 0; i < SPONGE_WIDTH; i++) {
            witness[representative_map[u_params[u_start + tid] + 2 * SPONGE_WIDTH + 1 + 4 + SPONGE_WIDTH * (HALF_N_FULL_ROUNDS - 1) + N_PARTIAL_ROUNDS + r * SPONGE_WIDTH + i]] =
                base_field::field3_to_field2(state[i]);
            }
            poseidon::apply_non_linearity(state);
            poseidon::apply_mds_matrix_new(state);
            round_ctr++;
        }
        for (int i = 0; i < SPONGE_WIDTH; i++) {
            witness[representative_map[u_params[u_start + tid] + SPONGE_WIDTH + i]] = base_field::field3_to_field2(state[i]);
        }
    }
}

extern "C" __global__ void poseidon_generator_new_kernel(
    base_field *witness,
    const unsigned *rep_map,
    const size_t *params,
    const unsigned write_start,
    const unsigned read_start,
    const unsigned param_start,
    const unsigned length
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = length;
    unsigned read_count = 0, write_count = 0;

    if (tid < length) {
        poseidon::poseidon_state state{};
        for (int i = 0; i < SPONGE_WIDTH; i++) {
            state[i] = base_field::into<3>(witness[rep_map[read_start + tid + stride * read_count++]]);
        }

        base_field swap_value = witness[rep_map[read_start + tid + stride * read_count++]];

        for (int i = 0; i < 4; i++) {
            base_field delta_i = base_field::mul(swap_value, base_field::sub(base_field::field3_to_field2(state[i + 4]), base_field::field3_to_field2(state[i])));
            witness[write_start + tid + stride * write_count++] = delta_i;
        }

        if (base_field::to_u64(swap_value) == 1) {
            for (int i = 0; i < 4; i++) {
            field<3> temp = state[i];
            state[i] = state[i + 4];
            state[i + 4] = temp;
            }
        }

        int round_ctr = 0;

        for (int r = 0; r < HALF_N_FULL_ROUNDS; r++) {
            poseidon::apply_round_constants(state, round_ctr);
            if (r != 0) {
                for (int i = 0; i < SPONGE_WIDTH; i++) {
                    witness[write_start + tid + stride * write_count++] = base_field::field3_to_field2(state[i]);
                }
            }
            poseidon::apply_non_linearity(state);
            poseidon::apply_mds_matrix_new(state);
            round_ctr++;
        }
        poseidon::partial_rounds_init(state);

        for (int r = 0; r < N_PARTIAL_ROUNDS; r++) {
            witness[write_start + tid + stride * write_count++] = base_field::field3_to_field2(state[0]);
            poseidon::partial_round_optimized(state, r + HALF_N_FULL_ROUNDS);
        }

        round_ctr += N_PARTIAL_ROUNDS;

        for (int r = 0; r < HALF_N_FULL_ROUNDS; r++) {
            poseidon::apply_round_constants(state, round_ctr);
            for (int i = 0; i < SPONGE_WIDTH; i++) {
            witness[write_start + tid + stride * write_count++] =
                base_field::field3_to_field2(state[i]);
            }
            poseidon::apply_non_linearity(state);
            poseidon::apply_mds_matrix_new(state);
            round_ctr++;
        }
        for (int i = 0; i < SPONGE_WIDTH; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::field3_to_field2(state[i]);
        }
    }
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

DEVICE_FORCEINLINE base_field load_from_either(const base_field *lde_start, const base_field *start, const unsigned num_ldes, const unsigned lde_degree_bits, const unsigned degree_bits, const unsigned index) {
    return (index < num_ldes) ? memory::load_cs(lde_start + (index << lde_degree_bits)) : memory::load_cs(start + (index << degree_bits));
}

DEVICE_FORCEINLINE base_field compute_filter_partial(const base_field *ldes_ptr, const base_field *polys_ptr, const unsigned num_ldes,
        const unsigned row, const unsigned selector_index, const unsigned start, const unsigned end, const unsigned lde_degree_bits, const unsigned degree_bits,
        const bool verbose=false) {
    base_field s = load_from_either(ldes_ptr, polys_ptr, num_ldes, lde_degree_bits, degree_bits, selector_index);
    base_field filter{0x1, 0x0};
    const base_field UNUSED_SELECTOR{0xffffffff, 0x0};
    // if (verbose) printf("s: %llu, start: %d, end: %d, row: %d\n", s, start, end, row);
    for (unsigned i = start; i < end; i++) {
      if (i != row) {
          filter = base_field::mul(filter, base_field::sub(base_field::from_u64(i), s));
      }
      if (verbose) printf("i: %d, filter: %llu\n", i, filter);
    }
    if (true) { // many selectors
      filter = base_field::mul(filter, base_field::sub(UNUSED_SELECTOR, s));
    }
    if (verbose) printf("filter: %llu\n", filter);
    return filter;
}

// Poseidon!
extern "C" __launch_bounds__(128, 8) __global__
    void poseidon_gate_partial(const base_field *ldes_ptr, const base_field *polys_ptr, const base_field *public_inputs_hash, base_field *output, const base_field *alphas,
              const unsigned degree_bits, const unsigned rate_bits, const unsigned num_wires, const unsigned num_wires_ldes, const unsigned num_const_sigmas, 
              const unsigned num_const_sigmas_ldes, const unsigned part, const unsigned num_gate_constraints, const unsigned num_selectors, const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index, const unsigned start, const unsigned end) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned nrows = 1 << degree_bits;
  const unsigned lde_degree_bits = degree_bits + rate_bits;
  const unsigned row_step = nrows / TILE_Q;
  const unsigned rev_i = reverse_bits(part, rate_bits);
  // Calculate constraints
  if (g_id < row_step) {
#pragma unroll
    for (unsigned j = 0; j < TILE_Q; j++) {
      // const unsigned lde_started = reverse_bits(g_id + j * row_step, degree_bits);
      const unsigned index = g_id + j * row_step;
      const unsigned index_lde = index + (rev_i << degree_bits);
      const base_field *consts_start = polys_ptr + index;
      const base_field *consts_lde_start = ldes_ptr + index_lde;
      const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
      const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);
      // Calculate filter
      base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);

      base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
      base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

      // const base_field *wire_start = wires_ptr + index;
      base_field *cons_start = output + index;
      unsigned cons_offset = 0;
      const base_field swap = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 24);
      // const base_field swap = memory::load_cs(wire_start + 24 * nrows);
    //   memory::store_cs(cons_start + cons_offset, base_field::mul(swap, base_field::add(swap, base_field::minus_one())));
    //   cons_offset += nrows;
      base_field cons = base_field::mul(swap, base_field::add(swap, base_field::minus_one()));
      mul_pow_add(cons, acc, alphas, alpha_pows);
      // base_field state[12];
      poseidon_state state{};
#pragma unroll
      for (unsigned i = 0; i < 4; i++) {
        base_field input_lhs = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i);
        base_field input_rhs = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i + 4);
        base_field delta_i = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i + 25);
        // base_field input_lhs = memory::load_cs(wire_start + i * nrows);
        // base_field input_rhs = memory::load_cs(wire_start + (i + 4) * nrows);
        // base_field delta_i = memory::load_cs(wire_start + (i + 25) * nrows);
        // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::mul(swap, base_field::sub(input_rhs, input_lhs)), delta_i));
        // cons_offset += nrows;
        cons = base_field::sub(base_field::mul(swap, base_field::sub(input_rhs, input_lhs)), delta_i);
        mul_pow_add(cons, acc, alphas, alpha_pows);
        state[i] = base_field::into<3>(base_field::add(input_lhs, delta_i));
        state[i + 4] = base_field::into<3>(base_field::sub(input_rhs, delta_i));
      }
#pragma unroll
      for (unsigned i = 8; i < 12; i++) {
        state[i] = base_field::into<3>(load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i));
        // state[i] = base_field::into<3>(memory::load_cs(wire_start + i * nrows));
      }

      unsigned round_ctr = 0;
      // First set of full rounds, 36
#pragma unroll
      for (unsigned r = 0; r < 4; r++) {
        poseidon::apply_round_constants(state, round_ctr);
        if (r != 0) {
#pragma unroll
          for (unsigned i = 0; i < 12; i++) {
            base_field sbox_in = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 29 + 12 * (r - 1) + i);
            // base_field sbox_in = memory::load_cs(wire_start + (29 + 12 * (r - 1) + i) * nrows);
            // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), sbox_in));
            // cons_offset += nrows;
            cons = base_field::sub(base_field::field3_to_field2(state[i]), sbox_in);
            mul_pow_add(cons, acc, alphas, alpha_pows);
            state[i] = base_field::into<3>(sbox_in);
          }
        }
        poseidon::apply_non_linearity(state);
        poseidon::apply_mds_matrix_new(state);
        round_ctr += 1;
      }

      // Partial rounds, 22
      poseidon::partial_rounds_init(state);
#pragma unroll
      for (unsigned r = 0; r < 22; r++) {
        base_field sbox_in = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 29 + 12 * 3 + r);
        // base_field sbox_in = memory::load_cs(wire_start + (29 + 12 * 3 + r) * nrows); // START_FULL_0 + SPONGE_WIDTH * (poseidon::HALF_N_FULL_ROUNDS - 1) + round
        // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[0]), sbox_in));
        // cons_offset += nrows;
        cons = base_field::sub(base_field::field3_to_field2(state[0]), sbox_in);
        mul_pow_add(cons, acc, alphas, alpha_pows);
        state[0] = base_field::into<3>(sbox_in);
        poseidon::partial_round_optimized(state, r + 4); // + HALF_NUM_FULL_ROUNDS
      }
      round_ctr += 22;

      // Second set of full rounds, 48
#pragma unroll
      for (unsigned r = 0; r < 4; r++) {
        poseidon::apply_round_constants(state, round_ctr);
#pragma unroll
        for (unsigned i = 0; i < 12; i++) {
          base_field sbox_in = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 87 + 12 * r + i);
          // base_field sbox_in = memory::load_cs(wire_start + (87 + 12 * r + i) * nrows);
        //   memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), sbox_in));
        //   cons_offset += nrows;
          cons = base_field::sub(base_field::field3_to_field2(state[i]), sbox_in);
          mul_pow_add(cons, acc, alphas, alpha_pows);
          state[i] = base_field::into<3>(sbox_in);
        }
        poseidon::apply_non_linearity(state);
        poseidon::apply_mds_matrix_new(state);
        round_ctr += 1;
      }

      // Output, 12
#pragma unroll
      for (unsigned i = 0; i < 12; i++) {
        base_field output = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i + 12);
        // base_field output = memory::load_cs(wire_start + (i + 12) * nrows);
        // memory::store_cs(cons_start + cons_offset, base_field::sub(base_field::field3_to_field2(state[i]), output));
        // cons_offset += nrows;
        cons = base_field::sub(base_field::field3_to_field2(state[i]), output);
        mul_pow_add(cons, acc, alphas, alpha_pows);
      }
      // Store to quotient
#pragma unroll
      for (unsigned k = 0; k < N_CLGS; k++) {
        base_field quotient_acc = output[index + (k << lde_degree_bits)];
        output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
      }
    }
  }
}