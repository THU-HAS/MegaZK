#include "goldilocks_extension.cuh"

using namespace goldilocks;

#define TILE_Q 16
#define N_CLGS 2
// #define RATE 8

// #include "poseidon_single_thread.cuh"

// typedef field<3> poseidon_state[12];

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

DEVICE_FORCEINLINE void mul_pow_add(base_field in, base_field *acc, const base_field *alphas, base_field *alpha_pows) {
#pragma unroll
    for (unsigned i = 0; i < N_CLGS; i++) { // TODO: check if constant is faster
        acc[i] = acc[i] + alpha_pows[i] * in;
        alpha_pows[i] = alpha_pows[i] * alphas[i];
    }
}

DEVICE_FORCEINLINE base_field compute_filter_partial(const base_field *ldes_ptr, const base_field *polys_ptr, const unsigned num_ldes,
        const unsigned row, const unsigned selector_index, const unsigned start, const unsigned end, const unsigned lde_degree_bits, const unsigned degree_bits,
        const bool verbose=false) {
    base_field s = load_from_either(ldes_ptr, polys_ptr, num_ldes, lde_degree_bits, degree_bits, selector_index);
    // if (selector_index < num_ldes) {
    //     s = memory::load_cs(ldes_ptr + (selector_index << lde_degree_bits));
    // } else {
    //     s = memory::load_cs(polys_ptr + (selector_index << degree_bits));
    // }
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

extern "C" __launch_bounds__(128, 8) __global__
    void z_partial_partial(const base_field *points_ptr, const base_field *z_h_coset_ptr, const base_field *k_is_ptr, const base_field *alphas,
      const base_field *ldes_ptr, const base_field *polys_ptr, const base_field *betas, const base_field *gammas, base_field *output, const unsigned degree_bits,
      const unsigned part, const unsigned num_wires, const unsigned num_wires_ldes, const unsigned num_const_sigmas, const unsigned num_const_sigmas_ldes, unsigned num_challenges,
      const unsigned num_routed_wires, const unsigned num_partial_products, const unsigned num_partial_products_ldes, const unsigned num_gate_constraints
    ) {
  const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned rate_bits = 3;
  const unsigned nrows = 1 << degree_bits;
  const unsigned lde_degree_bits = degree_bits + rate_bits;
  const unsigned row_step = nrows / TILE_Q;
  const unsigned rev_i = reverse_bits(part, rate_bits);
  const unsigned num_zpp = (num_routed_wires >> 3) * num_challenges;
  const unsigned rate = 8;
  // const unsigned stride = num_partial_products + 1;
  const unsigned const_offset = num_const_sigmas - num_routed_wires;
//   const unsigned part_r = reverse_bits(part, 3);
  if (g_id < row_step) {
#pragma unroll
    for (unsigned j = 0; j < TILE_Q; j++) {
      const unsigned index = g_id + j * row_step;
      const unsigned index_lde = index + (rev_i << degree_bits);
      const unsigned idx_r = reverse_bits(index, degree_bits);
      const unsigned lde_started = index;
      const unsigned next_lde_started = reverse_bits((idx_r + 1) % nrows, degree_bits);
      const unsigned next_lde_started_lde = next_lde_started + (rev_i << degree_bits);
      const base_field *consts_start = polys_ptr + index;
      const base_field *consts_lde_start = ldes_ptr + index_lde;
      // const base_field *consts_start = polys_ptr + index + (const_offset << degree_bits);
      // const base_field *consts_lde_start = ldes_ptr + index_lde + (const_offset << lde_degree_bits);
      const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
      const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);
      const base_field *zpps_start = polys_ptr + index + (num_const_sigmas + num_wires << degree_bits);
      const base_field *zpps_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes + num_wires_ldes << lde_degree_bits);
      const base_field *next_zpps_start = polys_ptr + next_lde_started + (num_const_sigmas + num_wires << degree_bits);
      const base_field *next_zpps_lde_start = ldes_ptr + next_lde_started_lde + (num_const_sigmas_ldes + num_wires_ldes << lde_degree_bits);

      // const base_field *wire_start = wires_ptr + lde_started;
      // const base_field *const_start = constants_ptr + lde_started + const_offset * nrows;
    //   unsigned cons_offset = num_challenges * nrows;
    //   base_field *out_start = output + index;
      const base_field x = base_field::mul(base_field::from_u64(7), memory::load_cs(points_ptr + idx_r * rate + part));
      // const base_field *zpp_start = zs_partial_products_ptr + lde_started;
      // const base_field *next_zpp_start = zs_partial_products_ptr + next_lde_started;
      const base_field l_0_x = base_field::mul(memory::load_cs(z_h_coset_ptr + part),
                                               base_field::inv(base_field::mul(base_field::from_u64(nrows), // WARNING: rate bits!
                                                                               base_field::add(x, base_field::minus_one()))));
      // if (g_id == 0) printf("gid: %d, l0x: %lld, x: %lld, index: %d\n", g_id, base_field::to_u64(l_0_x), base_field::to_u64(x), index);
      base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
      base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}}; // assert num_challenges = 2!

      // z1 terms
#pragma unroll
      for (unsigned k = 0; k < N_CLGS; k++) {
        const base_field z_x = load_from_either(zpps_lde_start, zpps_start, num_partial_products_ldes, lde_degree_bits, degree_bits, k);
        // const base_field z_x = memory::load_cs(zpp_start + k * nrows);
        base_field cons = base_field::mul(l_0_x, base_field::add(z_x, base_field::minus_one()));
        // if (g_id == 0 && j == 3) printf("cons: %llu\n", cons);
        mul_pow_add(cons, acc, alphas, alpha_pows);
      }

      // pp terms
#pragma unroll
      for (unsigned k = 0; k < N_CLGS; k++) {
        // base_field *start = out_start + k * stride;
        const base_field beta = memory::load_cs(betas + k);
        const base_field gamma = memory::load_cs(gammas + k);
        const base_field z_x = load_from_either(zpps_lde_start, zpps_start, num_partial_products_ldes, lde_degree_bits, degree_bits, k);
        const base_field z_gx = load_from_either(next_zpps_lde_start, next_zpps_start, num_partial_products_ldes, lde_degree_bits, degree_bits, k);
        // const base_field z_x = memory::load_cs(zpp_start + k * nrows);
        // const base_field z_gx = memory::load_cs(next_zpp_start + k * nrows);

        const base_field *pps_start = zpps_start + ((num_challenges + k * num_partial_products) << degree_bits);
        const base_field *pps_lde_start = zpps_lde_start + ((num_challenges + k * num_partial_products) << lde_degree_bits);
        // const base_field *pp_start = zpp_start + (num_challenges + k * num_partial_products) * nrows;
        base_field numerator_acc{0x1, 0x0}, denominator_acc{0x1, 0x0};
        for (unsigned r = 0; r < rate; r++) {
          const base_field wire = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, r);
          // const base_field wire = memory::load_cs(wire_start + r * nrows);
          numerator_acc = base_field::mul(
              numerator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, base_field::mul(x, memory::load_cs(k_is_ptr + r))), gamma)));
          const base_field sigma = load_from_either(consts_lde_start, consts_start, num_const_sigmas_ldes, lde_degree_bits, degree_bits, r + const_offset);
          denominator_acc =
              base_field::mul(denominator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, sigma), gamma)));
          // denominator_acc =
          //     base_field::mul(denominator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, memory::load_cs(const_start + r * nrows)), gamma)));
        }
        base_field pp = load_from_either(zpps_lde_start, zpps_start, num_partial_products_ldes, lde_degree_bits, degree_bits, num_challenges + k * num_partial_products);
        // base_field pp = memory::load_cs(pp_start);
        base_field cons = base_field::sub(base_field::mul(z_x, numerator_acc), base_field::mul(pp, denominator_acc));
        // if (g_id == 0 && j == 3) printf("cons: %llu\n", cons);
        mul_pow_add(cons, acc, alphas, alpha_pows);
        // memory::store_cs(out_start + cons_offset, con);
        // cons_offset += nrows;
        for (unsigned i = 1; i < num_partial_products; i++) {
          base_field numerator_acc{0x1, 0x0}, denominator_acc{0x1, 0x0};
          for (unsigned r = 0; r < rate; r++) {
            unsigned col = i * rate + r;
            const base_field wire = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, col);
            // const base_field wire = memory::load_cs(wire_start + col * nrows);
            numerator_acc = base_field::mul(
                numerator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, base_field::mul(x, memory::load_cs(k_is_ptr + col))), gamma)));
            const base_field sigma = load_from_either(consts_lde_start, consts_start, num_const_sigmas_ldes, lde_degree_bits, degree_bits, col + const_offset);
            denominator_acc =
                base_field::mul(denominator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, sigma), gamma)));
            // denominator_acc =
            //     base_field::mul(denominator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, memory::load_cs(const_start + col * nrows)), gamma)));
          }
          base_field pp_prev = pp;
          pp = load_from_either(zpps_lde_start, zpps_start, num_partial_products_ldes, lde_degree_bits, degree_bits, num_challenges + k * num_partial_products + i);
          // pp = memory::load_cs(pp_start + i * nrows);
          // base_field pp_prev = memory::load_cs(pp_start + i - 1);
          base_field cons = base_field::sub(base_field::mul(pp_prev, numerator_acc), base_field::mul(pp, denominator_acc));
          // if (g_id == 0 && j == 3) printf("cons: %llu\n", cons);
          mul_pow_add(cons, acc, alphas, alpha_pows);
        //   memory::store_cs(out_start + cons_offset, con);
        //   cons_offset += nrows;
        }
        numerator_acc = base_field::from_u64(1);
        denominator_acc = base_field::from_u64(1);
        for (unsigned r = 0; r < rate; r++) {
          unsigned col = num_partial_products * rate + r;
          const base_field wire = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, col);
          // const base_field wire = memory::load_cs(wire_start + col * nrows);
          numerator_acc = base_field::mul(
              numerator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, base_field::mul(x, memory::load_cs(k_is_ptr + col))), gamma)));
          const base_field sigma = load_from_either(consts_lde_start, consts_start, num_const_sigmas_ldes, lde_degree_bits, degree_bits, col + const_offset);
          denominator_acc =
              base_field::mul(denominator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, sigma), gamma)));
          // denominator_acc =
          //     base_field::mul(denominator_acc, base_field::add(wire, base_field::add(base_field::mul(beta, memory::load_cs(const_start + col * nrows)), gamma)));
        }
        cons = base_field::sub(base_field::mul(pp, numerator_acc), base_field::mul(z_gx, denominator_acc));
        // if (g_id == 0 && j == 3) printf("cons: %llu\n", cons);
        mul_pow_add(cons, acc, alphas, alpha_pows);
        // if (g_id == 0 && j == 3) printf("acc[0]: %llu\n", acc[0]);
        // memory::store_cs(out_start + cons_offset, con);
        // cons_offset += nrows;
      }

      // Store to quotient
#pragma unroll
      for (unsigned k = 0; k < N_CLGS; k++) {
        base_field quotient_acc = output[index + (k << lde_degree_bits)];
        // if (g_id == 0 && j == 3 && k == 0) printf("quotient_acc: %llu\n", quotient_acc);
        // base_field res = (acc[k] + quotient_acc * alpha_pows[k]) * z_h_coset_ptr[rate + (index % rate)];
        base_field res = (acc[k] + quotient_acc * alpha_pows[k]) * z_h_coset_ptr[rate + part];
        output[index + (k << lde_degree_bits)] = res;
        // if (g_id == 0 && j == 3 && k == 0) printf("q_value: %llu, to_add: %llu, z_h_inv: %llu\n", res, quotient_acc * alpha_pows[k], z_h_coset_ptr[rate + (index % rate)]);
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void noop_gate_partial(const base_field *ldes_ptr, const base_field *polys_ptr, const base_field *public_inputs_hash, base_field *output, const base_field *alphas,
                       const unsigned degree_bits, const unsigned rate_bits, const unsigned num_wires, const unsigned num_wires_ldes, const unsigned num_const_sigmas, 
                       const unsigned num_const_sigmas_ldes, const unsigned part, const unsigned num_gate_constraints, const unsigned num_selectors, const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index, const unsigned start, const unsigned end) {}

extern "C" __launch_bounds__(128, 8) __global__
    void constant_gate_partial(const base_field *ldes_ptr, const base_field *polys_ptr, const base_field *public_inputs_hash, base_field *output, const base_field *alphas,
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
      // if (g_id == 0 && j == 3) printf("filter: %llu\n", filter);
      base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
      base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};
      for (unsigned i = 0; i < num_constants; i++) {
        // base_field constraint = base_field::sub(memory::load_cs(constants_ptr + lde_started * num_const_sigmas + (i + num_selectors)),
        //                                         memory::load_cs(wires_ptr + lde_started * num_wires + i) //
        // );
        // base_field constant = (i + num_selectors < num_const_sigmas_ldes) ? 
        //     memory::load_cs(consts_lde_start + ((i + num_selectors) << lde_degree_bits)) : memory::load_cs(consts_start + ((i + num_selectors) << degree_bits));
        base_field constant = load_from_either(consts_lde_start, consts_start, num_const_sigmas_ldes, lde_degree_bits, degree_bits, i + num_selectors);
        // base_field res = (i < num_wires_ldes) ?
        //     memory::load_cs(wires_lde_start + (i << lde_degree_bits)) : memory::load_cs(wires_start + (i << degree_bits));
        base_field res = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i);
        base_field cons = constant - res;
        // base_field cons = base_field::sub(memory::load_cs(constants_ptr + index + (i + num_selectors) * nrows),
        //                                         memory::load_cs(wires_ptr + index + i * nrows) //
        // );
        // if (g_id == 0 && j == 3) printf("cons: %llu\n", cons);
        mul_pow_add(cons, acc, alphas, alpha_pows);
        // if (g_id == 0 && j == 3) printf("acc[0]: %llu, filtered[0]: %llu\n", acc[0], filter * acc[0]);
        // memory::store_cs(output + g_id + j * row_step + i * nrows, constraint);
      }
    //   if (g_id == 0) printf("acc0: %llu\n", acc[0]);
      // Store to quotient
#pragma unroll
      for (unsigned k = 0; k < N_CLGS; k++) {
        base_field quotient_acc = output[index + (k << lde_degree_bits)];
        // if (g_id == 0) printf("quotient_acc: %llu\n", quotient_acc);
        output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
    void public_input_gate_partial(const base_field *ldes_ptr, const base_field *polys_ptr, const base_field *public_inputs_hash, base_field *output, const base_field *alphas,
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
      // base_field filter = compute_filter_partial(index, ldes_ptr, polys_ptr, num_const_sigmas_ldes, row, selector_index, start, end, nrows, rate_bits);

      base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
      base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};
#pragma unroll
      for (unsigned i = 0; i < 4; i++) {
        // base_field constraint = base_field::sub(memory::load_cs(wires_ptr + lde_started * num_wires + i), memory::load_cs(public_inputs_hash + i));
        // base_field constraint = base_field::sub(memory::load_cs(wires_ptr + index + i * nrows), memory::load_cs(public_inputs_hash + i));
        // memory::store_cs(output + g_id + j * row_step + i * nrows, constraint);

        base_field cons = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i) - public_inputs_hash[i];
        // base_field cons = base_field::sub(memory::load_cs(wires_ptr + index + i * nrows), memory::load_cs(public_inputs_hash + i));
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

extern "C" __launch_bounds__(128, 8) __global__
    void arithmetic_gate_partial(const base_field *ldes_ptr, const base_field *polys_ptr, const base_field *public_inputs_hash, base_field *output, const base_field *alphas,
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
  const unsigned num_ops = param1;
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
      const base_field c0 = load_from_either(consts_lde_start, consts_start, num_const_sigmas_ldes, lde_degree_bits, degree_bits, num_selectors);
      const base_field c1 = load_from_either(consts_lde_start, consts_start, num_const_sigmas_ldes, lde_degree_bits, degree_bits, num_selectors + 1);
      // const base_field c0 = memory::load_cs(constants_ptr + index + num_selectors * nrows);
      // const base_field c1 = memory::load_cs(constants_ptr + index + (num_selectors + 1) * nrows);
#pragma unroll
      for (unsigned i = 0; i < num_ops; i++) {
        base_field m0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i * 4);
        base_field m1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i * 4 + 1);
        base_field ad = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i * 4 + 2);
        base_field oput = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i * 4 + 3);
        // base_field m0 = memory::load_cs(wires_ptr + index + (4 * i) * nrows);
        // base_field m1 = memory::load_cs(wires_ptr + index + (4 * i + 1) * nrows);
        // base_field ad = memory::load_cs(wires_ptr + index + (4 * i + 2) * nrows);
        // base_field oput = memory::load_cs(wires_ptr + index + (4 * i + 3) * nrows);
        base_field computed = base_field::add(base_field::mul(base_field::mul(m0, m1), c0), base_field::mul(ad, c1));
        base_field cons = base_field::sub(oput, computed);
        mul_pow_add(cons, acc, alphas, alpha_pows);
        // memory::store_cs(output + index + i * nrows, constraint);
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

extern "C" __launch_bounds__(128, 8) __global__
    void u32_add_many_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_addends = param1;
    const unsigned num_ops = param2;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};
            for (unsigned i = 0; i < num_ops; i++) {
                base_field carry = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (num_addends + 3) * i + num_addends);
                // base_field carry = memory::load_cs(wires_ptr + index + ((num_addends + 3) * i + num_addends) * nrows);
                base_field computed_output = carry;
                for (unsigned j = 0; j < num_addends; j++) {
                    base_field addend = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (num_addends + 3) * i + j);
                    // base_field addend = memory::load_cs(wires_ptr + index + ((num_addends + 3) * i + j) * nrows);
                    computed_output = base_field::add(computed_output, addend);
                }
                base_field output_result = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (num_addends + 3) * i + num_addends + 1);
                base_field output_carry = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (num_addends + 3) * i + num_addends + 2);
                // base_field output_result = memory::load_cs(wires_ptr + index + ((num_addends + 3) * i + num_addends + 1) * nrows);
                // base_field output_carry = memory::load_cs(wires_ptr + index + ((num_addends + 3) * i + num_addends + 2) * nrows);

                base_field base = base_field::from_u64(1ULL << 32);
                base_field combined_output = base_field::add(base_field::mul(output_carry, base), output_result);

                base_field cons = base_field::sub(combined_output, computed_output);
                mul_pow_add(cons, acc, alphas, alpha_pows);

                base_field combined_result_limbs = base_field::zero();
                base_field combined_carry_limbs = base_field::zero();

                base = base_field::from_u64(1ULL << 2); // 2 is the self.limb_bits()
                for (int j = 18 - 1; j >= 0; j--) { // 18 is the self.num_limbs()
                    base_field this_limb = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (num_addends + 3) * num_ops + 18 * i + j);
                    // base_field this_limb = memory::load_cs(wires_ptr + index + ((num_addends + 3) * num_ops + 18 * i + j) * nrows);
                    uint64_t max_limb = 1ULL << 2;
                    base_field product = base_field::one();
                    for (unsigned k = 0; k < max_limb; k++) {
                        product = base_field::mul(product, base_field::sub(this_limb, base_field::from_u64(k)));
                    }
                    mul_pow_add(product, acc, alphas, alpha_pows);

                    if (j < 16) { // 16 is the self.num_result_limbs()
                        combined_result_limbs = base_field::add(base_field::mul(base, combined_result_limbs), this_limb);
                    } else {
                        combined_carry_limbs = base_field::add(base_field::mul(base, combined_carry_limbs), this_limb);
                    }
                }
                cons = base_field::sub(combined_result_limbs, output_result);
                mul_pow_add(cons, acc, alphas, alpha_pows);
                cons = base_field::sub(combined_carry_limbs, output_carry);
                mul_pow_add(cons, acc, alphas, alpha_pows);
            }
            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void u32_arithmetic_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);    const unsigned num_ops = param1;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};
            for (unsigned i = 0; i < num_ops; i++) {
                base_field multiplicand_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i);
                base_field multiplicand_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i + 1);
                base_field addend = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i + 2);
                // base_field multiplicand_0 = memory::load_cs(wires_ptr + index + (6 * i) * nrows);
                // base_field multiplicand_1 = memory::load_cs(wires_ptr + index + (6 * i + 1) * nrows);
                // base_field addend = memory::load_cs(wires_ptr + index + (6 * i + 2) * nrows);

                base_field computed_output = base_field::add(base_field::mul(multiplicand_0, multiplicand_1), addend);

                base_field output_low = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i + 3);
                base_field output_high = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i + 4);
                base_field inverse = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i + 5);
                // base_field output_low = memory::load_cs(wires_ptr + index + (6 * i + 3) * nrows);
                // base_field output_high = memory::load_cs(wires_ptr + index + (6 * i + 4) * nrows);
                // base_field inverse = memory::load_cs(wires_ptr + index + (6 * i + 5) * nrows);

                base_field base = base_field::from_u64(1ULL << 32);
                base_field one = base_field::one();
                base_field u32_max = base_field::from_u64(UINT_MAX);

                base_field diff = base_field::sub(u32_max, output_high);
                base_field hi_not_max = base_field::sub(base_field::mul(inverse, diff), one);
                base_field hi_not_max_or_lo_zero = base_field::mul(hi_not_max, output_low);

                mul_pow_add(hi_not_max_or_lo_zero, acc, alphas, alpha_pows);

                base_field combined_output = base_field::add(base_field::mul(output_high, base), output_low);

                base_field cons = base_field::sub(combined_output, computed_output);
                mul_pow_add(cons, acc, alphas, alpha_pows);

                base_field combined_low_limbs = base_field::zero();
                base_field combined_high_limbs = base_field::zero();

                size_t midpoint = 16;
                base = base_field::from_u64(1ULL << 2); // 2 is the self.limb_bits()
                for(int j = 32 - 1; j >= 0; j--) { // 32 is the self.num_limbs()
                    base_field this_limb = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * num_ops + 32 * i + j);
                    // base_field this_limb = memory::load_cs(wires_ptr + index + (6 * num_ops + 32 * i + j) * nrows);
                    uint64_t max_limb = 1ULL << 2; // 2 is the self.limb_bits()
                    base_field product = base_field::one();
                    for (unsigned k = 0; k < max_limb; k++) {
                        product = base_field::mul(product, base_field::sub(this_limb, base_field::from_u64(k)));
                    }
                    mul_pow_add(product, acc, alphas, alpha_pows);

                    if (j < midpoint) {
                        combined_low_limbs = base_field::add(base_field::mul(base, combined_low_limbs), this_limb);
                    } else {
                        combined_high_limbs = base_field::add(base_field::mul(base, combined_high_limbs), this_limb);
                    }
                }
                cons = base_field::sub(combined_low_limbs, output_low);
                mul_pow_add(cons, acc, alphas, alpha_pows);
                cons = base_field::sub(combined_high_limbs, output_high);
                mul_pow_add(cons, acc, alphas, alpha_pows);
            }
            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void comparison_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_bits = param1;
    const unsigned num_chunks = param2;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            base_field first_input = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 0);
            base_field second_input = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 1);
            // base_field first_input = memory::load_cs(wires_ptr + index);
            // base_field second_input = memory::load_cs(wires_ptr + index + nrows);

            uint64_t chunk_bits = (num_bits + num_chunks - 1) / num_chunks;
            uint64_t chunk_size = 1ULL << chunk_bits;

            base_field first_chunks_combined = base_field::zero();
            base_field second_chunks_combined = base_field::zero();
            for(int i = num_chunks - 1; i >= 0; i--) {
                base_field first_chunk = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + i);
                base_field second_chunk = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + num_chunks + i);
                // base_field first_chunk = memory::load_cs(wires_ptr + index + (4 + i) * nrows);
                // base_field second_chunk = memory::load_cs(wires_ptr + index + (4 + num_chunks + i) * nrows);

                first_chunks_combined = base_field::add(base_field::mul(first_chunks_combined, base_field::from_u64(chunk_size)), first_chunk);
                second_chunks_combined = base_field::add(base_field::mul(second_chunks_combined, base_field::from_u64(chunk_size)), second_chunk);
            }

            base_field cons = base_field::sub(first_chunks_combined, first_input);
            mul_pow_add(cons, acc, alphas, alpha_pows);
            cons = base_field::sub(second_chunks_combined, second_input);
            mul_pow_add(cons, acc, alphas, alpha_pows);

            base_field most_significant_diff_so_far = base_field::zero();

            for(unsigned i = 0; i < num_chunks; i++) {
                base_field first_chunk = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + i);
                base_field second_chunk = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + num_chunks + i);
                // base_field first_chunk = memory::load_cs(wires_ptr + index + (4 + i) * nrows);
                // base_field second_chunk = memory::load_cs(wires_ptr + index + (4 + num_chunks + i) * nrows);

                base_field first_product = base_field::one();
                base_field second_product = base_field::one();
                for(unsigned x = 0; x < chunk_size; x++) {
                    first_product = base_field::mul(first_product, base_field::sub(first_chunk, base_field::from_u64(x)));
                    second_product = base_field::mul(second_product, base_field::sub(second_chunk, base_field::from_u64(x)));
                }
                mul_pow_add(first_product, acc, alphas, alpha_pows);
                mul_pow_add(second_product, acc, alphas, alpha_pows);

                base_field difference = base_field::sub(second_chunk, first_chunk);
                base_field equality_dummy = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + 2 * num_chunks + i);
                base_field chunks_equal = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + 3 * num_chunks + i);
                // base_field equality_dummy = memory::load_cs(wires_ptr + index + (4 + 2 * num_chunks + i) * nrows);
                // base_field chunks_equal = memory::load_cs(wires_ptr + index + (4 + 3 * num_chunks + i) * nrows);

                cons = base_field::sub(base_field::mul(difference, equality_dummy), base_field::sub(base_field::one(), chunks_equal));
                mul_pow_add(cons, acc, alphas, alpha_pows);
                cons = base_field::mul(chunks_equal, difference);
                mul_pow_add(cons, acc, alphas, alpha_pows);

                base_field intermediate_value = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + 4 * num_chunks + i);
                // base_field intermediate_value = memory::load_cs(wires_ptr + index + (4 + 4 * num_chunks + i) * nrows);
                cons = base_field::sub(intermediate_value, base_field::mul(chunks_equal, most_significant_diff_so_far));
                mul_pow_add(cons, acc, alphas, alpha_pows);

                most_significant_diff_so_far = base_field::add(intermediate_value, base_field::mul(base_field::sub(base_field::one(), chunks_equal), difference));
            }

            base_field most_significant_diff = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 3);
            // base_field most_significant_diff = memory::load_cs(wires_ptr + index + 3 * nrows);
            cons = base_field::sub(most_significant_diff, most_significant_diff_so_far);
            mul_pow_add(cons, acc, alphas, alpha_pows);

            for(unsigned i = 0; i < chunk_bits + 1; i++){
                base_field bit = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + 5 * num_chunks + i);
                // base_field bit = memory::load_cs(wires_ptr + index + (4 + 5 * num_chunks + i) * nrows);
                cons = base_field::mul(bit, base_field::sub(base_field::one(), bit));
                mul_pow_add(cons, acc, alphas, alpha_pows);
            }

            base_field bits_combined = base_field::zero();
            for(int i = chunk_bits; i >= 0; i--) {
                base_field bit = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + 5 * num_chunks + i);
                // base_field bit = memory::load_cs(wires_ptr + index + (4 + 5 * num_chunks + i) * nrows);
                bits_combined = base_field::add(base_field::mul(bits_combined, base_field::from_u64(2)), bit);
            }
            base_field two_n = base_field::from_u64(1ULL << chunk_bits);
            cons = base_field::sub(base_field::add(two_n, most_significant_diff), bits_combined);
            mul_pow_add(cons, acc, alphas, alpha_pows);


            base_field result_bool = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 2);
            base_field bit = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + 5 * num_chunks + chunk_bits);
            // base_field result_bool = memory::load_cs(wires_ptr + index + 2 * nrows);
            // base_field bit = memory::load_cs(wires_ptr + index + (4 + 5 * num_chunks + chunk_bits) * nrows);
            cons = base_field::sub(result_bool, bit);
            mul_pow_add(cons, acc, alphas, alpha_pows);

            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void u32_interleave_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_ops = param1;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            for(unsigned i = 0; i < num_ops; i++) {
                base_field x = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 2 * i);
                base_field x_interleaved = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 2 * i + 1);
                // base_field x = memory::load_cs(wires_ptr + index + (2 * i) * nrows);
                // base_field x_interleaved = memory::load_cs(wires_ptr + index + (2 * i + 1) * nrows);

                base_field computed_x = base_field::zero();
                base_field computed_x_interleaved = base_field::zero();
                for(unsigned j = 0; j < 32; j++){
                    base_field bit = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 2 * num_ops + 32 * i + j);
                    // base_field bit = memory::load_cs(wires_ptr + index + (2 * num_ops + 32 * i + j) * nrows);
                    computed_x = base_field::add(base_field::mul(computed_x, base_field::from_u64(2)), bit);
                    computed_x_interleaved = base_field::add(base_field::mul(computed_x_interleaved, base_field::from_u64(4)), bit);
                }
                base_field cons = base_field::sub(computed_x, x);
                mul_pow_add(cons, acc, alphas, alpha_pows);
                cons = base_field::sub(computed_x_interleaved, x_interleaved);
                mul_pow_add(cons, acc, alphas, alpha_pows);

                for(unsigned j = 0; j < 32; j++){
                    base_field bit = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 2 * num_ops + 32 * i + j);
                    // base_field bit = memory::load_cs(wires_ptr + index + (2 * num_ops + 32 * i + j) * nrows);
                    base_field product = base_field::one();
                    for(unsigned k = 0; k < 2; k++){
                        product = base_field::mul(product, base_field::sub(bit, base_field::from_u64(k)));
                    }
                    mul_pow_add(product, acc, alphas, alpha_pows);
                }
            }
            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void u32_range_check_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_input_limbs = param1;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            base_field base = base_field::from_u64(1ULL << 2);
            for(unsigned i = 0; i < num_input_limbs; i++) {
                base_field input_limb = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i);
                // base_field input_limb = memory::load_cs(wires_ptr + index + i * nrows);

                base_field computed_sum = base_field::zero();
                for(int j = 15; j >= 0; j--) {
                    base_field aux_limb = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, num_input_limbs + 16 * i + j);
                    // base_field aux_limb = memory::load_cs(wires_ptr + index + (num_input_limbs + 16 * i + j) * nrows);
                    computed_sum = base_field::add(base_field::mul(base, computed_sum), aux_limb);
                }
                base_field cons = base_field::sub(computed_sum, input_limb);
                mul_pow_add(cons, acc, alphas, alpha_pows);

                for(unsigned j = 0; j < 16; j++) {
                    base_field aux_limb = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, num_input_limbs + 16 * i + j);
                    // base_field aux_limb = memory::load_cs(wires_ptr + index + (num_input_limbs + 16 * i + j) * nrows);
                    base_field product = base_field::one();
                    for(unsigned k = 0; k < 4; k++) { // 1 << 2
                        product = base_field::mul(product, base_field::sub(aux_limb, base_field::from_u64(k)));
                    }
                    mul_pow_add(product, acc, alphas, alpha_pows);
                }
            }
            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void u32_subtraction_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_ops = param1;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            for(unsigned i = 0; i < num_ops; i++) {
                base_field input_x = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 5 * i);
                base_field input_y = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 5 * i + 1);
                base_field input_borrow = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 5 * i + 2);
                // base_field input_x = memory::load_cs(wires_ptr + index + (5 * i) * nrows);
                // base_field input_y = memory::load_cs(wires_ptr + index + (5 * i + 1) * nrows);
                // base_field input_borrow = memory::load_cs(wires_ptr + index + (5 * i + 2) * nrows);

                base_field result_initial = base_field::sub(base_field::sub(input_x, input_y), input_borrow);
                base_field base = base_field::from_u64(1ULL << 32);

                base_field output_result = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 5 * i + 3);
                base_field output_borrow = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 5 * i + 4);
                // base_field output_result = memory::load_cs(wires_ptr + index + (5 * i + 3) * nrows);
                // base_field output_borrow = memory::load_cs(wires_ptr + index + (5 * i + 4) * nrows);

                base_field cons = base_field::sub(output_result, base_field::add(result_initial, base_field::mul(output_borrow, base)));
                mul_pow_add(cons, acc, alphas, alpha_pows);

                base_field combined_limbs = base_field::zero();
                base_field limb_base = base_field::from_u64(1ULL << 2);
                for(int j = 15; j >= 0; j--) {
                    base_field this_limb = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 5 * num_ops + 16 * i + j);
                    // base_field this_limb = memory::load_cs(wires_ptr + index + (5 * num_ops + 16 * i + j) * nrows);
                    uint64_t max_limb = 1ULL << 2;
                    base_field product = base_field::one();
                    for(unsigned k = 0; k < max_limb; k++) {
                        product = base_field::mul(product, base_field::sub(this_limb, base_field::from_u64(k)));
                    }
                    mul_pow_add(product, acc, alphas, alpha_pows);

                    combined_limbs = base_field::add(base_field::mul(limb_base, combined_limbs), this_limb);
                }
                cons = base_field::sub(combined_limbs, output_result);
                mul_pow_add(cons, acc, alphas, alpha_pows);

                cons = base_field::mul(output_borrow, base_field::sub(base_field::one(), output_borrow));
                mul_pow_add(cons, acc, alphas, alpha_pows);
            }
            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void uninterleave_to_u32_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_ops = param1;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            for(unsigned i = 0; i < num_ops; i++) {
                base_field x_interleaved = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 3 * i);
                // base_field x_interleaved = memory::load_cs(wires_ptr + index + (3 * i) * nrows);

                base_field computed_x_interleaved = base_field::zero();
                for(unsigned j = 0; j < 64; j++) {
                    base_field bit = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 3 * num_ops + 64 * i + j);
                    // base_field bit = memory::load_cs(wires_ptr + index + (3 * num_ops + 64 * i + j) * nrows);
                    computed_x_interleaved = base_field::add(base_field::mul(computed_x_interleaved, base_field::from_u64(2)), bit);
                }
                base_field cons = base_field::sub(computed_x_interleaved, x_interleaved);
                mul_pow_add(cons, acc, alphas, alpha_pows);

                base_field x_evens = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 3 * i + 1);
                base_field x_odds = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 3 * i + 2);
                // base_field x_evens = memory::load_cs(wires_ptr + index + (3 * i + 1) * nrows);
                // base_field x_odds = memory::load_cs(wires_ptr + index + (3 * i + 2) * nrows);

                base_field computed_x_evens = base_field::zero();
                base_field computed_x_odds = base_field::zero();

                for(unsigned j = 0; j < 32; j++) {
                    base_field jth_even = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 3 * num_ops + 64 * i + 2 * j);
                    base_field jth_odd = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 3 * num_ops + 64 * i + 2 * j + 1);
                    // base_field jth_even = memory::load_cs(wires_ptr + index + (3 * num_ops + 64 * i + 2 * j) * nrows);
                    // base_field jth_odd = memory::load_cs(wires_ptr + index + (3 * num_ops + 64 * i + 2 * j + 1) * nrows);

                    base_field coeff = base_field::from_u64(1ULL << (32 - j - 1));
                    computed_x_evens = base_field::add(computed_x_evens, base_field::mul(jth_even, coeff));
                    computed_x_odds = base_field::add(computed_x_odds, base_field::mul(jth_odd, coeff));
                }

                cons = base_field::sub(computed_x_evens, x_evens);
                mul_pow_add(cons, acc, alphas, alpha_pows);
                cons = base_field::sub(computed_x_odds, x_odds);
                mul_pow_add(cons, acc, alphas, alpha_pows);

                for(unsigned j = 0; j < 64; j++) {
                    base_field bit = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 3 * num_ops + 64 * i + j);
                    // base_field bit = memory::load_cs(wires_ptr + index + (3 * num_ops + 64 * i + j) * nrows);
                    base_field product = base_field::one();
                    for(unsigned k = 0; k < 2; k++) {
                        product = base_field::mul(product, base_field::sub(bit, base_field::from_u64(k)));
                    }
                    mul_pow_add(product, acc, alphas, alpha_pows);
                }
            }
            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void base_sum_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_limbs = param1;
    const unsigned B = param2;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            base_field sum = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 0);
            // base_field sum = memory::load_cs(wires_ptr + index);
            base_field computed_sum = base_field::zero();
            for(int i = num_limbs; i > 0; i--) {
                base_field limb = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, i);
                // base_field limb = memory::load_cs(wires_ptr + index + i * nrows);
                computed_sum = base_field::add(base_field::mul(computed_sum, base_field::from_u64(B)), limb);
            }
            base_field cons = base_field::sub(computed_sum, sum);
            mul_pow_add(cons, acc, alphas, alpha_pows);

            for(unsigned i = 0; i < num_limbs; i++) {
                base_field limb = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 1 + i);
                // base_field limb = memory::load_cs(wires_ptr + index + (1 + i) * nrows);
                base_field product = base_field::one();
                for(unsigned j = 0; j < B; j++) {
                    product = base_field::mul(product, base_field::sub(limb, base_field::from_u64(j)));
                }
                mul_pow_add(product, acc, alphas, alpha_pows);
            }
            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void random_access_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned bits = param1;
    const unsigned num_copies = param2;
    const unsigned num_extra_constants = param3;

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            uint64_t vec_size = 1ULL << bits;
            for(unsigned copy = 0; copy < num_copies; copy++){
                base_field access_index = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (2 + vec_size) * copy);
                base_field claimed_element = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (2 + vec_size) * copy + 1);
                // base_field access_index = memory::load_cs(wires_ptr + index + ((2 + vec_size) * copy) * nrows);
                // base_field claimed_element = memory::load_cs(wires_ptr + index + ((2 + vec_size) * copy + 1) * nrows);

                for(unsigned i = 0; i < bits; i++){
                    base_field bit = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (2 + vec_size) * num_copies + num_extra_constants + bits * copy + i);
                    // base_field bit = memory::load_cs(wires_ptr + index + ((2 + vec_size) * num_copies + num_extra_constants + bits * copy + i) * nrows);
                    base_field cons = base_field::mul(bit, base_field::sub(bit, base_field::one()));
                    mul_pow_add(cons, acc, alphas, alpha_pows);
                }

                base_field reconstructed_index = base_field::zero();
                for(int i = bits - 1; i >= 0; i--){
                    base_field bit = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (2 + vec_size) * num_copies + num_extra_constants + bits * copy + i);
                    // base_field bit = memory::load_cs(wires_ptr + index + ((2 + vec_size) * num_copies + num_extra_constants + bits * copy + i) * nrows);
                    reconstructed_index = base_field::add(base_field::add(reconstructed_index, reconstructed_index), bit);
                }
                base_field cons = base_field::sub(reconstructed_index, access_index);
                mul_pow_add(cons, acc, alphas, alpha_pows);

                // base_field* list_items = (base_field*)malloc(vec_size * sizeof(base_field));
                base_field list_items[16]; // TODO: variable size
                for(unsigned i = 0; i < vec_size; i++){
                    list_items[i] = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (2 + vec_size) * copy + 2 + i);
                    // list_items[i] = memory::load_cs(wires_ptr + index + ((2 + vec_size) * copy + 2 + i) * nrows);
                }
                size_t new_size = vec_size;
                for(unsigned i = 0; i < bits; i++){
                    new_size = new_size / 2;
                    // base_field* new_list_items = (base_field*)malloc(new_size * sizeof(base_field));
                    base_field bit = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (2 + vec_size) * num_copies + num_extra_constants + bits * copy + i);
                    // base_field bit = memory::load_cs(wires_ptr + index + ((2 + vec_size) * num_copies + num_extra_constants + bits * copy + i) * nrows);
                    for(unsigned j = 0; j < new_size; j++){
                        base_field x = list_items[2 * j];
                        base_field y = list_items[2 * j + 1];
                        // new_list_items[j] = base_field::add(x, base_field::mul(bit, base_field::sub(y, x)));
                        list_items[j] = base_field::add(x, base_field::mul(bit, base_field::sub(y, x)));
                    }
                    // free(list_items);
                    // list_items = new_list_items;
                }

                cons = base_field::sub(list_items[0], claimed_element);
                mul_pow_add(cons, acc, alphas, alpha_pows);
            }
            for(unsigned i = 0; i < num_extra_constants; i++){
                base_field constant = load_from_either(consts_lde_start, consts_start, num_wires_ldes, lde_degree_bits, degree_bits, i + num_selectors);
                base_field claimed_constant = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, (2 + vec_size) * num_copies + i);
                // base_field constant = memory::load_cs(constants_ptr + index + (i + num_selectors) * nrows);
                // base_field claimed_constant = memory::load_cs(wires_ptr + index + ((2 + vec_size) * num_copies + i) * nrows);
                base_field cons = base_field::sub(constant, claimed_constant);
                mul_pow_add(cons, acc, alphas, alpha_pows);
            }
            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void arithmetic_extension_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_ops = 10; // TODO

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            base_field const_0 = load_from_either(consts_lde_start, consts_start, num_wires_ldes, lde_degree_bits, degree_bits, num_selectors);
            base_field const_1 = load_from_either(consts_lde_start, consts_start, num_wires_ldes, lde_degree_bits, degree_bits, 1 + num_selectors);
            // base_field const_0 = memory::load_cs(constants_ptr + index + num_selectors * nrows);
            // base_field const_1 = memory::load_cs(constants_ptr + index + (1 + num_selectors) * nrows);

            for(unsigned i = 0; i < num_ops; i++){
                base_field m_00 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 8 * i);
                base_field m_01 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 8 * i + 1);
                // base_field m_00 = memory::load_cs(wires_ptr + index + (8 * i) * nrows);
                // base_field m_01 = memory::load_cs(wires_ptr + index + (8 * i + 1) * nrows);
                extension_field multiplicand_0 = {m_00, m_01};

                base_field m_10 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 8 * i + 2);
                base_field m_11 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 8 * i + 3);
                // base_field m_10 = memory::load_cs(wires_ptr + index + (8 * i + 2) * nrows);
                // base_field m_11 = memory::load_cs(wires_ptr + index + (8 * i + 3) * nrows);
                extension_field multiplicand_1 = {m_10, m_11};

                base_field a_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 8 * i + 4);
                base_field a_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 8 * i + 5);
                // base_field a_0 = memory::load_cs(wires_ptr + index + (8 * i + 4) * nrows);
                // base_field a_1 = memory::load_cs(wires_ptr + index + (8 * i + 5) * nrows);
                extension_field addend = {a_0, a_1};

                base_field o_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 8 * i + 6);
                base_field o_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 8 * i + 7);
                // base_field o_0 = memory::load_cs(wires_ptr + index + (8 * i + 6) * nrows);
                // base_field o_1 = memory::load_cs(wires_ptr + index + (8 * i + 7) * nrows);
                extension_field output = {o_0, o_1};

                extension_field computed_output = extension_field::add(
                    extension_field::mul(
                        extension_field::mul(multiplicand_0, multiplicand_1),
                        const_0
                    ),
                    extension_field::mul(addend, const_1)
                );

                extension_field cons = extension_field::sub(output, computed_output);
                mul_pow_add(cons[0], acc, alphas, alpha_pows);
                mul_pow_add(cons[1], acc, alphas, alpha_pows);
            }
            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void mul_extension_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_ops = 13; // TODO

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            base_field const_0 = load_from_either(consts_lde_start, consts_start, num_wires_ldes, lde_degree_bits, degree_bits, num_selectors);
            // base_field const_0 = memory::load_cs(constants_ptr + index + num_selectors * nrows);

            for(unsigned i = 0; i < num_ops; i++){
                base_field m_00 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i);
                base_field m_01 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i + 1);
                // base_field m_00 = memory::load_cs(wires_ptr + index + (6 * i) * nrows);
                // base_field m_01 = memory::load_cs(wires_ptr + index + (6 * i + 1) * nrows);
                extension_field multiplicand_0 = {m_00, m_01};

                base_field m_10 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i + 2);
                base_field m_11 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i + 3);
                // base_field m_10 = memory::load_cs(wires_ptr + index + (6 * i + 2) * nrows);
                // base_field m_11 = memory::load_cs(wires_ptr + index + (6 * i + 3) * nrows);
                extension_field multiplicand_1 = {m_10, m_11};

                base_field o_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i + 4);
                base_field o_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 * i + 5);
                // base_field o_0 = memory::load_cs(wires_ptr + index + (6 * i + 4) * nrows);
                // base_field o_1 = memory::load_cs(wires_ptr + index + (6 * i + 5) * nrows);
                extension_field output = {o_0, o_1};

                extension_field computed_output = extension_field::mul(
                        extension_field::mul(multiplicand_0, multiplicand_1),
                        const_0
                    );

                extension_field cons = extension_field::sub(output, computed_output);
                mul_pow_add(cons[0], acc, alphas, alpha_pows);
                mul_pow_add(cons[1], acc, alphas, alpha_pows);
            }
            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void reducing_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_coeffs = 43; // TODO

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            base_field a_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 2);
            base_field a_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 3);
            // base_field a_0 = memory::load_cs(wires_ptr + index + 2 * nrows);
            // base_field a_1 = memory::load_cs(wires_ptr + index + 3 * nrows);
            extension_field alpha = {a_0, a_1};

            base_field o_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4);
            base_field o_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 5);
            // base_field o_0 = memory::load_cs(wires_ptr + index + 4 * nrows);
            // base_field o_1 = memory::load_cs(wires_ptr + index + 5 * nrows);
            extension_field old_acc = {o_0, o_1};

            extension_field acc2 = old_acc;
            for (unsigned i = 0; i < num_coeffs - 1; i++) {
                base_field coeff_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 + i);
                // base_field coeff_0 = memory::load_cs(wires_ptr + index + (6 + i) * nrows);
                extension_field coeff = {coeff_0, 0};

                base_field accs_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 + num_coeffs + 2 * i);
                base_field accs_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 + num_coeffs + 2 * i + 1);
                // base_field accs_0 = memory::load_cs(wires_ptr + index + (6 + num_coeffs + 2 * i) * nrows);
                // base_field accs_1 = memory::load_cs(wires_ptr + index + (6 + num_coeffs + 2 * i + 1) * nrows);
                extension_field accs = {accs_0, accs_1};
                extension_field cons = extension_field::sub(
                    extension_field::add(
                        extension_field::mul(acc2, alpha),
                        coeff
                    ),
                    accs
                );
                mul_pow_add(cons[0], acc, alphas, alpha_pows);
                mul_pow_add(cons[1], acc, alphas, alpha_pows);
                acc2 = accs;
            }
            base_field coeff_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 5 + num_coeffs);
            // base_field coeff_0 = memory::load_cs(wires_ptr + index + (5 + num_coeffs) * nrows);
            extension_field coeff = {coeff_0, 0};
            base_field accs_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 0);
            base_field accs_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 1);
            // base_field accs_0 = memory::load_cs(wires_ptr + index);
            // base_field accs_1 = memory::load_cs(wires_ptr + index + nrows);
            extension_field accs = {accs_0, accs_1};
            extension_field cons = extension_field::sub(
                extension_field::add(
                    extension_field::mul(acc2, alpha),
                    coeff
                ),
                accs
            );
            mul_pow_add(cons[0], acc, alphas, alpha_pows);
            mul_pow_add(cons[1], acc, alphas, alpha_pows);

            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void reducing_extension_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned num_coeffs = 32; // TODO

    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            base_field a_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 2);
            base_field a_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 3);
            // base_field a_0 = memory::load_cs(wires_ptr + index + 2 * nrows);
            // base_field a_1 = memory::load_cs(wires_ptr + index + 3 * nrows);
            extension_field alpha = {a_0, a_1};

            base_field o_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4);
            base_field o_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 5);
            // base_field o_0 = memory::load_cs(wires_ptr + index + 4 * nrows);
            // base_field o_1 = memory::load_cs(wires_ptr + index + 5 * nrows);
            extension_field old_acc = {o_0, o_1};

            extension_field acc2 = old_acc;
            for (unsigned i = 0; i < num_coeffs - 1; i++) {
                base_field coeff_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 + 2 * i);
                base_field coeff_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 + 2 * i + 1);
                // base_field coeff_0 = memory::load_cs(wires_ptr + index + (6 + 2 * i) * nrows);
                // base_field coeff_1 = memory::load_cs(wires_ptr + index + (6 + 2 * i + 1) * nrows);
                extension_field coeff = {coeff_0, coeff_1};

                base_field accs_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 + num_coeffs + 2 * i);
                base_field accs_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 6 + num_coeffs + 2 * i + 1);
                // base_field accs_0 = memory::load_cs(wires_ptr + index + (6 + num_coeffs + 2 * i) * nrows);
                // base_field accs_1 = memory::load_cs(wires_ptr + index + (6 + num_coeffs + 2 * i + 1) * nrows);
                extension_field accs = {accs_0, accs_1};
                extension_field cons = extension_field::sub(
                    extension_field::add(
                        extension_field::mul(acc2, alpha),
                        coeff
                    ),
                    accs
                );
                mul_pow_add(cons[0], acc, alphas, alpha_pows);
                mul_pow_add(cons[1], acc, alphas, alpha_pows);
                acc2 = accs;
            }
            base_field coeff_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 4 + 2 * num_coeffs);
            base_field coeff_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 5 + 2 * num_coeffs);
            // base_field coeff_0 = memory::load_cs(wires_ptr + index + (4 + 2 * num_coeffs) * nrows);
            // base_field coeff_1 = memory::load_cs(wires_ptr + index + (5 + 2 * num_coeffs) * nrows);
            extension_field coeff = {coeff_0, coeff_1};
            base_field accs_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 0);
            base_field accs_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 1);
            // base_field accs_0 = memory::load_cs(wires_ptr + index);
            // base_field accs_1 = memory::load_cs(wires_ptr + index + nrows);
            extension_field accs = {accs_0, accs_1};
            extension_field cons = extension_field::sub(
                extension_field::add(
                    extension_field::mul(acc2, alpha),
                    coeff
                ),
                accs
            );
            mul_pow_add(cons[0], acc, alphas, alpha_pows);
            mul_pow_add(cons[1], acc, alphas, alpha_pows);

            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

__constant__ constexpr uint64_t MDS_MATRIX_CIRC[12] = {17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20};
__constant__ constexpr uint64_t MDS_MATRIX_DIAG[12] = {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

extern "C" __launch_bounds__(128, 8) __global__
    void poseidon_mds_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            extension_field inputs[12];
            for (int i = 0; i < 12; i++) {
                base_field input_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 2 * i);
                base_field input_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 2 * i + 1);
                // base_field input_0 = memory::load_cs(wires_ptr + index + 2 * i * nrows);
                // base_field input_1 = memory::load_cs(wires_ptr + index + (2 * i + 1) * nrows);
                inputs[i] = {input_0, input_1};
            }

            extension_field computed_outputs[12];
            for (int r = 0; r < 12; r++) {
                computed_outputs[r] = extension_field::zero();
                for (int i = 0; i < 12; i++) {
                    extension_field mds_matrix_circ = {MDS_MATRIX_CIRC[i], 0};
                    computed_outputs[r] = extension_field::add(
                        computed_outputs[r],
                        extension_field::mul(inputs[(i + r) % 12], mds_matrix_circ)
                    );
                }
                extension_field mds_matrix_diag = {MDS_MATRIX_DIAG[r], 0};
                computed_outputs[r] = extension_field::add(
                    computed_outputs[r],
                    extension_field::mul(inputs[r], mds_matrix_diag)
                );
            }

            for (int i = 0; i < 12; i++) {
                base_field output_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 24 + 2 * i);
                base_field output_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 24 + 2 * i + 1);
                // base_field output_0 = memory::load_cs(wires_ptr + index + (24 + 2 * i) * nrows);
                // base_field output_1 = memory::load_cs(wires_ptr + index + (24 + 2 * i + 1) * nrows);
                extension_field output = {output_0, output_1};
                extension_field cons = extension_field::sub(output, computed_outputs[i]);
                mul_pow_add(cons[0], acc, alphas, alpha_pows);
                mul_pow_add(cons[1], acc, alphas, alpha_pows);
            }

            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}

extern "C" __launch_bounds__(128, 8) __global__
    void coset_interpolation_gate_partial(
        const base_field *ldes_ptr,
        const base_field *polys_ptr,
        const base_field *public_inputs_hash,
        base_field *output,
        const base_field *alphas,
        const unsigned degree_bits,
        const unsigned rate_bits,
        const unsigned num_wires,
        const unsigned num_wires_ldes,
        const unsigned num_const_sigmas,
        const unsigned num_const_sigmas_ldes,
        const unsigned part,
        const unsigned num_gate_constraints,
        const unsigned num_selectors,
        const unsigned num_constants,
        const unsigned param1,
        const unsigned param2,
        const unsigned param3,
        const unsigned row,
        const unsigned selector_index,
        const unsigned start,
        const unsigned end
) {
    const unsigned g_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned nrows = 1 << degree_bits;
    const unsigned lde_degree_bits = degree_bits + rate_bits;
    const unsigned row_step = nrows / TILE_Q;
    const unsigned rev_i = reverse_bits(part, rate_bits);
    const unsigned subgroup_bits = 4; // TODO
    const unsigned degree = 6; // TODO

// CosetInterpolationGate { subgroup_bits: 4, degree: 6, barycentric_weights: [17293822565076172801, 18374686475376656385, 18446744069413535745, 281474976645120, 17592186044416, 256, 18446744000695107601, 18446744065119617025, 1152921504338411520, 72057594037927936, 1048576, 18446462594437939201, 18446726477228539905, 18446744069414584065, 68719476720, 4294967296], _phantom: PhantomData<plonky2_field::goldilocks_field::GoldilocksField> }<D=2>
    if (g_id < row_step) {
        for (unsigned t = 0; t < TILE_Q; t++) {
            const unsigned index = g_id + t * row_step;
            const unsigned index_lde = index + (rev_i << degree_bits);
            const base_field *consts_start = polys_ptr + index;
            const base_field *consts_lde_start = ldes_ptr + index_lde;
            const base_field *wires_start = polys_ptr + index + (num_const_sigmas << degree_bits);
            const base_field *wires_lde_start = ldes_ptr + index_lde + (num_const_sigmas_ldes << lde_degree_bits);

            base_field filter = compute_filter_partial(consts_lde_start, consts_start, num_const_sigmas_ldes, row, selector_index, start, end, lde_degree_bits, degree_bits);
            base_field acc[N_CLGS] = {{0x0, 0x0}, {0x0, 0x0}};
            base_field alpha_pows[N_CLGS] = {{0x1, 0x0}, {0x1, 0x0}};

            base_field shift = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 0);
            base_field evaluation_point_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 33);
            base_field evaluation_point_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 34);
            // base_field shift = memory::load_cs(wires_ptr + index);
            // base_field evaluation_point_0 = memory::load_cs(wires_ptr + index + 33 * nrows);
            // base_field evaluation_point_1 = memory::load_cs(wires_ptr + index + 34 * nrows);
            extension_field evaluation_point = {evaluation_point_0, evaluation_point_1};

            base_field shifted_evaluation_point_0 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 185);
            base_field shifted_evaluation_point_1 = load_from_either(wires_lde_start, wires_start, num_wires_ldes, lde_degree_bits, degree_bits, 186);
            // base_field shifted_evaluation_point_0 = memory::load_cs(wires_ptr + index + 185 * nrows);
            // base_field shifted_evaluation_point_1 = memory::load_cs(wires_ptr + index + 186 * nrows);
            extension_field shifted_evaluation_point = {shifted_evaluation_point_0, shifted_evaluation_point_1};

            extension_field cons = extension_field::sub(
                evaluation_point,
                extension_field::mul(
                    shifted_evaluation_point,
                    extension_field{shift, 0}
                )
            );
            mul_pow_add(cons[0], acc, alphas, alpha_pows);
            mul_pow_add(cons[1], acc, alphas, alpha_pows);

            #pragma unroll
            for (unsigned k = 0; k < N_CLGS; k++) {
                base_field quotient_acc = output[index + (k << lde_degree_bits)];
                output[index + (k << lde_degree_bits)] = filter * acc[k] + quotient_acc;
            }
        }
    }
}
