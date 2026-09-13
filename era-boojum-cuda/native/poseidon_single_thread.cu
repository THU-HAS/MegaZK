#include "goldilocks.cuh"

#if __CUDA_ARCH__ == 800
#define USE_SHARED_MEMORY
#endif
#include "poseidon_single_thread.cuh"
#if __CUDA_ARCH__ == 800
#undef USE_SHARED_MEMORY
#endif

namespace poseidon {

using namespace goldilocks;

static DEVICE_FORCEINLINE void permutation_old(poseidon_state &state) {
#pragma unroll 1
  for (unsigned round = 0; round < TOTAL_NUM_ROUNDS; round++) {
    if (round < HALF_NUM_FULL_ROUNDS || round >= HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS) {
      if (round != HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS)
        apply_round_constants(state, round);
      apply_non_linearity(state);
      if (round == HALF_NUM_FULL_ROUNDS - 1) {
        apply_fused_round_constants(state);
        full_and_partial_round_fused_mul(state);
      } else {
        apply_mds_matrix(state);
      }
    } else {
      partial_round_optimized_old(state, round);
    }
  }
}

// static DEVICE_FORCEINLINE void display(poseidon_state &state) {
//   // printf("%s", info);
// #pragma unroll 1
//   for (unsigned i = 0; i < 12; i++) {
//     field<2> value = field<3>::field3_to_field2(state[i]);
//     printf("(0x%x%x), ", value[1], value[0]);
//   }
//   printf("\n");
// }

static DEVICE_FORCEINLINE void permutation(poseidon_state &state) {
  // printf("start: ");
  // display(state);
#pragma unroll 1
  for (unsigned round = 0; round < TOTAL_NUM_ROUNDS; round++) {
    if (round < HALF_NUM_FULL_ROUNDS || round >= HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS) {
      apply_round_constants(state, round);
      // printf("constants: ");
      // display(state);
      apply_non_linearity(state);
      // printf("s-box: ");
      // display(state);
      apply_mds_matrix_new(state);
      // printf("mds: ");
      // display(state);
    } else {
      if (round == HALF_NUM_FULL_ROUNDS) {
        partial_rounds_init(state);
        // printf("patial rounds init: ");
        // display(state);
      }
      partial_round_optimized(state, round);
      // printf("patial rounds: ");
      // display(state);
    }
  }
}

#if __CUDA_ARCH__ == 800
#define MIN_BLOCKS_COUNT 10
#else
#define MIN_BLOCKS_COUNT 12
#endif
EXTERN __launch_bounds__(64, MIN_BLOCKS_COUNT) __global__
    void poseidon_single_thread_leaves_kernel(const base_field *values, base_field *results, const unsigned rows_count, const unsigned cols_count,
                                              const unsigned count, bool load_intermediate, bool store_intermediate) {
  single_thread_leaves_impl<permutation>(values, results, rows_count, cols_count, count, load_intermediate, store_intermediate);
}

EXTERN __launch_bounds__(64, MIN_BLOCKS_COUNT) __global__
    void poseidon_single_thread_leaves_partial_kernel(const base_field *static_input, const base_field *dynamic_input, base_field *output, 
      const unsigned log_n, const unsigned rate_bits, const unsigned num_ntts, const unsigned num_ldes, const unsigned part) {
  poseidon_common::single_thread_leaves_partial_impl<permutation>(static_input, dynamic_input, output, log_n, rate_bits, num_ntts, num_ldes, part);
}
#undef MIN_BLOCKS_COUNT

EXTERN __launch_bounds__(64, 12) __global__ void poseidon_single_thread_nodes_kernel(const base_field *values, base_field *results, const unsigned count) {
  static_assert(RATE == 2 * CAPACITY);
  single_thread_nodes_impl<permutation>(values, results, count);
}

} // namespace poseidon
