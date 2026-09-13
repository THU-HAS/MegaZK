#include "generate.cuh"

extern "C" __global__ void arithmetic_base_generator_kernel(
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
        size_t index_start = u_params[u_start + tid];

        base_field multiplicand_0 = witness[representative_map[index_start]];
        base_field multiplicand_1 = witness[representative_map[index_start + 1]];
        base_field addend = witness[representative_map[index_start + 2]];

        base_field computed_output =
            base_field::add(
                base_field::mul(
                    base_field::mul(multiplicand_0, multiplicand_1),
                    f_params[f_start + 2 * tid]
                ),
                base_field::mul(addend, f_params[f_start + 2 * tid + 1])
            );
        witness[representative_map[index_start + 3]] = computed_output;
    }
}

extern "C" __global__ void random_value_generator_kernel(
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
        // base_field random_value = base_field::rand();
        base_field random_value = base_field::zero();
        witness[representative_map[u_params[u_start + tid]]] = random_value;
    }
}

extern "C" __global__ void constant_generator_kernel(
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
        witness[representative_map[u_params[u_start + tid]]] = f_params[f_start + tid];
    }
}

extern "C" __global__ void u32_add_many_generator_kernel(
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
        size_t index_start = u_params[u_start + 4 * tid];
        size_t i = u_params[u_start + 4 * tid + 1];
        size_t num_addends = u_params[u_start +  4 * tid + 2];
        size_t num_ops = u_params[u_start +  4 * tid + 3];

        base_field addend;
        base_field output = base_field::zero();
        base_field carry = witness[representative_map[index_start + (num_addends + 3) * i + num_addends]];

        for (int j = 0; j < num_addends; j++) {
            addend = witness[representative_map[index_start + (num_addends + 3) * i + j]];
            output = base_field::add(output, addend);
        }

        output = base_field::add(output, carry);

        uint64_t output_u64 = base_field::to_canonical_u64(output);

        uint64_t output_carry_u64 = output_u64 >> 32;
        uint64_t output_result_u64 = output_u64 & ((uint64_t(1) << 32) - 1);

        base_field output_carry = base_field::from_u64(output_carry_u64);
        base_field output_result = base_field::from_u64(output_result_u64);

        witness[representative_map[index_start + (num_addends + 3) * i + num_addends + 2]] = output_carry;
        witness[representative_map[index_start + (num_addends + 3) * i + num_addends + 1]] = output_result;

        size_t num_result_limbs = (32 + 2 - 1) / 2; // 16
        size_t num_carry_limbs = (4 + 2 - 1) / 2;   // 2
        uint64_t limb_base = uint64_t(1) << 2;      // 4

        for (int j = 0; j < num_result_limbs; j++) {
            base_field ret = base_field::from_u64(output_result_u64 % limb_base);
            witness[representative_map[index_start + (num_addends + 3) * num_ops + (num_result_limbs + num_carry_limbs) * i + j]] = ret;
            output_result_u64 /= limb_base;
        }

        for (int j = 0; j < num_carry_limbs; j++) {
            base_field ret = base_field::from_u64(output_carry_u64 % limb_base);
            witness[representative_map[index_start + (num_addends + 3) * num_ops + (num_result_limbs + num_carry_limbs) * i + num_result_limbs + j]] = ret;
            output_carry_u64 /= limb_base;
        }
    }
}

extern "C" __global__ void u32_arithmetic_generator_kernel(
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
        size_t index_start = u_params[u_start + 3 * tid];
        size_t i = u_params[u_start + 3 * tid + 1];
        size_t num_ops = u_params[u_start + 3 * tid + 2];

        base_field multiplicand_0 = witness[representative_map[index_start + 6 * i]];
        base_field multiplicand_1 = witness[representative_map[index_start + 6 * i + 1]];
        base_field addend = witness[representative_map[index_start + 6 * i + 2]];

        base_field output = base_field::add(base_field::mul(multiplicand_0, multiplicand_1), addend);
        uint64_t output_u64 = base_field::to_canonical_u64(output);

        uint64_t output_high_u64 = output_u64 >> 32;
        uint64_t output_low_u64 = output_u64 & ((uint64_t(1) << 32) - 1);

        base_field output_high = base_field::from_u64(output_high_u64);
        base_field output_low = base_field::from_u64(output_low_u64);

        witness[representative_map[index_start + 6 * i + 4]] = output_high;
        witness[representative_map[index_start + 6 * i + 3]] = output_low;

        uint64_t diff = uint64_t(UINT_MAX) - output_high_u64;

        base_field inverse = (diff == 0) ? base_field::zero() : base_field::inv(base_field::from_u64(diff));

        witness[representative_map[index_start + 6 * i + 5]] = inverse;

        int num_limbs = 32;
        uint64_t limb_base = uint64_t(1) << 2;

        for (int j = 0; j < num_limbs; j++) {
            base_field ret = base_field::from_u64(output_u64 % limb_base);
            witness[representative_map[index_start + 6 * num_ops + num_limbs * i + j]] = ret;
            output_u64 /= limb_base;
        }
    }
}

extern "C" __global__ void comparison_generator_kernel(
    base_field *witness,
    const unsigned *representative_map,
    const size_t *u_params,
    const base_field *f_params,
    const unsigned u_start,
    const unsigned f_start,
    const unsigned length
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        size_t index_start = u_params[u_start + 3 * tid];
        size_t num_bits = u_params[u_start + 3 * tid + 1];
        size_t num_chunks = u_params[u_start + 3 * tid + 2];

        base_field first_input = witness[representative_map[index_start]];
        base_field second_input = witness[representative_map[index_start + 1]];

        uint64_t first_input_u64 = base_field::to_canonical_u64(first_input);
        uint64_t second_input_u64 = base_field::to_canonical_u64(second_input);
        base_field result = base_field::from_u64(first_input_u64 <= second_input_u64);

        witness[representative_map[index_start + 2]] = result;

        uint64_t chunk_bits = (num_bits + num_chunks - 1) / num_chunks;
        uint64_t chunk_size = uint64_t(1) << chunk_bits;
        base_field most_significant_diff_so_far = base_field::zero();
        base_field chunks_equal, equality_dummie, intermediate_value;

        for (int i = 0; i < num_chunks; i++) {
            uint64_t first_input_chunk = (first_input_u64 % chunk_size);
            first_input_u64 /= chunk_size;

            uint64_t second_input_chunk = (second_input_u64 % chunk_size);
            second_input_u64 /= chunk_size;

            bool is_equal = (first_input_chunk != second_input_chunk);
            if (is_equal) {
                most_significant_diff_so_far = base_field::sub(
                    base_field::from_u64(second_input_chunk),
                    base_field::from_u64(first_input_chunk)
                );
            }

            intermediate_value = is_equal ? base_field::zero() : most_significant_diff_so_far;
            chunks_equal = is_equal ? base_field::zero() : base_field::one();
            equality_dummie = is_equal ? base_field::inv(
                base_field::sub(
                    base_field::from_u64(second_input_chunk),
                    base_field::from_u64(first_input_chunk)
                )
            ) : base_field::one();


            witness[representative_map[index_start + 4 + i]] = base_field::from_u64(first_input_chunk);
            witness[representative_map[index_start + 4 + num_chunks + i]] = base_field::from_u64(second_input_chunk);
            witness[representative_map[index_start + 4 + 2 * num_chunks + i]] = equality_dummie;
            witness[representative_map[index_start + 4 + 3 * num_chunks + i]] = chunks_equal;
            witness[representative_map[index_start + 4 + 4 * num_chunks + i]] = intermediate_value;
        }

        base_field most_significant_diff = most_significant_diff_so_far;
        witness[representative_map[index_start + 3]] = most_significant_diff;

        base_field two_n = base_field::from_u64(uint64_t(1) << chunk_bits);
        uint64_t two_n_plus_msd = base_field::to_canonical_u64(base_field::add(two_n, most_significant_diff));

        for (int i = 0; i < chunk_bits + 1; i++) {
            uint64_t msd_bit_u64 = two_n_plus_msd % 2;
            two_n_plus_msd /= 2;

            base_field msd_bit = base_field::from_u64(msd_bit_u64);
            witness[representative_map[index_start + 4 + 5 * num_chunks + i]] = msd_bit;
        }
    }
}

extern "C" __global__ void u32_interleave_generator_kernel(
    base_field *witness,
    const unsigned *representative_map,
    const size_t *u_params,
    const base_field *f_params,
    const unsigned u_start,
    const unsigned f_start,
    const unsigned length
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        size_t index_start = u_params[u_start + 3 * tid];
        size_t i = u_params[u_start + 3 * tid + 1];
        size_t num_ops = u_params[u_start + 3 * tid + 2];

        base_field x = witness[representative_map[index_start + 2 * i]];
        uint64_t x_interleaved = 0;

        size_t num_bits = 32;
        size_t begin = num_ops * 2;

        for (int j = 0; j < num_bits; j++) {
            size_t bit_wire_index = begin + num_bits * i + j;
            uint64_t x_u64 = base_field::to_canonical_u64(x);
            uint64_t bit = (x_u64 >> (num_bits - j - 1)) % 2;
            witness[representative_map[index_start + bit_wire_index]] = base_field::from_u64(bit);
            x_interleaved += bit * (uint64_t(1) << (2 * (num_bits - j - 1)));
        }

        witness[representative_map[index_start + 2 * i + 1]] = base_field::from_u64(x_interleaved);
    }
}

extern "C" __global__ void u32_range_check_generator_kernel(
    base_field *witness,
    const unsigned *representative_map,
    const size_t *u_params,
    const base_field *f_params,
    const unsigned u_start,
    const unsigned f_start,
    const unsigned length
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        size_t index_start = u_params[u_start + 2 * tid];
        size_t num_input_limbs = u_params[u_start + 2 * tid + 1];

        for (int i = 0; i < num_input_limbs; i++) {
            uint64_t sum_value_u64 = base_field::to_canonical_u64(witness[representative_map[index_start + i]]);
            uint32_t sum_value = uint32_t(sum_value_u64);
            uint32_t base = uint32_t(1) << 2;

            size_t aux_limbs_per_input_limb = (32 + 2 - 1) / 2;

            for (int j = 0; j < aux_limbs_per_input_limb; j++) {
                uint32_t limb_value = sum_value % base;
                sum_value /= base;
                witness[representative_map[index_start + num_input_limbs + aux_limbs_per_input_limb * i + j]] = base_field::from_u64(uint64_t(limb_value));
            }
        }
    }
}

extern "C" __global__ void u32_subtraction_generator_kernel(
    base_field *witness,
    const unsigned *representative_map,
    const size_t *u_params,
    const base_field *f_params,
    const unsigned u_start,
    const unsigned f_start,
    const unsigned length
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        size_t index_start = u_params[u_start + 3 * tid];
        size_t i = u_params[u_start + 3 * tid + 1];
        size_t num_ops = u_params[u_start + 3 * tid + 2];

        base_field input_x = witness[representative_map[index_start + 5 * i]];
        base_field input_y = witness[representative_map[index_start + 5 * i + 1]];
        base_field input_borrow = witness[representative_map[index_start + 5 * i + 2]];

        base_field result_initial = base_field::sub(base_field::sub(input_x, input_y), input_borrow);
        uint64_t result_initial_u64 = base_field::to_canonical_u64(result_initial);
        base_field output_borrow = result_initial_u64 > uint64_t(1) << 32 ? base_field::one() : base_field::zero();

        base_field base = base_field::from_u64(uint64_t(1) << 32);
        base_field output_result = base_field::add(result_initial, base_field::mul(base, output_borrow));

        witness[representative_map[index_start + 5 * i + 3]] = output_result;
        witness[representative_map[index_start + 5 * i + 4]] = output_borrow;

        uint64_t output_result_u64 = base_field::to_canonical_u64(output_result);

        size_t num_limbs = 32 / 2;
        uint64_t limb_base = uint64_t(1) << 2;

        for (int j = 0; j < num_limbs; j++) {
            uint64_t output_limb = output_result_u64 % limb_base;
            output_result_u64 /= limb_base;

            witness[representative_map[index_start + 5 * num_ops + num_limbs * i + j]] = base_field::from_u64(output_limb);
        }
    }
}

extern "C" __global__ void uninterleave_to_u32_generator_kernel(
    base_field *witness,
    const unsigned *representative_map,
    const size_t *u_params,
    const base_field *f_params,
    const unsigned u_start,
    const unsigned f_start,
    const unsigned length
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        size_t index_start = u_params[u_start + 3 * tid];
        size_t i = u_params[u_start + 3 * tid + 1];
        size_t num_ops = u_params[u_start + 3 * tid + 2];

        uint64_t x_interleaved = base_field::to_canonical_u64(witness[representative_map[index_start + 3 * i]]);

        uint64_t x_evens = 0;
        uint64_t x_odds = 0;

        size_t num_bits = 64;
        size_t start = num_ops * 3;

        for (int j = 0; j < num_bits / 2; j++) {
            size_t shift = 2 * (num_bits / 2 - j - 1);
            uint64_t jth_even = (x_interleaved >> (shift + 1)) % 2;
            uint64_t jth_odd = (x_interleaved >> shift) % 2;

            witness[representative_map[index_start + 2 * j + start + num_bits * i ]] = base_field::from_u64(jth_even);
            witness[representative_map[index_start + 2 * j + 1 + start + num_bits * i]] = base_field::from_u64(jth_odd);

            uint64_t coeff = uint64_t(1) << (num_bits / 2 - j - 1);
            x_evens += jth_even * coeff;
            x_odds += jth_odd * coeff;
        }

        witness[representative_map[index_start + 3 * i + 1]] = base_field::from_u64(x_evens);
        witness[representative_map[index_start + 3 * i + 2]] = base_field::from_u64(x_odds);
    }
}

extern "C" __global__ void base_sum_generator_kernel(
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
        size_t start = u_start;
        for (int i = 0; i < tid; i++) {
            start += 3 + u_params[start + 2];
        }

        size_t index_start = u_params[start];
        size_t B = u_params[start + 1];
        size_t num_limbs = u_params[start + 2];

        base_field sum = base_field::zero();
        for(int i = num_limbs - 1; i >= 0; i--) {
            base_field limb = witness[representative_map[u_params[start + 3 + i]]];
            sum = base_field::add(base_field::mul(sum, base_field::from_u64(B)), limb);
        }
        witness[representative_map[index_start]] = sum;
    }
}

extern "C" __global__ void base_split_generator_kernel(
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
        size_t index_start = u_params[u_start + 3 * tid];
        size_t B = u_params[u_start + 3 * tid + 1];
        size_t num_limbs = u_params[u_start + 3 * tid + 2];

        uint64_t sum_value_u64 = base_field::to_canonical_u64(witness[representative_map[index_start]]);
        size_t sum_value = size_t(sum_value_u64);

        for(int i = 0; i < num_limbs; i++) {
            size_t limb_value = sum_value % B;
            sum_value /= B;
            witness[representative_map[index_start + 1 + i]] = base_field::from_u64(limb_value);
        }
    }
}

extern "C" __global__ void random_access_generator_kernel(
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
        size_t index_start = u_params[u_start + 5 * tid];
        size_t copy = u_params[u_start + 5 * tid + 1];
        size_t bits = u_params[u_start + 5 * tid + 2];
        size_t num_copies = u_params[u_start + 5 * tid + 3];
        size_t num_extra_constants = u_params[u_start + 5 * tid + 4];

        size_t vec_size = size_t(1) << bits;
        base_field access_index_f = witness[representative_map[index_start + (2 + vec_size) * copy]];
        uint64_t access_index = base_field::to_canonical_u64(access_index_f);

        witness[representative_map[index_start + (2 + vec_size) * copy + 1]] = witness[representative_map[index_start + (2 + vec_size) * copy + 2 + access_index]];

        for(int i = 0; i < bits; i++) {
            base_field bit = base_field::from_u64(((access_index >> i) & 1) != 0);
            witness[representative_map[index_start + (2 + vec_size) * num_copies + num_extra_constants + copy * bits + i]] = bit;
        }
    }
}

extern "C" __global__ void equality_generator_kernel(
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
        base_field x = witness[representative_map[u_params[u_start + 4 * tid]]];
        base_field y = witness[representative_map[u_params[u_start + 4 * tid + 1]]];

        uint64_t x_u64 = base_field::to_canonical_u64(x);
        uint64_t y_u64 = base_field::to_canonical_u64(y);

        bool is_equal = (x_u64 == y_u64);
        base_field diff = base_field::sub(x, y);
        base_field inv = is_equal ? base_field::zero() : base_field::inv(diff);

        witness[representative_map[u_params[u_start + 4 * tid + 2]]] = base_field::from_u64(is_equal ? 1 : 0);
        witness[representative_map[u_params[u_start + 4 * tid + 3]]] = inv;
    }
}

extern "C" __global__ void curve_point_decompression_generator_kernel(
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
        uint8_t bytes[32];

        for(int byte_idx = 0; byte_idx < 32; byte_idx++){
            uint8_t byte = 0;
            for(int bit_idx = 0; bit_idx < 8; bit_idx++){
                uint64_t bit = base_field::to_canonical_u64(witness[representative_map[u_params[u_start + 272 * tid + 8 * byte_idx + bit_idx]]]);
                byte |= (bit != 0) << (7 - bit_idx);
            }

            bytes[31 - byte_idx] = byte;
        }

        FieldElement51 Y = FieldElement51::from_bytes(bytes);

        FieldElement51 Z = FieldElement51::one();
        FieldElement51 YY = Y.square();
        FieldElement51 u = YY - Z;
        FieldElement51 v = YY * FieldElement51::EDWARDS_D() + Z;
        FieldElement51 X = FieldElement51::sqrt_ratio_i(u, v);

        bool negate = (bytes[31] >> 7) != 0;
        X = negate ? (-X) : X;

        uint8_t s[32];
        uint8_t t[32];
        X.as_bytes(s);
        Y.as_bytes(t);

        BigUint x = BigUint::from_bytes_le(s);
        BigUint y = BigUint::from_bytes_le(t);

        x = x % Ed25519Base::order();
        y = y % Ed25519Base::order();

        for (int i = 0; i < 8; i++) {
            witness[representative_map[u_params[u_start + 272 * tid + 256 + i]]] = base_field::from_u64(x.digits[i]);
            witness[representative_map[u_params[u_start + 272 * tid + 264 + i]]] = base_field::from_u64(y.digits[i]);
        }
    }
}

extern "C" __global__ void non_native_addition_generator_kernel(
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
        size_t start = u_start;
        for (int i = 0; i < tid; i++) {
            start += 12 + u_params[start + 1] + u_params[start + 2];
        }

        size_t type = u_params[start];
        size_t num_a_target = u_params[start + 1];
        size_t num_b_target = u_params[start + 2];

        BigUint a = BigUint::zero();
        BigUint b = BigUint::zero();

        for(int i = (int)num_a_target - 1; i >= 0; i--) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[representative_map[u_params[start + 3 + i]]]);
            BigUint temp = BigUint(temp_u64);
            a = (a << 32) + temp;
        }

        for(int i = (int)num_b_target - 1; i >= 0; i--) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[representative_map[u_params[start + 3 + num_a_target + i]]]);
            BigUint temp = BigUint(temp_u64);
            b = (b << 32) + temp;
        }

        // Ed25519Base a_ed = Ed25519Base::from_noncanonical_biguint(a);
        // Ed25519Base b_ed = Ed25519Base::from_noncanonical_biguint(b);

        // a = a_ed.to_canonical_biguint();
        // b = b_ed.to_canonical_biguint();

        BigUint orders[5];
        // orders[0] = goldilocks::order();
        orders[1] = Ed25519Base::order();
        // orders[2] = Ed25519Scalar::order();
        orders[3] = Secp256K1Base::order();
        orders[4] = Secp256K1Scalar::order();

        bool mod_table[5] = {false, true, true, false, false};

        BigUint order = orders[type];
        bool need_mod = mod_table[type];

        BigUint a_temp = (a >= order) ? (a - order) : a;
        BigUint b_temp = (b >= order) ? (b - order) : b;

        a = need_mod ? (a_temp % order) : a_temp;
        b = need_mod ? (b_temp % order) : b_temp;

        BigUint sum = a + b;

        bool overflow = sum > order;
        if (overflow) {
            sum = sum - order;
        }

        for (int i = 0; i < 8; i++) {
            witness[representative_map[u_params[start + 3 + num_a_target + num_b_target + i]]] = base_field::from_u64(sum.digits[i]);
        }
        witness[representative_map[u_params[start + 3 + num_a_target + num_b_target + 8]]] = base_field::from_u64(overflow ? 1 : 0);
    }
}

extern "C" __global__ void non_native_subtraction_generator_kernel(
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
        size_t start = u_start;
        for (int i = 0; i < tid; i++) {
            start += 12 + u_params[start + 1] + u_params[start + 2];
        }

        size_t type = u_params[start];
        size_t num_a_target = u_params[start + 1];
        size_t num_b_target = u_params[start + 2];

        BigUint a = BigUint::zero();
        BigUint b = BigUint::zero();

        for(int i = (int)num_a_target - 1; i >= 0; i--) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[representative_map[u_params[start + 3 + i]]]);
            BigUint temp = BigUint(temp_u64);
            a = (a << 32) + temp;
        }

        for(int i = (int)num_b_target - 1; i >= 0; i--) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[representative_map[u_params[start + 3 + num_a_target + i]]]);
            BigUint temp = BigUint(temp_u64);
            b = (b << 32) + temp;
        }

        BigUint orders[5];
        orders[1] = Ed25519Base::order();
        orders[3] = Secp256K1Base::order();
        orders[4] = Secp256K1Scalar::order();

        bool mod_table[5] = {false, true, true, false, false};

        BigUint order = orders[type];
        bool need_mod = mod_table[type];

        BigUint a_temp = (a >= order) ? (a - order) : a;
        BigUint b_temp = (b >= order) ? (b - order) : b;

        a = need_mod ? (a_temp % order) : a_temp;
        b = need_mod ? (b_temp % order) : b_temp;

        bool less = a < b;

        if (less) {
            a = a + order;
        }

        BigUint diff = a - b;
        for (int i = 0; i < 8; i++) {
            witness[representative_map[u_params[start + 3 + num_a_target + num_b_target + i]]] = base_field::from_u64(diff.digits[i]);
        }
        witness[representative_map[u_params[start + 3 + num_a_target + num_b_target + 8]]] = base_field::from_u64(less ? 1 : 0);
    }
}

extern "C" __global__ void non_native_multiplication_generator_kernel(
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
        size_t start = u_start;
        for (int i = 0; i < tid; i++) {
            start += 19 + u_params[start + 1] + u_params[start + 2];
        }

        size_t type = u_params[start];
        size_t num_a_target = u_params[start + 1];
        size_t num_overflow_target = u_params[start + 2];

        BigUint a = BigUint::zero();
        BigUint b = BigUint::zero();

        for(int i = (int)num_a_target - 1; i >= 0; i--) {
            uint64_t temp_a_u64 = base_field::to_canonical_u64(witness[representative_map[u_params[start + 3 + i]]]);
            BigUint temp_a = BigUint(temp_a_u64);
            a = (a << 32) + temp_a;
        }

        for(int i = 7; i >= 0; i--) {
            uint64_t temp_b_u64 = base_field::to_canonical_u64(witness[representative_map[u_params[start + 3 + num_a_target + i]]]);
            BigUint temp_b = BigUint(temp_b_u64);
            b = (b << 32) + temp_b;
        }

        BigUint orders[5];
        orders[1] = Ed25519Base::order();
        orders[3] = Secp256K1Base::order();
        orders[4] = Secp256K1Scalar::order();

        bool mod_table[5] = {false, true, true, false, false};

        BigUint order = orders[type];
        bool need_mod = mod_table[type];

        BigUint a_temp = (a >= order) ? (a - order) : a;
        BigUint b_temp = (b >= order) ? (b - order) : b;

        a = need_mod ? (a_temp % order) : a_temp;
        b = need_mod ? (b_temp % order) : b_temp;

        BigUint prod = a * b;
        BigUint overflow, prod_reduced;

        BigUint::div_rem(prod, order, overflow, prod_reduced);

        for (int i = 0; i < 8; i++) {
            witness[representative_map[u_params[start + 3 + num_a_target + 8 + i]]] = base_field::from_u64(prod_reduced.digits[i]);
        }
        for (int i = 0; i < num_overflow_target; i++) {
            witness[representative_map[u_params[start + 3 + num_a_target + 16 + i]]] = base_field::from_u64(overflow.digits[i]);
        }
    }
}

extern "C" __global__ void non_native_inverse_generator_kernel(
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
        size_t type = u_params[u_start + 25 * tid];

        BigUint x = BigUint::zero();
        for(int i = 7; i >= 0; i--) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[representative_map[u_params[u_start + 1 + 25 * tid + i]]]);
            BigUint temp = BigUint(temp_u64);
            x = (x << 32) + temp;
        }

        BigUint orders[5];
        orders[1] = Ed25519Base::order();
        orders[3] = Secp256K1Base::order();
        orders[4] = Secp256K1Scalar::order();

        bool mod_table[5] = {false, true, true, false, false};

        BigUint order = orders[type];
        bool need_mod = mod_table[type];

        BigUint x_temp = (x >= order) ? (x - order) : x;
        x = need_mod ? (x_temp % order) : x_temp;

        BigUint inv = x.inverse_eea(order);

        BigUint prod = x * inv;
        BigUint div = prod / order;

        for (int i = 0; i < 8; i++) {
            witness[representative_map[u_params[u_start + 1 + 25 * tid + 8 + i]]] = base_field::from_u64(div.digits[i]);
        }
        for (int i = 0; i < 8; i++) {
            witness[representative_map[u_params[u_start + 1 + 25 * tid + 16 + i]]] = base_field::from_u64(inv.digits[i]);
        }
    }

    // warp parallel fermat inverse
    // int tid = blockIdx.x;
    // if (tid < length) {
    //     __shared__ uint64_t s_temps[8];
    //     size_t type = u_params[u_start + 25 * tid];
    //     if (threadIdx.x < 8) {
    //         uint64_t temp_u64 = base_field::to_canonical_u64(
    //         witness[ representative_map[u_params[u_start + 1 + 25 * tid + threadIdx.x]]]
    //         );
    //         s_temps[threadIdx.x] = temp_u64;
    //     }
    //     __syncwarp();
    //     BigUint x = BigUint::zero();
    //     for (int i = 7; i >= 0; --i) {
    //         x = (x << 32) + BigUint(s_temps[i]);
    //     }

    //     Ed25519Base x_ed = Ed25519Base::from_noncanonical_BigUint(x);
    //     Ed25519Base inv_ed = x_ed.inverse_warp_parallel();
    //     // Ed25519Base inv_ed = x_ed;

    //     if(threadIdx.x == 0) {

    //         x = x_ed.to_canonical_BigUint();
    //         BigUint inv = inv_ed.to_canonical_BigUint();

    //         BigUint prod = x * inv;
    //         BigUint div = prod / Ed25519Base::order();

    //         for (int i = 0; i < 8; i++) {
    //             witness[representative_map[u_params[u_start + 1 + 25 * tid + 8 + i]]] = base_field::from_u64(div.digits[i]);
    //         }
    //         for (int i = 0; i < 8; i++) {
    //             witness[representative_map[u_params[u_start + 1 + 25 * tid + 16 + i]]] = base_field::from_u64(inv.digits[i]);
    //         }
    //     }
    // }
}

extern "C" __global__ void biguint_div_rem_generator_kernel(
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
        BigUint a = BigUint::zero();
        BigUint b = BigUint::zero();

        for(int i = 15; i >= 0; i--) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[representative_map[u_params[u_start + 41 * tid + i]]]);
            BigUint temp = BigUint(temp_u64);
            a = (a << 32) + temp;
        }

        for(int i = 7; i >= 0; i--) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[representative_map[u_params[u_start + 41 * tid + 16 + i]]]);
            BigUint temp = BigUint(temp_u64);
            b = (b << 32) + temp;
        }

        BigUint div, rem;

        BigUint::div_rem(a, b, div, rem);

        for (int i = 0; i < 9; i++) {
            witness[representative_map[u_params[u_start + 41 * tid + 24 + i]]] = base_field::from_u64(div.digits[i]);
        }

        for (int i = 0; i < 8; i++) {
            witness[representative_map[u_params[u_start + 41 * tid + 33 + i]]] = base_field::from_u64(rem.digits[i]);
        }
    }
}

extern "C" __global__ void glv_decomposition_generator_kernel(
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
        BigUint k_biguint = BigUint::zero();

        for(int i = 7; i >= 0; i--) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[representative_map[u_params[u_start + 18 * tid + i]]]);
            BigUint temp = BigUint(temp_u64);
            k_biguint = (k_biguint << 32) + temp;
        }

        Secp256K1Scalar k = Secp256K1Scalar::from_noncanonical_biguint(k_biguint);

        BigUint p = Secp256K1Scalar::order();
        BigUint c1_biguint, c2_biguint, remainder;

        BigUint dividend = Secp256K1Scalar::B2().to_canonical_biguint() * k.to_canonical_biguint();
        BigUint::div_rem(dividend, p, c1_biguint, remainder);

        if ((remainder + remainder) >= p) {
            c1_biguint = c1_biguint + BigUint::one();
        }
        Secp256K1Scalar c1 = Secp256K1Scalar::from_noncanonical_biguint(c1_biguint);

        dividend = Secp256K1Scalar::MINUS_B1().to_canonical_biguint() * k.to_canonical_biguint();
        BigUint::div_rem(dividend, p, c2_biguint, remainder);

        if ((remainder + remainder) >= p) {
            c2_biguint = c2_biguint + BigUint::one();
        }
        Secp256K1Scalar c2 = Secp256K1Scalar::from_noncanonical_biguint(c2_biguint);

        Secp256K1Scalar k1_raw = k - c1 * Secp256K1Scalar::A1() - c2 * Secp256K1Scalar::A2();
        Secp256K1Scalar k2_raw = c1 * Secp256K1Scalar::MINUS_B1() - c2 * Secp256K1Scalar::B2();

        BigUint cmp = Secp256K1Scalar::order() / BigUint::two();

        int k1_neg = k1_raw.to_canonical_biguint() > cmp;
        Secp256K1Scalar k1 = k1_neg ? Secp256K1Scalar::from_noncanonical_biguint(p - k1_raw.to_canonical_biguint()) : k1_raw;

        int k2_neg = k2_raw.to_canonical_biguint() > cmp;
        Secp256K1Scalar k2 = k2_neg ? Secp256K1Scalar::from_noncanonical_biguint(p - k2_raw.to_canonical_biguint()) : k2_raw;

        BigUint k1_biguint = k1.to_canonical_biguint();
        BigUint k2_biguint = k2.to_canonical_biguint();

        for (int i = 0; i < 4; i++) {
            witness[representative_map[u_params[u_start + 18 * tid + 8 + i]]] = base_field::from_u64(k1_biguint.digits[i]);
            witness[representative_map[u_params[u_start + 18 * tid + 12 + i]]] = base_field::from_u64(k2_biguint.digits[i]);
        }

        witness[representative_map[u_params[u_start + 18 * tid + 16]]] = base_field::from_u64(k1_neg);
        witness[representative_map[u_params[u_start + 18 * tid + 17]]] = base_field::from_u64(k2_neg);
    }
}

extern "C" __global__ void wire_split_generator_kernel(
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
        size_t start = u_start;
        for (int i = 0; i < tid; i++) {
            start += 3 + u_params[start + 2];
        }

        uint64_t integer_value = base_field::to_canonical_u64(witness[representative_map[u_params[start]]]);
        size_t num_limbs = u_params[start + 1];
        size_t num_gate = u_params[start + 2];

        for(unsigned i = 0; i < num_gate; i++) {
            bool less_than = (num_limbs < 64);
            uint64_t truncated_value = less_than ? (integer_value & ((uint64_t(1) << num_limbs) - 1)) : integer_value;
            integer_value = less_than ? (integer_value >> num_limbs) : 0;

            witness[representative_map[u_params[start + 3 + i]]] = base_field::from_u64(truncated_value);
        }
    }
}

extern "C" __global__ void mul_extension_generator_kernel(
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
        size_t index_start = u_params[u_start + tid];

        base_field m_00 = witness[representative_map[index_start]];
        base_field m_01 = witness[representative_map[index_start + 1]];

        extension_field multiplicand_0 = {m_00, m_01};

        base_field m_10 = witness[representative_map[index_start + 2]];
        base_field m_11 = witness[representative_map[index_start + 3]];

        extension_field multiplicand_1 = {m_10, m_11};

        extension_field computed_output = extension_field::mul(
            extension_field::mul(multiplicand_0, multiplicand_1),
            f_params[f_start + tid]
        );

        witness[representative_map[index_start + 4]] = computed_output[0];
        witness[representative_map[index_start + 5]] = computed_output[1];
    }
}


extern "C" __global__ void arithmetic_extension_generator_kernel(
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
        size_t index_start = u_params[u_start + tid];

        base_field m_00 = witness[representative_map[index_start]];
        base_field m_01 = witness[representative_map[index_start + 1]];

        extension_field multiplicand_0 = {m_00, m_01};

        base_field m_10 = witness[representative_map[index_start + 2]];
        base_field m_11 = witness[representative_map[index_start + 3]];

        extension_field multiplicand_1 = {m_10, m_11};

        base_field addend_0 = witness[representative_map[index_start + 4]];
        base_field addend_1 = witness[representative_map[index_start + 5]];

        extension_field addend = {addend_0, addend_1};

        extension_field computed_output = extension_field::add(
            extension_field::mul(
                extension_field::mul(multiplicand_0, multiplicand_1),
                f_params[f_start + 2 * tid]
            ),
            extension_field::mul(addend, f_params[f_start + 1 + 2 * tid])
        );

        witness[representative_map[index_start + 6]] = computed_output[0];
        witness[representative_map[index_start + 7]] = computed_output[1];

    }
}

extern "C" __global__ void reducing_extension_generator_kernel(
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
        size_t index_start = u_params[u_start + 2 * tid];
        size_t num_coeffs = u_params[u_start + 1 + 2 * tid];

        base_field alpha_0 = witness[representative_map[index_start + 2]];
        base_field alpha_1 = witness[representative_map[index_start + 3]];

        extension_field alpha = {alpha_0, alpha_1};

        base_field old_acc_0 = witness[representative_map[index_start + 4]];
        base_field old_acc_1 = witness[representative_map[index_start + 5]];

        extension_field old_acc = {old_acc_0, old_acc_1};

        extension_field acc = old_acc;
        for (int i = 0; i < num_coeffs - 1; i++) {
            base_field coeff_0 = witness[representative_map[index_start + 6 + 2 * i]];
            base_field coeff_1 = witness[representative_map[index_start + 7 + 2 * i]];

            extension_field coeff = {coeff_0, coeff_1};

            extension_field computed_acc = extension_field::add(
                extension_field::mul(acc, alpha),
                coeff
            );

            witness[representative_map[index_start + 6 + 2 * num_coeffs + 2 * i]] = computed_acc[0];
            witness[representative_map[index_start + 7 + 2 * num_coeffs + 2 * i]] = computed_acc[1];

            acc = computed_acc;
        }

        base_field coeff_0 = witness[representative_map[index_start + 4 + 2 * num_coeffs]];
        base_field coeff_1 = witness[representative_map[index_start + 5 + 2 * num_coeffs]];

        extension_field coeff = {coeff_0, coeff_1};

        extension_field computed_acc = extension_field::add(
            extension_field::mul(acc, alpha),
            coeff
        );

        witness[representative_map[index_start]] = computed_acc[0];
        witness[representative_map[index_start + 1]] = computed_acc[1];
    }
}

extern "C" __global__ void reducing_generator_kernel(
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
        size_t index_start = u_params[u_start + 2 * tid];
        size_t num_coeffs = u_params[u_start + 1 + 2 * tid];

        base_field alpha_0 = witness[representative_map[index_start + 2]];
        base_field alpha_1 = witness[representative_map[index_start + 3]];

        extension_field alpha = {alpha_0, alpha_1};

        base_field old_acc_0 = witness[representative_map[index_start + 4]];
        base_field old_acc_1 = witness[representative_map[index_start + 5]];

        extension_field old_acc = {old_acc_0, old_acc_1};

        extension_field acc = old_acc;
        for (int i = 0; i < num_coeffs - 1; i++) {
            base_field coeff_0 = witness[representative_map[index_start + 6 + i]];
            extension_field coeff = {coeff_0, 0};

            extension_field computed_acc = extension_field::add(
                extension_field::mul(acc, alpha),
                coeff
            );

            witness[representative_map[index_start + 6 + num_coeffs + 2 * i]] = computed_acc[0];
            witness[representative_map[index_start + 7 + num_coeffs + 2 * i]] = computed_acc[1];

            acc = computed_acc;
        }

        base_field coeff_0 = witness[representative_map[index_start + 5 + num_coeffs]];
        extension_field coeff = {coeff_0, 0};
        extension_field computed_acc = extension_field::add(
            extension_field::mul(acc, alpha),
            coeff
        );

        witness[representative_map[index_start]] = computed_acc[0];
        witness[representative_map[index_start + 1]] = computed_acc[1];
    }
}

extern "C" __global__ void quotient_generator_extension_kernel(
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
        base_field num_0 = witness[representative_map[u_params[u_start + 6 * tid]]];
        base_field num_1 = witness[representative_map[u_params[u_start + 6 * tid + 1]]];
        extension_field numerator = {num_0, num_1};

        base_field dem_0 = witness[representative_map[u_params[u_start + 6 * tid + 2]]];
        base_field dem_1 = witness[representative_map[u_params[u_start + 6 * tid + 3]]];
        extension_field denominator = {dem_0, dem_1};

        extension_field computed_output = extension_field::mul(
            numerator,
            extension_field::inv(denominator)
        );

        witness[representative_map[u_params[u_start + 6 * tid + 4]]] = computed_output[0];
        witness[representative_map[u_params[u_start + 6 * tid + 5]]] = computed_output[1];
    }
}

extern "C" __global__ void interpolation_generator_kernel(
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
        // size_t subgroup_bits = 4;
        size_t num_points = 16; // 1 << subgroup_bits;
        size_t degree = 6;

        size_t index_start = u_params[u_start + tid];

        base_field evaluation_point_0 = witness[representative_map[index_start + 2 * num_points + 1]];
        base_field evaluation_point_1 = witness[representative_map[index_start + 2 * num_points + 2]];
        extension_field evaluation_point = {evaluation_point_0, evaluation_point_1};

        base_field shift = witness[representative_map[index_start]];
        extension_field shifted_evaluation_point = extension_field::mul(
            evaluation_point,
            base_field::inv(shift)
        );

        witness[representative_map[index_start + 10 * num_points + 25]] = shifted_evaluation_point[0];
        witness[representative_map[index_start + 10 * num_points + 26]] = shifted_evaluation_point[1];

        extension_field computed_eval = extension_field::zero();
        extension_field computed_prod = extension_field::one();
        for (int i = 0; i< degree; i++) {
            base_field domain_0 = f_params[f_start + tid * 32 + i];
            extension_field domain = {domain_0, 0};

            base_field value_0 = witness[representative_map[index_start + 2 * i + 1]];
            base_field value_1 = witness[representative_map[index_start + 2 * i + 2]];
            extension_field value = {value_0, value_1};

            base_field weight_0 = f_params[f_start + tid * 32 + 16 + i];
            extension_field weight = {weight_0, 0};

            extension_field weight_value = extension_field::mul(value, weight);
            extension_field term = extension_field::sub(shifted_evaluation_point, domain);
            computed_eval = extension_field::add(
                extension_field::mul(computed_eval, term),
                extension_field::mul(weight_value, computed_prod)
            );
            computed_prod = extension_field::mul(computed_prod, term);
        }

        for (int i = 0; i < 2; i++) { // num_intermediates = (num_points - 2) / (degree - 1) = (16-2)/(6-1) = 2
            witness[representative_map[index_start + 2 * num_points + 2 * i + 5]] = computed_eval[0];
            witness[representative_map[index_start + 2 * num_points + 2 * i + 6]] = computed_eval[1];

            witness[representative_map[index_start + 2 * num_points + 2 * i + 9]] = computed_prod[0];
            witness[representative_map[index_start + 2 * num_points + 2 * i + 10]] = computed_prod[1];

            size_t start_index = 1 + (degree - 1) * (i + 1);
            size_t end_index = min((start_index + degree - 1), num_points);

            for (int j = start_index; j < end_index; j++) {
                base_field domain_0 = f_params[f_start + tid * 32 + j];
                extension_field domain = {domain_0, 0};

                base_field value_0 = witness[representative_map[index_start + 2 * j + 1]];
                base_field value_1 = witness[representative_map[index_start + 2 * j + 2]];
                extension_field value = {value_0, value_1};

                base_field weight_0 = f_params[f_start + tid * 32 + 16 + j];
                extension_field weight = {weight_0, 0};

                extension_field weight_value = extension_field::mul(value, weight);
                extension_field term = extension_field::sub(shifted_evaluation_point, domain);
                computed_eval = extension_field::add(
                    extension_field::mul(computed_eval, term),
                    extension_field::mul(weight_value, computed_prod)
                );
                computed_prod = extension_field::mul(computed_prod, term);
            }
        }

        witness[representative_map[index_start + 2 * num_points + 3]] = computed_eval[0];
        witness[representative_map[index_start + 2 * num_points + 4]] = computed_eval[1];
    }
}

extern "C" __global__ void poseidon_mds_generator_kernel(
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
        size_t index_start = u_params[u_start + tid];

        extension_field inputs[12];
        for (int i = 0; i < 12; i++) {
            base_field input_0 = witness[representative_map[index_start + 2 * i]];
            base_field input_1 = witness[representative_map[index_start + 2 * i + 1]];
            inputs[i] = {input_0, input_1};
        }

        extension_field outputs[12];
        for (int r = 0; r < 12; r++) {
            outputs[r] = extension_field::zero();
            for (int i = 0; i < 12; i++) {
                extension_field mds_matrix_circ = {MDS_MATRIX_CIRC[i], 0};
                outputs[r] = extension_field::add(
                    outputs[r],
                    extension_field::mul(inputs[(i + r) % 12], mds_matrix_circ)
                );
            }
            extension_field mds_matrix_diag = {MDS_MATRIX_DIAG[r], 0};
            outputs[r] = extension_field::add(
                outputs[r],
                extension_field::mul(inputs[r], mds_matrix_diag)
            );
        }

        for (int i = 0; i < 12; i++) {
            witness[representative_map[index_start + 24 + 2 * i]] = outputs[i][0];
            witness[representative_map[index_start + 24 + 2 * i + 1]] = outputs[i][1];
        }
    }
}

extern "C" __global__ void generate_full_witness_kernel(
    base_field *full_witness,
    base_field *witness,
    const unsigned *representative_map
) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    if (representative_map[i * blockDim.x + j] == UINT_MAX) {
        full_witness[j * gridDim.x + i] = base_field::zero();
        return;
    }
    full_witness[j * gridDim.x + i] = witness[representative_map[i * blockDim.x + j]];
}

extern "C" __global__ void set_initial_values_kernel(
    base_field *witness,
    const size_t *initial_index,
    base_field *initial_values,
    unsigned length
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        witness[initial_index[tid]] = initial_values[tid];
    }
}

extern "C" __global__ void arithmetic_base_generator_new_kernel(
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

    if (tid < length) {
        base_field multiplicand_0 = witness[rep_map[read_start + tid]];
        base_field multiplicand_1 = witness[rep_map[read_start + tid + 1 * stride]];
        base_field addend = witness[rep_map[read_start + tid + 2 * stride]];

        base_field computed_output =
            base_field::add(
                base_field::mul(
                    base_field::mul(multiplicand_0, multiplicand_1),
                    base_field::from_u64(params[param_start + tid])
                ),
                base_field::mul(addend, base_field::from_u64(params[param_start + tid + stride]))
            );
        witness[write_start + tid] = computed_output;
    }
}

extern "C" __global__ void random_value_generator_new_kernel(
    base_field *witness,
    const unsigned *rep_map,
    const size_t *params,
    const unsigned write_start,
    const unsigned read_start,
    const unsigned param_start,
    const unsigned length
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        // base_field random_value = base_field::rand();
        base_field random_value = base_field::zero();
        // base_field random_value = base_field::from_u64(tid);
        witness[write_start + tid] = random_value;
    }
}

extern "C" __global__ void constant_generator_new_kernel(
    base_field *witness,
    const unsigned *rep_map,
    const size_t *params,
    const unsigned write_start,
    const unsigned read_start,
    const unsigned param_start,
    const unsigned length
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < length) {
        witness[write_start + tid] = base_field::from_u64(params[param_start + tid]);
    }
}

extern "C" __global__ void u32_add_many_generator_new_kernel(
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
        size_t num_addends = params[param_start + tid];

        base_field addend;
        base_field output = base_field::zero();

        for (int j = 0; j < num_addends; j++) {
            addend = witness[rep_map[read_start + tid + stride * read_count++]];
            output = base_field::add(output, addend);
        }

        base_field carry = witness[rep_map[read_start + tid + stride * read_count++]];
        output = base_field::add(output, carry);

        uint64_t output_u64 = base_field::to_canonical_u64(output);

        uint64_t output_carry_u64 = output_u64 >> 32;
        uint64_t output_result_u64 = output_u64 & ((uint64_t(1) << 32) - 1);

        base_field output_carry = base_field::from_u64(output_carry_u64);
        base_field output_result = base_field::from_u64(output_result_u64);

        size_t num_result_limbs = (32 + 2 - 1) / 2; // 16
        size_t num_carry_limbs = (4 + 2 - 1) / 2;   // 2
        uint64_t limb_base = uint64_t(1) << 2;      // 4

        for (int j = 0; j < num_result_limbs; j++) {
            base_field ret = base_field::from_u64(output_result_u64 % limb_base);
            witness[write_start + tid + stride * write_count++] = ret;
            output_result_u64 /= limb_base;
        }

        for (int j = 0; j < num_carry_limbs; j++) {
            base_field ret = base_field::from_u64(output_carry_u64 % limb_base);
            witness[write_start + tid + stride * write_count++] = ret;
            output_carry_u64 /= limb_base;
        }

        witness[write_start + tid + stride * write_count++] = output_carry;
        witness[write_start + tid + stride * write_count++] = output_result;
    }
}

extern "C" __global__ void u32_arithmetic_generator_new_kernel(
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
        base_field multiplicand_0 = witness[rep_map[read_start + tid + stride * read_count++]];
        base_field multiplicand_1 = witness[rep_map[read_start + tid + stride * read_count++]];
        base_field addend = witness[rep_map[read_start + tid + stride * read_count++]];

        base_field output = base_field::add(base_field::mul(multiplicand_0, multiplicand_1), addend);
        uint64_t output_u64 = base_field::to_canonical_u64(output);

        uint64_t output_high_u64 = output_u64 >> 32;
        uint64_t output_low_u64 = output_u64 & ((uint64_t(1) << 32) - 1);

        base_field output_high = base_field::from_u64(output_high_u64);
        base_field output_low = base_field::from_u64(output_low_u64);

        witness[write_start + tid + stride * write_count++] = output_high;
        witness[write_start + tid + stride * write_count++] = output_low;

        uint64_t diff = uint64_t(UINT_MAX) - output_high_u64;

        base_field inverse = (diff == 0) ? base_field::zero() : base_field::inv(base_field::from_u64(diff));

        witness[write_start + tid + stride * write_count++] = inverse;

        int num_limbs = 32;
        uint64_t limb_base = uint64_t(1) << 2;

        for (int j = 0; j < num_limbs; j++) {
            base_field ret = base_field::from_u64(output_u64 % limb_base);
            witness[write_start + tid + stride * write_count++] = ret;
            output_u64 /= limb_base;
        }

    }
}

extern "C" __global__ void comparison_generator_new_kernel(
    base_field *witness,
    const unsigned *rep_map,
    const size_t *params,
    const unsigned write_start,
    const unsigned read_start,
    const unsigned param_start,
    const unsigned length
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = length;
    unsigned read_count = 0, write_count = 0;

    if (tid < length) {
        size_t num_bits = params[param_start + tid];
        size_t num_chunks = params[param_start + tid + stride];

        base_field first_input = witness[rep_map[read_start + tid + stride * read_count++]];
        base_field second_input = witness[rep_map[read_start + tid + stride * read_count++]];

        uint64_t first_input_u64 = base_field::to_canonical_u64(first_input);
        uint64_t second_input_u64 = base_field::to_canonical_u64(second_input);
        base_field result = base_field::from_u64(first_input_u64 <= second_input_u64);
        
        witness[write_start + tid + stride * write_count++] = result;

        uint64_t chunk_bits = (num_bits + num_chunks - 1) / num_chunks;
        uint64_t chunk_size = uint64_t(1) << chunk_bits;
        base_field most_significant_diff_so_far = base_field::zero();
        base_field chunks_equal, equality_dummie, intermediate_value;

        for (int i = 0; i < num_chunks; i++) {
            uint64_t first_input_chunk = (first_input_u64 % chunk_size);
            first_input_u64 /= chunk_size;

            uint64_t second_input_chunk = (second_input_u64 % chunk_size);
            second_input_u64 /= chunk_size;

            bool is_equal = (first_input_chunk != second_input_chunk);
            if (is_equal) {
                most_significant_diff_so_far = base_field::sub(
                    base_field::from_u64(second_input_chunk),
                    base_field::from_u64(first_input_chunk)
                );
            }

            intermediate_value = is_equal ? base_field::zero() : most_significant_diff_so_far;
            chunks_equal = is_equal ? base_field::zero() : base_field::one();
            equality_dummie = is_equal ? base_field::inv(
                base_field::sub(
                    base_field::from_u64(second_input_chunk),
                    base_field::from_u64(first_input_chunk)
                )
            ) : base_field::one();


            witness[write_start + tid + stride * write_count++] = base_field::from_u64(first_input_chunk);
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(second_input_chunk);
            witness[write_start + tid + stride * write_count++] = equality_dummie;
            witness[write_start + tid + stride * write_count++] = chunks_equal;
            witness[write_start + tid + stride * write_count++] = intermediate_value;
        }

        base_field most_significant_diff = most_significant_diff_so_far;
        witness[write_start + tid + stride * write_count++] = most_significant_diff;

        base_field two_n = base_field::from_u64(uint64_t(1) << chunk_bits);
        uint64_t two_n_plus_msd = base_field::to_canonical_u64(base_field::add(two_n, most_significant_diff));

        for (int i = 0; i < chunk_bits + 1; i++) {
            uint64_t msd_bit_u64 = two_n_plus_msd % 2;
            two_n_plus_msd /= 2;

            base_field msd_bit = base_field::from_u64(msd_bit_u64);
            witness[write_start + tid + stride * write_count++] = msd_bit;
        }
    }
}

extern "C" __global__ void u32_interleave_generator_new_kernel(
    base_field *witness,
    const unsigned *rep_map,
    const size_t *params,
    const unsigned write_start,
    const unsigned read_start,
    const unsigned param_start,
    const unsigned length
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = length;
    unsigned read_count = 0, write_count = 0;

    if (tid < length) {
        base_field x = witness[rep_map[read_start + tid + stride * read_count++]];
        uint64_t x_u64 = base_field::to_canonical_u64(x);
        uint64_t x_interleaved = 0;
        size_t num_bits = 32;

        for (int j = 0; j < num_bits; j++) {
            uint64_t bit = (x_u64 >> (num_bits - j - 1)) % 2;
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(bit);
            x_interleaved += bit * (uint64_t(1) << (2 * (num_bits - j - 1)));
        }

        witness[write_start + tid + stride * write_count++] = base_field::from_u64(x_interleaved);
    }
}

extern "C" __global__ void u32_range_check_generator_new_kernel(
    base_field *witness,
    const unsigned *rep_map,
    const size_t *params,
    const unsigned write_start,
    const unsigned read_start,
    const unsigned param_start,
    const unsigned length
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = length;
    unsigned read_count = 0, write_count = 0;

    if (tid < length) {
        size_t num_input_limbs = params[param_start + tid];

        for (int i = 0; i < num_input_limbs; i++) {
            uint64_t sum_value_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            uint32_t sum_value = uint32_t(sum_value_u64);
            uint32_t base = uint32_t(1) << 2;

            size_t aux_limbs_per_input_limb = (32 + 2 - 1) / 2;

            for (int j = 0; j < aux_limbs_per_input_limb; j++) {
                uint32_t limb_value = sum_value % base;
                sum_value /= base;
                witness[write_start + tid + stride * write_count++] = base_field::from_u64(uint64_t(limb_value));
            }
        }
    }
}

extern "C" __global__ void u32_subtraction_generator_new_kernel(
    base_field *witness,
    const unsigned *rep_map,
    const size_t *params,
    const unsigned write_start,
    const unsigned read_start,
    const unsigned param_start,
    const unsigned length
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = length;
    unsigned read_count = 0, write_count = 0;

    if (tid < length) {
        base_field input_x = witness[rep_map[read_start + tid + stride * read_count++]];
        base_field input_y = witness[rep_map[read_start + tid + stride * read_count++]];
        base_field input_borrow = witness[rep_map[read_start + tid + stride * read_count++]];

        base_field result_initial = base_field::sub(base_field::sub(input_x, input_y), input_borrow);
        uint64_t result_initial_u64 = base_field::to_canonical_u64(result_initial);
        base_field output_borrow = result_initial_u64 > uint64_t(1) << 32 ? base_field::one() : base_field::zero();

        base_field base = base_field::from_u64(uint64_t(1) << 32);
        base_field output_result = base_field::add(result_initial, base_field::mul(base, output_borrow));

        uint64_t output_result_u64 = base_field::to_canonical_u64(output_result);

        size_t num_limbs = 32 / 2;
        uint64_t limb_base = uint64_t(1) << 2;

        for (int j = 0; j < num_limbs; j++) {
            uint64_t output_limb = output_result_u64 % limb_base;
            output_result_u64 /= limb_base;

            witness[write_start + tid + stride * write_count++] = base_field::from_u64(output_limb);
        }

        witness[write_start + tid + stride * write_count++] = output_result;
        witness[write_start + tid + stride * write_count++] = output_borrow;
    }
}

extern "C" __global__ void uninterleave_to_u32_generator_new_kernel(
    base_field *witness,
    const unsigned *rep_map,
    const size_t *params,
    const unsigned write_start,
    const unsigned read_start,
    const unsigned param_start,
    const unsigned length
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned stride = length;
    unsigned read_count = 0, write_count = 0;

    if (tid < length) {
        uint64_t x_interleaved = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
        uint64_t x_evens = 0;
        uint64_t x_odds = 0;

        size_t num_bits = 64;

        for (int j = 0; j < num_bits / 2; j++) {
            size_t shift = 2 * (num_bits / 2 - j - 1);
            uint64_t jth_even = (x_interleaved >> (shift + 1)) % 2;
            uint64_t jth_odd = (x_interleaved >> shift) % 2;

            witness[write_start + tid + stride * write_count++] = base_field::from_u64(jth_even);
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(jth_odd);

            uint64_t coeff = uint64_t(1) << (num_bits / 2 - j - 1);
            x_evens += jth_even * coeff;
            x_odds += jth_odd * coeff;
        }

        witness[write_start + tid + stride * write_count++] = base_field::from_u64(x_evens);
        witness[write_start + tid + stride * write_count++] = base_field::from_u64(x_odds);
    }
}

extern "C" __global__ void base_sum_generator_new_kernel(
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
        size_t B = params[param_start + tid];
        size_t num_limbs = params[param_start + tid + stride];

        base_field sum = base_field::zero();
        for(int i = 0; i < num_limbs; i++) {
            base_field limb = witness[rep_map[read_start + tid + stride * read_count++]];
            sum = base_field::add(base_field::mul(sum, base_field::from_u64(B)), limb);
        }
        witness[write_start + tid + stride * write_count++] = sum;
    }
}

extern "C" __global__ void base_split_generator_new_kernel(
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

        size_t B = params[param_start + tid];
        size_t num_limbs = params[param_start + tid + stride];
        // size_t num_limbs = 32;

        uint64_t sum_value_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
        size_t sum_value = size_t(sum_value_u64);

        for(int i = 0; i < num_limbs; i++) {
            size_t limb_value = sum_value % B;
            sum_value /= B;
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(limb_value);
        }
    }
}


extern "C" __global__ void random_access_generator_new_kernel(
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
        size_t bits = params[param_start + tid];
        
        size_t vec_size = size_t(1) << bits;
        base_field access_index_f = witness[rep_map[read_start + tid + stride * read_count++]];
        uint64_t access_index = base_field::to_canonical_u64(access_index_f);
        read_count += access_index;

        witness[write_start + tid + stride * write_count++] = witness[rep_map[read_start + tid + stride * read_count]];

        for(int i = 0; i < bits; i++) {
            base_field bit = base_field::from_u64(((access_index >> i) & 1) != 0);
            witness[write_start + tid + stride * write_count++] = bit;
        }
    }
}

extern "C" __global__ void equality_generator_new_kernel(
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
        base_field x = witness[rep_map[read_start + tid + stride * read_count++]];
        base_field y = witness[rep_map[read_start + tid + stride * read_count++]];

        uint64_t x_u64 = base_field::to_canonical_u64(x);
        uint64_t y_u64 = base_field::to_canonical_u64(y);

        bool is_equal = (x_u64 == y_u64);
        base_field diff = base_field::sub(x, y);
        base_field inv = is_equal ? base_field::zero() : base_field::inv(diff);

        witness[write_start + tid + stride * write_count++] = base_field::from_u64(is_equal ? 1 : 0);
        witness[write_start + tid + stride * write_count++] = inv;
    }
}

extern "C" __global__ void curve_point_decompression_generator_new_kernel(
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
        uint8_t bytes[32];

        for(int byte_idx = 0; byte_idx < 32; byte_idx++){
            uint8_t byte = 0;
            for(int bit_idx = 0; bit_idx < 8; bit_idx++){
                uint64_t bit = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
                byte |= (bit != 0) << (7 - bit_idx);
            }

            bytes[31 - byte_idx] = byte;
        }

        FieldElement51 Y = FieldElement51::from_bytes(bytes);

        FieldElement51 Z = FieldElement51::one();
        FieldElement51 YY = Y.square();
        FieldElement51 u = YY - Z;
        FieldElement51 v = YY * FieldElement51::EDWARDS_D() + Z;
        FieldElement51 X = FieldElement51::sqrt_ratio_i(u, v);

        bool negate = (bytes[31] >> 7) != 0;
        X = negate ? (-X) : X;

        uint8_t s[32];
        uint8_t t[32];
        X.as_bytes(s);
        Y.as_bytes(t);

        BigUint x = BigUint::from_bytes_le(s);
        BigUint y = BigUint::from_bytes_le(t);

        x = x % Ed25519Base::order();
        y = y % Ed25519Base::order();

        for (int i = 0; i < 8; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(x.digits[i]);
        }

        for (int i = 0; i < 8; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(y.digits[i]);
        }
    }
}

extern "C" __global__ void non_native_addition_generator_new_kernel(
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
        size_t type = params[param_start + tid];
        size_t num_a_target = params[param_start + tid + stride];
        size_t num_b_target = params[param_start + tid + 2 * stride];

        BigUint a = BigUint::zero();
        BigUint b = BigUint::zero();

        for(int i = 0; i < num_a_target; i++) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            BigUint temp = BigUint(temp_u64);
            a = (a << 32) + temp;
        }

        for(int i = 0; i < num_b_target; i++) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            BigUint temp = BigUint(temp_u64);
            b = (b << 32) + temp;
        }

        BigUint orders[5];
        // orders[0] = goldilocks::order();
        orders[1] = Ed25519Base::order();
        // orders[2] = Ed25519Scalar::order();
        orders[3] = Secp256K1Base::order();
        orders[4] = Secp256K1Scalar::order();

        bool mod_table[5] = {false, true, true, false, false};

        BigUint order = orders[type];
        bool need_mod = mod_table[type];

        BigUint a_temp = (a >= order) ? (a - order) : a;
        BigUint b_temp = (b >= order) ? (b - order) : b;

        a = need_mod ? (a_temp % order) : a_temp;
        b = need_mod ? (b_temp % order) : b_temp;

        BigUint sum = a + b;

        bool overflow = sum > order;
        if (overflow) {
            sum = sum - order;
        }

        for (int i = 0; i < 8; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(sum.digits[i]);
        }
        witness[write_start + tid + stride * write_count++] = base_field::from_u64(overflow ? 1 : 0);
    }
}

extern "C" __global__ void non_native_subtraction_generator_new_kernel(
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
        size_t type = params[param_start + tid];
        size_t num_a_target = params[param_start + tid + stride];
        size_t num_b_target = params[param_start + tid + 2 * stride];

        BigUint a = BigUint::zero();
        BigUint b = BigUint::zero();

        for(int i = 0; i < num_a_target; i++) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            BigUint temp = BigUint(temp_u64);
            a = (a << 32) + temp;
        }

        for(int i = 0; i < num_b_target; i++) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            BigUint temp = BigUint(temp_u64);
            b = (b << 32) + temp;
        }

        BigUint orders[5];
        orders[1] = Ed25519Base::order();
        orders[3] = Secp256K1Base::order();
        orders[4] = Secp256K1Scalar::order();

        bool mod_table[5] = {false, true, true, false, false};

        BigUint order = orders[type];
        bool need_mod = mod_table[type];

        BigUint a_temp = (a >= order) ? (a - order) : a;
        BigUint b_temp = (b >= order) ? (b - order) : b;

        a = need_mod ? (a_temp % order) : a_temp;
        b = need_mod ? (b_temp % order) : b_temp;

        bool less = a < b;

        if (less) {
            a = a + order;
        }

        BigUint diff = a - b;
        for (int i = 0; i < 8; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(diff.digits[i]);
        }
        witness[write_start + tid + stride * write_count++] = base_field::from_u64(less ? 1 : 0);
    }
}

extern "C" __global__ void non_native_multiplication_generator_new_kernel(
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
        size_t type = params[param_start + tid];
        size_t num_a_target = params[param_start + tid + stride];
        size_t num_overflow_target = params[param_start + tid + 2 * stride];

        BigUint a = BigUint::zero();
        BigUint b = BigUint::zero();

        for(int i = 0; i < num_a_target; i++) {
            uint64_t temp_a_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            BigUint temp_a = BigUint(temp_a_u64);
            a = (a << 32) + temp_a;
        }

        for(int i = 0; i < 8; i++) {
            uint64_t temp_b_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            BigUint temp_b = BigUint(temp_b_u64);
            b = (b << 32) + temp_b;
        }

        BigUint orders[5];
        orders[1] = Ed25519Base::order();
        orders[3] = Secp256K1Base::order();
        orders[4] = Secp256K1Scalar::order();

        bool mod_table[5] = {false, true, true, false, false};

        BigUint order = orders[type];
        bool need_mod = mod_table[type];

        BigUint a_temp = (a >= order) ? (a - order) : a;
        BigUint b_temp = (b >= order) ? (b - order) : b;

        a = need_mod ? (a_temp % order) : a_temp;
        b = need_mod ? (b_temp % order) : b_temp;

        BigUint prod = a * b;
        BigUint overflow, prod_reduced;

        BigUint::div_rem(prod, order, overflow, prod_reduced);

        for (int i = 0; i < 8; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(prod_reduced.digits[i]);
        }
        for (int i = 0; i < num_overflow_target; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(overflow.digits[i]);
        }
    }
}

extern "C" __global__ void non_native_inverse_generator_new_kernel(
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
        size_t type = params[param_start + tid];

        BigUint x = BigUint::zero();
        for(int i = 0; i < 8; i++) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            BigUint temp = BigUint(temp_u64);
            x = (x << 32) + temp;
        }

        BigUint orders[5];
        orders[1] = Ed25519Base::order();
        orders[3] = Secp256K1Base::order();
        orders[4] = Secp256K1Scalar::order();

        bool mod_table[5] = {false, true, true, false, false};

        BigUint order = orders[type];
        bool need_mod = mod_table[type];

        BigUint x_temp = (x >= order) ? (x - order) : x;
        x = need_mod ? (x_temp % order) : x_temp;

        BigUint inv = x.inverse_eea(order);

        BigUint prod = x * inv;
        BigUint div = prod / order;

        for (int i = 0; i < 8; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(div.digits[i]);
        }
        for (int i = 0; i < 8; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(inv.digits[i]);
        }
    }
}

extern "C" __global__ void biguint_div_rem_generator_new_kernel(
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
        BigUint a = BigUint::zero();
        BigUint b = BigUint::zero();

        for(int i = 0; i < 16; i++) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            BigUint temp = BigUint(temp_u64);
            a = (a << 32) + temp;
        }

        for(int i = 0; i < 8; i++) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            BigUint temp = BigUint(temp_u64);
            b = (b << 32) + temp;
        }

        BigUint div, rem;

        BigUint::div_rem(a, b, div, rem);

        for (int i = 0; i < 9; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(div.digits[i]);
        }

        for (int i = 0; i < 8; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(rem.digits[i]);
        }
    }
}

extern "C" __global__ void glv_decomposition_generator_new_kernel(
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
        BigUint k_biguint = BigUint::zero();

        for(int i = 0; i < 8; i++) {
            uint64_t temp_u64 = base_field::to_canonical_u64(witness[rep_map[read_start + tid + stride * read_count++]]);
            BigUint temp = BigUint(temp_u64);
            k_biguint = (k_biguint << 32) + temp;
        }

        Secp256K1Scalar k = Secp256K1Scalar::from_noncanonical_biguint(k_biguint);

        BigUint p = Secp256K1Scalar::order();
        BigUint c1_biguint, c2_biguint, remainder;

        BigUint dividend = Secp256K1Scalar::B2().to_canonical_biguint() * k.to_canonical_biguint();
        BigUint::div_rem(dividend, p, c1_biguint, remainder);

        if ((remainder + remainder) >= p) {
            c1_biguint = c1_biguint + BigUint::one();
        }
        Secp256K1Scalar c1 = Secp256K1Scalar::from_noncanonical_biguint(c1_biguint);

        dividend = Secp256K1Scalar::MINUS_B1().to_canonical_biguint() * k.to_canonical_biguint();
        BigUint::div_rem(dividend, p, c2_biguint, remainder);

        if ((remainder + remainder) >= p) {
            c2_biguint = c2_biguint + BigUint::one();
        }
        Secp256K1Scalar c2 = Secp256K1Scalar::from_noncanonical_biguint(c2_biguint);

        Secp256K1Scalar k1_raw = k - c1 * Secp256K1Scalar::A1() - c2 * Secp256K1Scalar::A2();
        Secp256K1Scalar k2_raw = c1 * Secp256K1Scalar::MINUS_B1() - c2 * Secp256K1Scalar::B2();

        BigUint cmp = Secp256K1Scalar::order() / BigUint::two();

        int k1_neg = k1_raw.to_canonical_biguint() > cmp;
        Secp256K1Scalar k1 = k1_neg ? Secp256K1Scalar::from_noncanonical_biguint(p - k1_raw.to_canonical_biguint()) : k1_raw;

        int k2_neg = k2_raw.to_canonical_biguint() > cmp;
        Secp256K1Scalar k2 = k2_neg ? Secp256K1Scalar::from_noncanonical_biguint(p - k2_raw.to_canonical_biguint()) : k2_raw;

        BigUint k1_biguint = k1.to_canonical_biguint();
        BigUint k2_biguint = k2.to_canonical_biguint();

        for (int i = 0; i < 4; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(k1_biguint.digits[i]);
        }

        for (int i = 0; i < 4; i++) {
            witness[write_start + tid + stride * write_count++] = base_field::from_u64(k2_biguint.digits[i]);;
        }

        witness[write_start + tid + stride * write_count++] = base_field::from_u64(k1_neg);
        witness[write_start + tid + stride * write_count++] = base_field::from_u64(k2_neg);
    }
}