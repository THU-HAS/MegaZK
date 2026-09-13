#include "context.cuh"
#include "goldilocks_extension.cuh"

using namespace goldilocks;

typedef __uint128_t uint128_t;
#define LOW_51_BIT_MASK ((1ULL << 51) - 1)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define MAX_DIGITS 18

__constant__ constexpr uint64_t MDS_MATRIX_CIRC[12] = {17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20};
__constant__ constexpr uint64_t MDS_MATRIX_DIAG[12] = {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

class BigUint {
public:
    uint32_t digits[MAX_DIGITS];
    size_t length;

    __device__ static BigUint zero() {
        BigUint result;
        result.length = 1;
        return result;
    }

    __device__ static BigUint one() {
        BigUint result;
        result.digits[0] = 1U;
        result.length = 1;
        return result;
    }

    __device__ static BigUint two() {
        BigUint result;
        result.digits[0] = 2U;
        result.length = 1;
        return result;
    }

    __device__ BigUint() {
        initBigUint();
    }

    __device__ BigUint(uint64_t value) {
        initBigUint();
        digits[0] = (uint32_t)value;
        digits[1] = (uint32_t)(value >> 32);
        length = 2;
        trim();
    }

    __device__ BigUint(const BigUint &other) {
        length = other.length;
        memcpy(digits, other.digits, sizeof(digits));
    }

    __device__ void initBigUint() {
        length = 0;
        memset(digits, 0U, sizeof(digits));
    }

    __device__ int is_zero() const {
        return (length == 1 && digits[0] == 0) || length == 0;
    }

    __device__ int is_one() const {
        return length == 1 && digits[0] == 1;
    }

    __device__ BigUint &operator=(const BigUint &other) {
        length = other.length;
        memcpy(digits, other.digits, sizeof(digits));
        return *this;
    }

    __device__ BigUint operator+(const BigUint& other) const {

        if (is_zero()) {
            return other;
        }

        if (other.is_zero()) {
            return *this;
        }

        size_t maxLength = max(length, other.length);
        BigUint result;
        result.length = maxLength + 1;
        uint64_t carry = 0;

        for (size_t i = 0; i < maxLength || carry; i++) {
            carry += digits[i];
            carry += other.digits[i];
            result.digits[i] = (uint32_t)(carry & 0xFFFFFFFF);
            carry >>= 32;
        }

        result.trim();
        return result;
    }

    __device__ BigUint operator-(const BigUint& other) const {

        if (other.is_zero()) {
            return *this;
        }

        BigUint result;
        uint64_t borrow = 0;

        for (size_t i = 0; i < length; i++) {
            int64_t diff = (int64_t)digits[i] - other.digits[i] - borrow;

            borrow = diff < 0;

            if (borrow) {
                diff += ((uint64_t)1 << 32);
            }

            result.digits[i] = (uint32_t)diff;
            result.length++;
        }

        result.trim();
        return result;
    }

    __device__ BigUint operator*(const BigUint& other) const {
        if (is_zero() || other.is_zero()) {
            return BigUint::zero();
        }

        if (is_one()) {
            return other;
        }

        if (other.is_one()) {
            return *this;
        }

        BigUint result;
        result.length = length + other.length + 1;

        for (size_t i = 0; i < length; i++) {
            uint64_t carry = 0;
            for (size_t j = 0; j < other.length || carry; j++) {
                uint64_t prod = (uint64_t)digits[i] * (uint64_t)other.digits[j]
                              + (uint64_t)result.digits[i + j]
                              + carry;

                result.digits[i + j] = (uint32_t)(prod & 0xFFFFFFFF);
                carry = prod >> 32;
            }
        }

        result.trim();
        return result;
    }

    __device__ BigUint operator/(const BigUint& other) const {

        if (other.is_zero()) {
            printf("Division by zero\n");
            return;
        }

        if (*this < other) {
            return BigUint::zero();
        }

        uint64_t b = 4294967296ULL;

        BigUint u = *this;
        BigUint v = other;

        size_t n = v.length;
        size_t m = u.length - v.length;

        BigUint quotient;
        quotient.length = m + 1;

        // Normalize
        size_t shift = __builtin_clz(v.digits[n - 1]);
        v = v << shift;
        u = u << shift;

        for(ssize_t j = m; j >= 0; j--) {
            uint64_t uhat = (uint64_t)u.digits[j + n] * b + u.digits[j + n - 1];
            uint64_t qhat = uhat / v.digits[n - 1];
            uint64_t rhat = uhat % v.digits[n - 1];

            while ((qhat >= b) || (qhat * v.digits[n - 2] > (rhat * b + u.digits[j + n - 2]))) {
                qhat--;
                rhat += v.digits[n - 1];
                if (rhat >= b) break;
            }

            int64_t k = 0;
            for (size_t i = 0; i < n; i++) {
                uint64_t p = qhat * v.digits[i];
                int64_t t = u.digits[j + i] - k - (p & 0xFFFFFFFFLL);
                u.digits[j + i] = t;
                k = (p >> 32) - (t >> 32);
            }

            int64_t t = u.digits[j + n] - k;
            u.digits[j + n] = t;

            if (t < 0) {
                qhat--;
                k = 0;
                for (size_t i = 0; i < n; i++) {
                    uint64_t t = u.digits[j + i] + v.digits[i] + k;
                    u.digits[j + i] = t;
                    k = t >> 32;
                }
                u.digits[j + n] += k;
            }
            quotient.digits[j] = qhat;
        }

        quotient.trim();
        return quotient;
    }

    __device__ BigUint operator%(const BigUint& other) const {

        if (other.is_zero()) {
            printf("Division by zero\n");
            return;
        }

        if (*this < other) {
            return *this;
        }

        uint64_t b = 4294967296ULL;

        BigUint u = *this;
        BigUint v = other;

        size_t n = v.length;
        size_t m = u.length - v.length;

        // Normalize
        size_t shift = __builtin_clz(v.digits[n - 1]);
        v = v << shift;
        u = u << shift;

        for(ssize_t j = m; j >= 0; j--) {
            uint64_t uhat = (uint64_t)u.digits[j + n] * b + u.digits[j + n - 1];
            uint64_t qhat = uhat / v.digits[n - 1];
            uint64_t rhat = uhat % v.digits[n - 1];

            while ((qhat >= b) || (qhat * v.digits[n - 2] > (rhat * b + u.digits[j + n - 2]))) {
                qhat--;
                rhat += v.digits[n - 1];
                if (rhat >= b) break;
            }

            int64_t k = 0;
            for (size_t i = 0; i < n; i++) {
                uint64_t p = qhat * v.digits[i];
                int64_t t = u.digits[j + i] - k - (p & 0xFFFFFFFFLL);
                u.digits[j + i] = t;
                k = (p >> 32) - (t >> 32);
            }

            int64_t t = u.digits[j + n] - k;
            u.digits[j + n] = t;

            if (t < 0) {
                qhat--;
                k = 0;
                for (size_t i = 0; i < n; i++) {
                    uint64_t t = u.digits[j + i] + v.digits[i] + k;
                    u.digits[j + i] = t;
                    k = t >> 32;
                }
                u.digits[j + n] += k;
            }
        }

        // Unnormalize
        BigUint remainder;
        remainder.length = n;
        for (size_t i = 0; i < n; i++) {
            remainder.digits[i] = u.digits[i];
        }
        remainder.trim();
        remainder = remainder >> shift;
        return remainder;
    }

    __device__ static void div_rem(const BigUint& dividend, const BigUint& divisor, BigUint& quotient, BigUint& remainder) {

        if (divisor.is_zero()) {
            printf("Division by zero\n");
            return;
        }

        if (dividend < divisor) {
            quotient = BigUint::zero();
            remainder = dividend;
            return;
        }

        BigUint u = dividend;
        BigUint v = divisor;

        uint64_t b = 4294967296ULL;
        size_t n = v.length;
        size_t m = u.length - v.length;

        quotient.length = m + 1;

        // Normalize
        size_t shift = __builtin_clz(v.digits[n - 1]);
        v = v << shift;
        u = u << shift;

        for(ssize_t j = m; j >= 0; j--) {
            uint64_t uhat = (uint64_t)u.digits[j + n] * b + u.digits[j + n - 1];
            uint64_t qhat = uhat / v.digits[n - 1];
            uint64_t rhat = uhat % v.digits[n - 1];

            while ((qhat >= b) || (qhat * v.digits[n - 2] > (rhat * b + u.digits[j + n - 2]))) {
                qhat--;
                rhat += v.digits[n - 1];
                if (rhat >= b) break;
            }

            int64_t k = 0;
            for (size_t i = 0; i < n; i++) {
                uint64_t p = qhat * v.digits[i];
                int64_t t = u.digits[j + i] - k - (p & 0xFFFFFFFFLL);
                u.digits[j + i] = t;
                k = (p >> 32) - (t >> 32);
            }

            int64_t t = u.digits[j + n] - k;
            u.digits[j + n] = t;

            if (t < 0) {
                qhat--;
                k = 0;
                for (size_t i = 0; i < n; i++) {
                    uint64_t t = u.digits[j + i] + v.digits[i] + k;
                    u.digits[j + i] = t;
                    k = t >> 32;
                }
                u.digits[j + n] += k;
            }
            quotient.digits[j] = qhat;
        }

        quotient.trim();

        remainder.length = n;
        for (size_t i = 0; i < n; i++) {
            remainder.digits[i] = u.digits[i];
        }
        remainder.trim();
        remainder = remainder >> shift;
    }

    __device__ bool operator<(const BigUint& other) const {
        if (length != other.length) {
            return length < other.length;
        }

        for (int i = length - 1; i >= 0; i--) {
            if (digits[i] != other.digits[i]) {
                return digits[i] < other.digits[i];
            }
        }

        return false;
    }

    __device__ bool operator==(const BigUint& other) const {
        if (length != other.length) {
            return false;
        }

        for (int i = length - 1; i >= 0; i--) {
            if (digits[i] != other.digits[i]) {
                return false;
            }
        }

        return true;
    }

    __device__ bool operator>(const BigUint &other) const {
        return !(*this <= other);
    }

    __device__ bool operator<=(const BigUint &other) const {
        return *this < other || *this == other;
    }

    __device__ bool operator>=(const BigUint &other) const {
        return !(*this < other);
    }

    __device__ BigUint operator<<(size_t shift) const {
        if (shift == 0)
            return *this;

        BigUint result;
        size_t wordShift = shift / 32;
        size_t bitShift = shift % 32;
        result.length = length + wordShift + 1;

        uint32_t carry = 0;
        for (size_t i = 0; i < length; i++) {
            uint64_t temp = ((uint64_t)digits[i] << bitShift) | carry;
            result.digits[i + wordShift] = (uint32_t)temp;
            carry = temp >> 32;
        }

        if (carry) {
            result.digits[length + wordShift] = carry;
        }

        for (size_t i = 0; i < wordShift; i++) {
            result.digits[i] = 0;
        }

        result.trim();
        return result;
    }

    __device__ BigUint operator>>(size_t shift) const {
        if (shift == 0)
            return *this;

        BigUint result;
        size_t wordShift = shift / 32;
        size_t bitShift = shift % 32;

        if (wordShift >= length) {
            result = BigUint::zero();
            return result;
        }

        result.length = length - wordShift;

        uint32_t carry = 0;

        for (int i = length - 1; i >= (int)wordShift; i--) {
            uint64_t temp = ((uint64_t)carry << 32) | digits[i];
            result.digits[i - wordShift] = (uint32_t)(temp >> bitShift);
            carry = (uint32_t)(temp);
        }

        for (size_t i = result.length; i < MAX_DIGITS; i++) {
            result.digits[i] = 0;
        }

        return result;
    }

    __device__ BigUint inverse_eea(BigUint m) const;

    __device__ static BigUint from_bytes_le(uint8_t *bytes) {
        int BIG_DIGIT_BITS = 64;
        int bits = 8;
        uint8_t digits_per_big_digit = BIG_DIGIT_BITS / bits;
        int num_chunks = 32 / digits_per_big_digit;

        uint64_t value[4];
        for (int i = 0; i < num_chunks; i++) {
            uint64_t acc = 0;

            int chunk_start = (i + 1) * digits_per_big_digit - 1;
            int chunk_end = i * digits_per_big_digit;

            for (int j = chunk_start; j >= chunk_end; j--) {
                acc = (acc << bits) | bytes[j];
            }
            value[i] = acc;
        }

        BigUint result;
        for (int i = 0; i < 4; i++) {
            result.digits[2 * i] = (uint32_t)value[i];
            result.digits[2 * i + 1] = (uint32_t)(value[i] >> 32);
            result.length += 2;
        }

        return result;
    }

    __device__ void print() const {
        if (length == 0) {
            printf("0");
            return;
        }
        char buffer[MAX_DIGITS * 10] = {0};
        BigUint temp = *this;

        size_t index = 0;
        while (temp.length > 0) {
            uint64_t remainder = 0;
            for (int i = temp.length - 1; i >= 0; i--) {
                uint64_t value = ((uint64_t)remainder << 32) | temp.digits[i];
                temp.digits[i] = (uint32_t)(value / 10);
                remainder = value % 10;
            }
            buffer[index++] = '0' + (char)remainder;

            while (temp.length > 0 && temp.digits[temp.length - 1] == 0) {
                temp.length--;
            }
        }
        for (int i = index - 1; i >= 0; i--) {
            printf("%c", buffer[i]);
        }
        printf("\n");
    }

    __device__ void trim() {
        while (length > 0 && digits[length - 1] == 0) {
            length--;
        }
    }
};

enum class Sign {
    Minus = -1,
    NoSign = 0,
    Plus = 1
};

class Bigint {
public:
    BigUint value;
    Sign sign;

    __device__ static Bigint zero() {
        Bigint result;
        result.value = BigUint::zero();
        result.sign = Sign::NoSign;
        return result;
    }

    __device__ static Bigint one() {
        Bigint result;
        result.value = BigUint::one();
        result.sign = Sign::Plus;
        return result;
    }

    __device__ Bigint() : value(BigUint::zero()), sign(Sign::NoSign) {}

    __device__ Bigint(const BigUint& value, Sign sign = Sign::Plus) : value(value), sign(sign) {}

    __device__ Bigint(const Bigint& other) : value(other.value), sign(other.sign) {}

    __device__ Bigint& operator=(const Bigint& other) {
        if (this != &other) {
            value = other.value;
            sign = other.sign;
        }
        return *this;
    }

    __device__ bool is_zero() const {
        return value.is_zero() || (sign == Sign::NoSign);
    }

    __device__ Bigint operator-(const Bigint& other) const {
        if (other.is_zero()) {
            return *this;
        }

        if (is_zero()) {
            return Bigint(other.value, other.sign == Sign::Plus ? Sign::Minus : Sign::Plus);
        }

        if (sign != other.sign) {
            return Bigint(value + other.value, sign);
        }

        if (value == other.value) {
            return Bigint::zero();
        }

        return value > other.value ? Bigint(value - other.value, sign) : Bigint(other.value - value, sign == Sign::Plus ? Sign::Minus : Sign::Plus);
    }

    __device__ Bigint operator*(const BigUint& other) const {
        if (is_zero() || other.is_zero()) {
            return Bigint::zero();
        }

        BigUint product = value * other;

        return Bigint(product, sign);
    }

    __device__ Bigint operator/(const Bigint& divisor) const {
        if (divisor.is_zero()) {
            printf("Division by zero\n");
            return Bigint();
        }

        BigUint quotient = value / divisor.value;

        if (quotient.is_zero()) {
            return Bigint(BigUint::zero(), Sign::NoSign);
        }

        return sign == divisor.sign ? Bigint(quotient, Sign::Plus) : Bigint(quotient + BigUint::one(), Sign::Minus);
    }

    __device__ BigUint operator%(const BigUint& divisor) const {
        if (divisor.is_zero()) {
            printf("Division by zero\n");
            return BigUint::zero();
        }

        BigUint remainder = value % divisor;

        if (remainder.is_zero()) {
            return BigUint::zero();
        }

        return sign == Sign::Minus ? divisor - remainder : remainder;
    }
};

__device__ BigUint BigUint::inverse_eea(BigUint m) const {
    BigUint a = *this;

    if (a.is_zero()) {
        return BigUint::zero();
    }

    BigUint old_r, r, quotient;
    Bigint old_s, s;

    old_r = a;
    r = m;
    old_s = Bigint::one();
    s = Bigint::zero();

    while (!r.is_zero()) {
        quotient = old_r / r;

        BigUint temp_r = r;
        r = old_r - quotient * r;
        old_r = temp_r;

        Bigint temp_s = s;
        s = old_s - s * quotient;
        old_s = temp_s;
    }

    if (!old_r.is_one()) {
        return BigUint::zero();
    }

    return old_s % m;
}

class Ed25519Base {
public:
    uint64_t limbs[4];

    __device__ static BigUint order(){
        BigUint result;
        result.digits[0] = 0xFFFFFFED;
        result.digits[1] = 0xFFFFFFFF;
        result.digits[2] = 0xFFFFFFFF;
        result.digits[3] = 0xFFFFFFFF;
        result.digits[4] = 0xFFFFFFFF;
        result.digits[5] = 0xFFFFFFFF;
        result.digits[6] = 0xFFFFFFFF;
        result.digits[7] = 0x7FFFFFFF;
        result.length = 8;

        return result;
    }

    __device__ static Ed25519Base one(){
        return Ed25519Base(1ULL, 0ULL, 0ULL, 0ULL);
    }

    __device__ Ed25519Base() {
        limbs[0] = 0;
        limbs[1] = 0;
        limbs[2] = 0;
        limbs[3] = 0;
    }

    __device__ Ed25519Base(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        limbs[0] = a;
        limbs[1] = b;
        limbs[2] = c;
        limbs[3] = d;
    }

    __device__ static Ed25519Base from_noncanonical_biguint(BigUint a){
        Ed25519Base result;
        for (int i = 0; i < 4; i++) {
            uint64_t carry = uint64_t(a.digits[2 * i]) | (uint64_t(a.digits[2 * i + 1]) << 32);
            result.limbs[i] = carry;
        }
        return result;
    }

    __device__ Ed25519Base inverse_fermat() const {
        return this->exp_biguint(Ed25519Base::order() - BigUint::two());
    }

    __device__ Ed25519Base exp_biguint(const BigUint& power) const {
        Ed25519Base result = Ed25519Base::one();
        for(int i = 3; i >= 0; i--) {
            uint64_t digit = power.digits[2 * i] | (uint64_t(power.digits[2 * i + 1]) << 32);
            result = result.exp_power_of_2(64);
            result *= this->exp_u64(digit);
        }
        return result;
    }

    __device__ Ed25519Base exp_power_of_2(size_t power_log) const {
        Ed25519Base res = *this;
        for(int i = 0; i < power_log; i++) {
            res = res.square();
        }
        return res;
    }

    __device__ Ed25519Base square() const {
        return *this * *this;
    }

    __device__ Ed25519Base exp_u64(uint64_t power) const{
        Ed25519Base current = *this;
        Ed25519Base product = Ed25519Base::one();

        for(int j =0 ; j < 64 - __builtin_clzll(power); j++) {
            if ((power >> j & 1) != 0) {
                product *= current;
            }
            current = current.square();
        }
        return product;
    }

    __device__ Ed25519Base& operator=(const Ed25519Base& other) {
        if (this != &other) {
            for (int i = 0; i < 4; i++) {
                limbs[i] = other.limbs[i];
            }
        }
        return *this;
    }

    __device__ Ed25519Base operator*(const Ed25519Base& other) const {
        BigUint tmp = this->to_canonical_biguint() * other.to_canonical_biguint();
        tmp = tmp % Ed25519Base::order();
        Ed25519Base result = Ed25519Base::from_noncanonical_biguint(tmp);
        return result;
    }

    __device__ Ed25519Base operator*= (const Ed25519Base& other) {
        *this = *this * other;
        return *this;
    }

    __device__ BigUint to_canonical_biguint() const {
        BigUint result;
        for (int i = 0; i < 4; i++) {
            uint64_t carry = limbs[i];
            result.digits[2 * i] = (uint32_t)carry;
            result.digits[2 * i + 1] = (uint32_t)(carry >> 32);
            result.length += 2;
        }

        result.trim();
        result = result % Ed25519Base::order();

        return result;
    }
};

class FieldElement51{
public:
    uint64_t limbs[5];

    __device__ static FieldElement51 one(){
        return FieldElement51(1ULL, 0ULL, 0ULL, 0ULL, 0ULL);
    }
    __device__ static FieldElement51 EDWARDS_D(){
        return FieldElement51(
            929955233495203ULL,
            466365720129213ULL,
            1662059464998953ULL,
            2033849074728123ULL,
            1442794654840575ULL
        );
    }
    __device__ static FieldElement51 SQRT_M1(){
        return FieldElement51(
            1718705420411056,
            234908883556509,
            2233514472574048,
            2117202627021982,
            765476049583133
        );
    }

    __device__ FieldElement51() {
        for(int i = 0; i < 5; i++) {
            limbs[i] = 0;
        }
    }

    __device__ FieldElement51(uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e) {
        limbs[0] = a;
        limbs[1] = b;
        limbs[2] = c;
        limbs[3] = d;
        limbs[4] = e;
    }

    __device__ FieldElement51(uint64_t *a){
        for(int i = 0; i < 5; i++) {
            limbs[i] = a[i];
        }
    }

    __device__ static FieldElement51 from_bytes(uint8_t *bytes) {
        FieldElement51 result;
        result.limbs[0] = load8(&bytes[0]) & LOW_51_BIT_MASK;
        result.limbs[1] = (load8(&bytes[6]) >> 3) & LOW_51_BIT_MASK;
        result.limbs[2] = (load8(&bytes[12]) >> 6) & LOW_51_BIT_MASK;
        result.limbs[3] = (load8(&bytes[19]) >> 1) & LOW_51_BIT_MASK;
        result.limbs[4] = (load8(&bytes[24]) >> 12) & LOW_51_BIT_MASK;
        return result;
    }

    __device__ FieldElement51 operator=(const FieldElement51& other) {
        for (int i = 0; i < 5; i++) {
            limbs[i] = other.limbs[i];
        }
        return *this;
    }

    __device__ FieldElement51 operator+(const FieldElement51& other) const {
        FieldElement51 result;
        for (int i = 0; i < 5; i++) {
            result.limbs[i] = this->limbs[i] + other.limbs[i];
        }
        return result;
    }

    __device__ FieldElement51 operator-(const FieldElement51& other) const {
        uint64_t limbs[5] = {
            (this->limbs[0] + 36028797018963664ULL) - other.limbs[0],
            (this->limbs[1] + 36028797018963952ULL) - other.limbs[1],
            (this->limbs[2] + 36028797018963952ULL) - other.limbs[2],
            (this->limbs[3] + 36028797018963952ULL) - other.limbs[3],
            (this->limbs[4] + 36028797018963952ULL) - other.limbs[4]
        };

        return reduce(limbs);
    }

    __device__ FieldElement51 operator*(const FieldElement51& other) const {
        uint64_t a[5] = {this->limbs[0], this->limbs[1], this->limbs[2], this->limbs[3], this->limbs[4]};
        uint64_t b[5] = {other.limbs[0], other.limbs[1], other.limbs[2], other.limbs[3], other.limbs[4]};

        uint64_t b1_19 = b[1] * 19;
        uint64_t b2_19 = b[2] * 19;
        uint64_t b3_19 = b[3] * 19;
        uint64_t b4_19 = b[4] * 19;

        uint128_t c0 = m(a[0], b[0]) + m(a[4], b1_19) + m(a[3], b2_19) + m(a[2], b3_19) + m(a[1], b4_19);
        uint128_t c1 = m(a[1], b[0]) + m(a[0],  b[1]) + m(a[4], b2_19) + m(a[3], b3_19) + m(a[2], b4_19);
        uint128_t c2 = m(a[2], b[0]) + m(a[1],  b[1]) + m(a[0],  b[2]) + m(a[4], b3_19) + m(a[3], b4_19);
        uint128_t c3 = m(a[3], b[0]) + m(a[2],  b[1]) + m(a[1],  b[2]) + m(a[0],  b[3]) + m(a[4], b4_19);
        uint128_t c4 = m(a[4], b[0]) + m(a[3],  b[1]) + m(a[2],  b[2]) + m(a[1],  b[3]) + m(a[0],  b[4]);

        uint64_t out[5];

        c1 += uint128_t(uint64_t(c0 >> 51));
        out[0] = uint64_t(c0) & LOW_51_BIT_MASK;

        c2 += uint128_t(uint64_t(c1 >> 51));
        out[1] = uint64_t(c1) & LOW_51_BIT_MASK;

        c3 += uint128_t(uint64_t(c2 >> 51));
        out[2] = uint64_t(c2) & LOW_51_BIT_MASK;

        c4 += uint128_t(uint64_t(c3 >> 51));
        out[3] = uint64_t(c3) & LOW_51_BIT_MASK;

        uint64_t carry = uint64_t(c4 >> 51);
        out[4] = uint64_t(c4) & LOW_51_BIT_MASK;

        out[0] += carry * 19;
        out[1] += out[0] >> 51;
        out[0] &= LOW_51_BIT_MASK;

        return FieldElement51(out);
    }

    __device__ FieldElement51 operator*= (const FieldElement51& other) {
        *this = *this * other;
        return *this;
    }

    __device__ bool operator==(const FieldElement51& other) const{
        uint8_t a[32], b[32];
        this->as_bytes(a);
        other.as_bytes(b);

        for(int i = 0; i < 32; i++) {
            if(a[i] != b[i]) {
                return false;
            }
        }

        return true;
    }

    __device__ FieldElement51 operator-() const {

        uint64_t limbs[5] = {
            (36028797018963664ULL) - this->limbs[0],
            (36028797018963952ULL) - this->limbs[1],
            (36028797018963952ULL) - this->limbs[2],
            (36028797018963952ULL) - this->limbs[3],
            (36028797018963952ULL) - this->limbs[4]
        };

        return reduce(limbs);
    }

    __device__ FieldElement51 pow2k(unsigned k) const {
        uint64_t a[5] = {this->limbs[0], this->limbs[1], this->limbs[2], this->limbs[3], this->limbs[4]};

        while(1){
            uint64_t a3_19 = 19 * a[3];
            uint64_t a4_19 = 19 * a[4];

            uint128_t c0 = m(a[0],  a[0]) + 2 * (m(a[1], a4_19) + m(a[2], a3_19));
            uint128_t c1 = m(a[3], a3_19) + 2 * (m(a[0],  a[1]) + m(a[2], a4_19));
            uint128_t c2 = m(a[1],  a[1]) + 2 * (m(a[0],  a[2]) + m(a[4], a3_19));
            uint128_t c3 = m(a[4], a4_19) + 2 * (m(a[0],  a[3]) + m(a[1],  a[2]));
            uint128_t c4 = m(a[2],  a[2]) + 2 * (m(a[0],  a[4]) + m(a[1],  a[3]));

            c1 += uint128_t(uint64_t(c0 >> 51));
            a[0] = (uint64_t)c0 & LOW_51_BIT_MASK;

            c2 += uint128_t(uint64_t(c1 >> 51));
            a[1] = (uint64_t)c1 & LOW_51_BIT_MASK;

            c3 += uint128_t(uint64_t(c2 >> 51));
            a[2] = (uint64_t)c2 & LOW_51_BIT_MASK;

            c4 += uint128_t(uint64_t(c3 >> 51));
            a[3] = (uint64_t)c3 & LOW_51_BIT_MASK;

            uint64_t carry = uint64_t(c4 >> 51);
            a[4] = (uint64_t)c4 & LOW_51_BIT_MASK;

            a[0] += carry * 19;

            a[1] += a[0] >> 51;
            a[0] &= LOW_51_BIT_MASK;

            k--;

            if (k == 0) {
                break;
            }
        }

        return FieldElement51(a);
    }

    __device__ FieldElement51 square() const {
        return this->pow2k(1);
    }

    __device__ static FieldElement51 sqrt_ratio_i(FieldElement51 u, FieldElement51 v) {
        FieldElement51 v3 = u.square() * v;
        FieldElement51 v7 = v3.square() * v;
        FieldElement51 r = (u * v3) * (u * v7).pow_p58();
        FieldElement51 check = v * r.square();

        FieldElement51 i = FieldElement51::SQRT_M1();

        FieldElement51 r_prime = i * r;

        if((check == -u) || (check == (-u) * i)) {
            r = r_prime;
        }

        uint8_t a[32];
        r.as_bytes(a);
        if(a[0] & 1){
            r = -r;
        }

        return r;
    }

    __device__ void as_bytes(uint8_t *s) const {
        uint64_t limbs[5];
        for (int i = 0; i < 5; i++) {
            limbs[i] = this->limbs[i];
        }
        FieldElement51 ff = reduce(limbs);
        for (int i = 0; i < 5; i++) {
            limbs[i] = ff.limbs[i];
        }

        uint64_t q = (limbs[0] + 19) >> 51;
        q = (limbs[1] + q) >> 51;
        q = (limbs[2] + q) >> 51;
        q = (limbs[3] + q) >> 51;
        q = (limbs[4] + q) >> 51;

        limbs[0] += 19 * q;

        limbs[1] += limbs[0] >> 51;
        limbs[0] &= LOW_51_BIT_MASK;
        limbs[2] += limbs[1] >> 51;
        limbs[1] &= LOW_51_BIT_MASK;
        limbs[3] += limbs[2] >> 51;
        limbs[2] &= LOW_51_BIT_MASK;
        limbs[4] += limbs[3] >> 51;
        limbs[3] &= LOW_51_BIT_MASK;
        limbs[4] &= LOW_51_BIT_MASK;

        s[ 0] =   limbs[0]                           & 0xFF;
        s[ 1] =  (limbs[0] >>  8)                    & 0xFF;
        s[ 2] =  (limbs[0] >> 16)                    & 0xFF;
        s[ 3] =  (limbs[0] >> 24)                    & 0xFF;
        s[ 4] =  (limbs[0] >> 32)                    & 0xFF;
        s[ 5] =  (limbs[0] >> 40)                    & 0xFF;
        s[ 6] = ((limbs[0] >> 48) | (limbs[1] << 3)) & 0xFF;
        s[ 7] =  (limbs[1] >>  5)                    & 0xFF;
        s[ 8] =  (limbs[1] >> 13)                    & 0xFF;
        s[ 9] =  (limbs[1] >> 21)                    & 0xFF;
        s[10] =  (limbs[1] >> 29)                    & 0xFF;
        s[11] =  (limbs[1] >> 37)                    & 0xFF;
        s[12] = ((limbs[1] >> 45) | (limbs[2] << 6)) & 0xFF;
        s[13] =  (limbs[2] >>  2)                    & 0xFF;
        s[14] =  (limbs[2] >> 10)                    & 0xFF;
        s[15] =  (limbs[2] >> 18)                    & 0xFF;
        s[16] =  (limbs[2] >> 26)                    & 0xFF;
        s[17] =  (limbs[2] >> 34)                    & 0xFF;
        s[18] =  (limbs[2] >> 42)                    & 0xFF;
        s[19] = ((limbs[2] >> 50) | (limbs[3] << 1)) & 0xFF;
        s[20] =  (limbs[3] >>  7)                    & 0xFF;
        s[21] =  (limbs[3] >> 15)                    & 0xFF;
        s[22] =  (limbs[3] >> 23)                    & 0xFF;
        s[23] =  (limbs[3] >> 31)                    & 0xFF;
        s[24] =  (limbs[3] >> 39)                    & 0xFF;
        s[25] = ((limbs[3] >> 47) | (limbs[4] << 4)) & 0xFF;
        s[26] =  (limbs[4] >>  4)                    & 0xFF;
        s[27] =  (limbs[4] >> 12)                    & 0xFF;
        s[28] =  (limbs[4] >> 20)                    & 0xFF;
        s[29] =  (limbs[4] >> 28)                    & 0xFF;
        s[30] =  (limbs[4] >> 36)                    & 0xFF;
        s[31] =  (limbs[4] >> 44)                    & 0xFF;
    }

private:
    __device__ static uint64_t load8(uint8_t *input) {
        return (uint64_t(input[0]))
            | (uint64_t(input[1]) << 8)
            | (uint64_t(input[2]) << 16)
            | (uint64_t(input[3]) << 24)
            | (uint64_t(input[4]) << 32)
            | (uint64_t(input[5]) << 40)
            | (uint64_t(input[6]) << 48)
            | (uint64_t(input[7]) << 56);
    }

    __device__ static inline uint128_t m(uint64_t x, uint64_t y) {
        return (uint128_t)x * (uint128_t)y;
    }

    __device__ static inline FieldElement51 reduce(uint64_t *limbs) {
        uint64_t c0 = limbs[0] >> 51;
        uint64_t c1 = limbs[1] >> 51;
        uint64_t c2 = limbs[2] >> 51;
        uint64_t c3 = limbs[3] >> 51;
        uint64_t c4 = limbs[4] >> 51;

        limbs[0] &= LOW_51_BIT_MASK;
        limbs[1] &= LOW_51_BIT_MASK;
        limbs[2] &= LOW_51_BIT_MASK;
        limbs[3] &= LOW_51_BIT_MASK;
        limbs[4] &= LOW_51_BIT_MASK;

        limbs[0] += c4 * 19;
        limbs[1] += c0;
        limbs[2] += c1;
        limbs[3] += c2;
        limbs[4] += c3;

        return FieldElement51(limbs);
    }

    __device__ FieldElement51 pow_p58() {

        FieldElement51 t19 = this->pow22501();
        FieldElement51 t20 = t19.pow2k(2);
        FieldElement51 t21 = *this * t20;

        return t21;
    }

    __device__ FieldElement51 pow22501() {
        FieldElement51 t0  = this->square();
        FieldElement51 t1  = t0.square().square();
        FieldElement51 t2  = *this * t1;
        FieldElement51 t3  = t0 * t2;
        FieldElement51 t4  = t3.square();
        FieldElement51 t5  = t2 * t4;
        FieldElement51 t6  = t5.pow2k(5);
        FieldElement51 t7  = t6 * t5;
        FieldElement51 t8  = t7.pow2k(10);
        FieldElement51 t9  = t8 * t7;
        FieldElement51 t10 = t9.pow2k(20);
        FieldElement51 t11 = t10 * t9;
        FieldElement51 t12 = t11.pow2k(10);
        FieldElement51 t13 = t12 * t7;
        FieldElement51 t14 = t13.pow2k(50);
        FieldElement51 t15 = t14 * t13;
        FieldElement51 t16 = t15.pow2k(100);
        FieldElement51 t17 = t16 * t15;
        FieldElement51 t18 = t17.pow2k(50);
        FieldElement51 t19 = t18 * t13;

        return t19;
    }
};

class Secp256K1Base {
public:
    uint64_t limbs[4];

    __device__ static BigUint order(){
        BigUint result;
        result.digits[0] = 0xFFFFFC2F;
        result.digits[1] = 0xFFFFFFFE;
        result.digits[2] = 0xFFFFFFFF;
        result.digits[3] = 0xFFFFFFFF;
        result.digits[4] = 0xFFFFFFFF;
        result.digits[5] = 0xFFFFFFFF;
        result.digits[6] = 0xFFFFFFFF;
        result.digits[7] = 0xFFFFFFFF;
        result.length = 8;

        return result;
    }

    __device__ static Secp256K1Base one(){
        return Secp256K1Base(1ULL, 0ULL, 0ULL, 0ULL);
    }

    __device__ Secp256K1Base() {
        limbs[0] = 0;
        limbs[1] = 0;
        limbs[2] = 0;
        limbs[3] = 0;
    }

    __device__ Secp256K1Base(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        limbs[0] = a;
        limbs[1] = b;
        limbs[2] = c;
        limbs[3] = d;
    }
};

class Secp256K1Scalar {
public:
    uint64_t limbs[4];

    __device__ static BigUint order(){
        BigUint result;
        result.digits[0] = 0xD0364141;
        result.digits[1] = 0xBFD25E8C;
        result.digits[2] = 0xAF48A03B;
        result.digits[3] = 0xBAAEDCE6;
        result.digits[4] = 0xFFFFFFFE;
        result.digits[5] = 0xFFFFFFFF;
        result.digits[6] = 0xFFFFFFFF;
        result.digits[7] = 0xFFFFFFFF;
        result.length = 8;

        return result;
    }

    __device__ static Secp256K1Scalar GLV_BETA(){
        return Secp256K1Scalar(
            13923278643952681454ULL,
            11308619431505398165ULL,
            7954561588662645993ULL,
            8856726876819556112ULL
        );
    }

    __device__ static Secp256K1Scalar GLV_S(){
        return Secp256K1Scalar(
            16069571880186789234ULL,
            1310022930574435960ULL,
            11900229862571533402ULL,
            6008836872998760672ULL
        );
    }

    __device__ static Secp256K1Scalar A1(){
        return Secp256K1Scalar(16747920425669159701ULL, 3496713202691238861ULL, 0ULL, 0ULL);
    }

    __device__ static Secp256K1Scalar A2(){
        return Secp256K1Scalar(6323353552219852760ULL, 1498098850674701302ULL, 1ULL, 0ULL);
    }

    __device__ static Secp256K1Scalar MINUS_B1(){
        return Secp256K1Scalar(8022177200260244675ULL, 16448129721693014056ULL, 0ULL, 0ULL);
    }

    __device__ static Secp256K1Scalar B2(){
        return Secp256K1Scalar(16747920425669159701ULL, 3496713202691238861ULL, 0ULL, 0ULL);
    }

    __device__ static Secp256K1Scalar one(){
        return Secp256K1Scalar(1ULL, 0ULL, 0ULL, 0ULL);
    }

    __device__ Secp256K1Scalar() {
        limbs[0] = 0ULL;
        limbs[1] = 0ULL;
        limbs[2] = 0ULL;
        limbs[3] = 0ULL;
    }

    __device__ Secp256K1Scalar(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
        limbs[0] = a;
        limbs[1] = b;
        limbs[2] = c;
        limbs[3] = d;
    }

    __device__ Secp256K1Scalar(const Secp256K1Scalar& other) {
        for (int i = 0; i < 4; i++) {
            limbs[i] = other.limbs[i];
        }
    }

    __device__ int is_zero() const {
        return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
    }

    __device__ Secp256K1Scalar& operator=(const Secp256K1Scalar& other) {
        if (this != &other) {
            for (int i = 0; i < 4; i++) {
                limbs[i] = other.limbs[i];
            }
        }
        return *this;
    }

    __device__ static Secp256K1Scalar from_noncanonical_biguint(BigUint a){
        Secp256K1Scalar result;
        for (int i = 0; i < 4; i++) {
            uint64_t carry = uint64_t(a.digits[2 * i]) | (uint64_t(a.digits[2 * i + 1]) << 32);
            result.limbs[i] = carry;
        }
        return result;
    }

    __device__ BigUint to_canonical_biguint() const {
        BigUint result;
        for (int i = 0; i < 4; i++) {
            uint64_t carry = limbs[i];
            result.digits[2 * i] = (uint32_t)carry;
            result.digits[2 * i + 1] = (uint32_t)(carry >> 32);
            result.length += 2;
        }

        result.trim();
        if (result >= Secp256K1Scalar::order()) {
            result = result - Secp256K1Scalar::order();
        }
        return result;
    }

    // __device__ inline Secp256K1Scalar operator-() const{
    //     if (this->is_zero()) {
    //         return *this;
    //     }
    //     else {
    //         BigUint tmp = this->to_canonical_biguint();
    //         tmp = Secp256K1Scalar::order() - tmp;
    //         Secp256K1Scalar result = Secp256K1Scalar::from_noncanonical_biguint(tmp);
    //         return result;
    //     }
    // }

    __device__ Secp256K1Scalar operator+(const Secp256K1Scalar& other) const {
        BigUint tmp = this->to_canonical_biguint() + other.to_canonical_biguint();
        if (tmp >= Secp256K1Scalar::order()) {
            tmp = tmp - Secp256K1Scalar::order();
        }
        Secp256K1Scalar result = Secp256K1Scalar::from_noncanonical_biguint(tmp);
        return result;
    }

    // __device__ Secp256K1Scalar operator-(const Secp256K1Scalar& other) const {
    //     *this + (-other);
    // }

    __device__ Secp256K1Scalar operator-(const Secp256K1Scalar& other) const {
        if (other.is_zero()) {
            return *this;
        }

        Secp256K1Scalar tmp = Secp256K1Scalar::from_noncanonical_biguint(Secp256K1Scalar::order() - other.to_canonical_biguint());
        Secp256K1Scalar result = *this + tmp;
        return result;
    }

    __device__ Secp256K1Scalar operator*(const Secp256K1Scalar& other) const {
        BigUint tmp = this->to_canonical_biguint() * other.to_canonical_biguint();
        tmp = tmp % Secp256K1Scalar::order();
        Secp256K1Scalar result = Secp256K1Scalar::from_noncanonical_biguint(tmp);
        return result;
    }

    __device__ Secp256K1Scalar operator*= (const Secp256K1Scalar& other) {
        *this = *this * other;
        return *this;
    }
};
