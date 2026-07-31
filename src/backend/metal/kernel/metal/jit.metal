#include <metal_stdlib>
using namespace metal;

template<typename T> inline T af_jit_add(T lhs, T rhs) { return lhs + rhs; }
template<typename T> inline T af_jit_sub(T lhs, T rhs) { return lhs - rhs; }
template<typename T> inline T af_jit_mul(T lhs, T rhs) { return lhs * rhs; }
template<typename T> inline T af_jit_div(T lhs, T rhs) { return lhs / rhs; }

inline float af_jit_cabs(float2 value) {
    return sqrt(value.x * value.x + value.y * value.y);
}
inline float2 af_jit_mul(float2 lhs, float2 rhs) {
    return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                  lhs.x * rhs.y + lhs.y * rhs.x);
}
inline float2 af_jit_div(float2 lhs, float2 rhs) {
    float rhsAbs = af_jit_cabs(rhs);
    float inverse = 1.0f / rhsAbs;
    float x = inverse * rhs.x;
    float y = inverse * rhs.y;
    return float2((lhs.x * x + lhs.y * y) * inverse,
                  (lhs.y * x - lhs.x * y) * inverse);
}

inline float2 af_jit_exp(float2 value) {
    float magnitude = exp(value.x);
    return float2(magnitude * cos(value.y), magnitude * sin(value.y));
}
inline float2 af_jit_log(float2 value) {
    return float2(log(af_jit_cabs(value)), atan2(value.y, value.x));
}
inline float2 af_jit_sqrt(float2 value) {
    float magnitude = af_jit_cabs(value);
    float realPart = sqrt(max(0.0f, (magnitude + value.x) * 0.5f));
    float imagPart = sqrt(max(0.0f, (magnitude - value.x) * 0.5f));
    return float2(realPart, value.y < 0.0f ? -imagPart : imagPart);
}
inline float2 af_jit_sin(float2 value) {
    return float2(sin(value.x) * cosh(value.y),
                  cos(value.x) * sinh(value.y));
}
inline float2 af_jit_cos(float2 value) {
    return float2(cos(value.x) * cosh(value.y),
                  -sin(value.x) * sinh(value.y));
}
inline float2 af_jit_tan(float2 value) {
    return af_jit_div(af_jit_sin(value), af_jit_cos(value));
}
inline float2 af_jit_sinh(float2 value) {
    return float2(sinh(value.x) * cos(value.y),
                  cosh(value.x) * sin(value.y));
}
inline float2 af_jit_cosh(float2 value) {
    return float2(cosh(value.x) * cos(value.y),
                  sinh(value.x) * sin(value.y));
}
inline float2 af_jit_tanh(float2 value) {
    return af_jit_div(af_jit_sinh(value), af_jit_cosh(value));
}
inline float2 af_jit_asinh(float2 value) {
    return af_jit_log(value +
                      af_jit_sqrt(af_jit_mul(value, value) +
                                  float2(1.0f, 0.0f)));
}
inline float2 af_jit_acosh(float2 value) {
    return af_jit_log(value +
                      af_jit_mul(af_jit_sqrt(value + float2(1.0f, 0.0f)),
                                 af_jit_sqrt(value - float2(1.0f, 0.0f))));
}
inline float2 af_jit_atanh(float2 value) {
    return 0.5f *
           (af_jit_log(float2(1.0f, 0.0f) + value) -
            af_jit_log(float2(1.0f, 0.0f) - value));
}
inline float2 af_jit_asin(float2 value) {
    float2 iz(-value.y, value.x);
    float2 inside =
        iz + af_jit_sqrt(float2(1.0f, 0.0f) - af_jit_mul(value, value));
    float2 logged = af_jit_log(inside);
    return float2(logged.y, -logged.x);
}
inline float2 af_jit_acos(float2 value) {
    float2 result = af_jit_asin(value);
    return float2(1.5707963267948966f - result.x, -result.y);
}
inline float2 af_jit_atan(float2 value) {
    float2 iz(-value.y, value.x);
    float2 difference =
        af_jit_log(float2(1.0f, 0.0f) - iz) -
        af_jit_log(float2(1.0f, 0.0f) + iz);
    return float2(-0.5f * difference.y, 0.5f * difference.x);
}

#define AF_JIT_REAL_UNARY(NAME) \
template<typename T> inline T af_jit_##NAME(T value) { return NAME(value); }
AF_JIT_REAL_UNARY(sin)
AF_JIT_REAL_UNARY(cos)
AF_JIT_REAL_UNARY(tan)
AF_JIT_REAL_UNARY(asin)
AF_JIT_REAL_UNARY(acos)
AF_JIT_REAL_UNARY(atan)
AF_JIT_REAL_UNARY(sinh)
AF_JIT_REAL_UNARY(cosh)
AF_JIT_REAL_UNARY(tanh)
AF_JIT_REAL_UNARY(asinh)
AF_JIT_REAL_UNARY(acosh)
AF_JIT_REAL_UNARY(atanh)
AF_JIT_REAL_UNARY(exp)
AF_JIT_REAL_UNARY(log)
AF_JIT_REAL_UNARY(log10)
AF_JIT_REAL_UNARY(log2)
AF_JIT_REAL_UNARY(sqrt)
AF_JIT_REAL_UNARY(rsqrt)
AF_JIT_REAL_UNARY(trunc)
AF_JIT_REAL_UNARY(round)
AF_JIT_REAL_UNARY(ceil)
AF_JIT_REAL_UNARY(floor)
#undef AF_JIT_REAL_UNARY
template<typename T> inline char af_jit_signbit(T value) {
    return signbit(value);
}

template<typename T> inline char af_jit_eq(T lhs, T rhs) { return lhs == rhs; }
template<typename T> inline char af_jit_neq(T lhs, T rhs) { return lhs != rhs; }
template<typename T> inline char af_jit_lt(T lhs, T rhs) { return lhs < rhs; }
template<typename T> inline char af_jit_le(T lhs, T rhs) { return lhs <= rhs; }
template<typename T> inline char af_jit_gt(T lhs, T rhs) { return lhs > rhs; }
template<typename T> inline char af_jit_ge(T lhs, T rhs) { return lhs >= rhs; }
inline char af_jit_eq(float2 lhs, float2 rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y;
}
inline char af_jit_neq(float2 lhs, float2 rhs) {
    return !af_jit_eq(lhs, rhs);
}
inline char af_jit_lt(float2 lhs, float2 rhs) {
    return af_jit_cabs(lhs) < af_jit_cabs(rhs);
}
inline char af_jit_le(float2 lhs, float2 rhs) {
    return af_jit_cabs(lhs) <= af_jit_cabs(rhs);
}
inline char af_jit_gt(float2 lhs, float2 rhs) {
    return af_jit_cabs(lhs) > af_jit_cabs(rhs);
}
inline char af_jit_ge(float2 lhs, float2 rhs) {
    return af_jit_cabs(lhs) >= af_jit_cabs(rhs);
}

template<typename T> inline char af_jit_and(T lhs, T rhs) {
    return (lhs != T(0)) && (rhs != T(0));
}
template<typename T> inline char af_jit_or(T lhs, T rhs) {
    return (lhs != T(0)) || (rhs != T(0));
}
inline char af_jit_and(float2 lhs, float2 rhs) {
    return af_jit_cabs(lhs) != 0.0f && af_jit_cabs(rhs) != 0.0f;
}
inline char af_jit_or(float2 lhs, float2 rhs) {
    return af_jit_cabs(lhs) != 0.0f || af_jit_cabs(rhs) != 0.0f;
}

template<typename T> inline T af_jit_bitand(T lhs, T rhs) { return lhs & rhs; }
template<typename T> inline T af_jit_bitor(T lhs, T rhs) { return lhs | rhs; }
template<typename T> inline T af_jit_bitxor(T lhs, T rhs) { return lhs ^ rhs; }
template<typename T> inline T af_jit_bitshiftl(T lhs, T rhs) {
    return lhs << rhs;
}
template<typename T> inline T af_jit_bitshiftr(T lhs, T rhs) {
    return lhs >> rhs;
}

template<typename T> inline T af_jit_min(T lhs, T rhs) {
    return lhs < rhs ? lhs : rhs;
}
template<typename T> inline T af_jit_max(T lhs, T rhs) {
    return lhs > rhs ? lhs : rhs;
}
inline float2 af_jit_min(float2 lhs, float2 rhs) {
    return af_jit_cabs(lhs) < af_jit_cabs(rhs) ? lhs : rhs;
}
inline float2 af_jit_max(float2 lhs, float2 rhs) {
    return af_jit_cabs(lhs) > af_jit_cabs(rhs) ? lhs : rhs;
}

template<typename T> inline T af_jit_rem(T lhs, T rhs) { return lhs % rhs; }
template<typename T> inline T af_jit_mod(T lhs, T rhs) { return lhs % rhs; }
inline float af_jit_rem(float lhs, float rhs) {
    return lhs - rint(precise::divide(lhs, rhs)) * rhs;
}
inline half af_jit_rem(half lhs, half rhs) {
    float left = float(lhs);
    float right = float(rhs);
    return half(left - rint(precise::divide(left, right)) * right);
}
inline float af_jit_fmod(float lhs, float rhs) {
    uint xbits = as_type<uint>(lhs);
    uint ybits = as_type<uint>(rhs);
    const uint sign = xbits & 0x80000000u;
    int xExponent = int((xbits >> 23) & 0xffu);
    int yExponent = int((ybits >> 23) & 0xffu);
    xbits &= 0x7fffffffu;
    ybits &= 0x7fffffffu;

    if (ybits == 0u || ybits > 0x7f800000u ||
        xbits >= 0x7f800000u) {
        float product = lhs * rhs;
        return product / product;
    }
    if (xbits < ybits) return lhs;
    if (xbits == ybits) return as_type<float>(sign);

    if (xExponent == 0) {
        uint value = xbits << 9;
        while ((value >> 31) == 0u) {
            --xExponent;
            value <<= 1;
        }
        xbits <<= uint(-xExponent + 1);
    } else {
        xbits = (xbits & 0x007fffffu) | 0x00800000u;
    }
    if (yExponent == 0) {
        uint value = ybits << 9;
        while ((value >> 31) == 0u) {
            --yExponent;
            value <<= 1;
        }
        ybits <<= uint(-yExponent + 1);
    } else {
        ybits = (ybits & 0x007fffffu) | 0x00800000u;
    }

    while (xExponent > yExponent) {
        uint difference = xbits - ybits;
        if ((difference >> 31) == 0u) {
            if (difference == 0u) return as_type<float>(sign);
            xbits = difference;
        }
        xbits <<= 1;
        --xExponent;
    }
    uint difference = xbits - ybits;
    if ((difference >> 31) == 0u) {
        if (difference == 0u) return as_type<float>(sign);
        xbits = difference;
    }
    while ((xbits >> 23) == 0u) {
        xbits <<= 1;
        --xExponent;
    }
    if (xExponent > 0) {
        xbits = (xbits - 0x00800000u) | (uint(xExponent) << 23);
    } else {
        xbits >>= uint(-xExponent + 1);
    }
    return as_type<float>(xbits | sign);
}
inline float af_jit_mod(float lhs, float rhs) {
    return af_jit_fmod(lhs, rhs);
}
inline half af_jit_mod(half lhs, half rhs) {
    return half(af_jit_fmod(float(lhs), float(rhs)));
}

inline float af_jit_pow(float lhs, float rhs) { return pow(lhs, rhs); }
inline half af_jit_pow(half lhs, half rhs) {
    return half(pow(float(lhs), float(rhs)));
}
template<typename T> inline T af_jit_integer_pow(T base, T exponent) {
    float baseValue = float(base);
    float magnitude = exponent == T(2)
                          ? baseValue * baseValue
                          : precise::pow(abs(baseValue), float(exponent));
    if (base < T(0) && (exponent & T(1))) magnitude = -magnitude;
    return T(rint(magnitude));
}
#define AF_JIT_INTEGER_POW(TYPE) \
inline TYPE af_jit_pow(TYPE lhs, TYPE rhs) { \
    return af_jit_integer_pow(lhs, rhs); \
}
AF_JIT_INTEGER_POW(char)
AF_JIT_INTEGER_POW(uchar)
AF_JIT_INTEGER_POW(short)
AF_JIT_INTEGER_POW(ushort)
AF_JIT_INTEGER_POW(int)
AF_JIT_INTEGER_POW(uint)
AF_JIT_INTEGER_POW(long)
AF_JIT_INTEGER_POW(ulong)
#undef AF_JIT_INTEGER_POW
template<typename T> inline float2 af_jit_cplx(T lhs, T rhs) {
    return float2(float(lhs), float(rhs));
}
template<typename T> inline T af_jit_atan2(T lhs, T rhs) {
    return atan2(lhs, rhs);
}
template<typename T> inline T af_jit_hypot(T lhs, T rhs) {
    return sqrt(lhs * lhs + rhs * rhs);
}

template<typename T> inline float af_jit_cast_float(T value) {
    return float(value);
}
inline float af_jit_cast_float(float2 value) {
    return af_jit_cabs(value);
}
template<typename T> inline int af_jit_cast_int(T value) {
    return int(value);
}
inline int af_jit_cast_int(float2 value) {
    return int(af_jit_cabs(value));
}
template<typename T> inline uint af_jit_cast_uint(T value) {
    return uint(value);
}
inline uint af_jit_cast_uint(float2 value) {
    return uint(af_jit_cabs(value));
}
template<typename T> inline char af_jit_cast_char(T value) {
    return char(value);
}
inline char af_jit_cast_char(float2 value) {
    return char(af_jit_cabs(value));
}
template<typename T> inline uchar af_jit_cast_uchar(T value) {
    return uchar(value);
}
inline uchar af_jit_cast_uchar(float2 value) {
    return uchar(af_jit_cabs(value));
}
template<typename T> inline short af_jit_cast_short(T value) {
    return short(value);
}
inline short af_jit_cast_short(float2 value) {
    return short(af_jit_cabs(value));
}
template<typename T> inline ushort af_jit_cast_ushort(T value) {
    return ushort(value);
}
inline ushort af_jit_cast_ushort(float2 value) {
    return ushort(af_jit_cabs(value));
}
template<typename T> inline long af_jit_cast_long(T value) {
    return long(value);
}
inline long af_jit_cast_long(float2 value) {
    return long(af_jit_cabs(value));
}
template<typename T> inline ulong af_jit_cast_ulong(T value) {
    return ulong(value);
}
inline ulong af_jit_cast_ulong(float2 value) {
    return ulong(af_jit_cabs(value));
}
template<typename T> inline half af_jit_cast_half(T value) {
    return half(value);
}
inline half af_jit_cast_half(float2 value) {
    return half(af_jit_cabs(value));
}
template<typename T> inline float2 af_jit_cast_cfloat(T value) {
    return float2(float(value), 0.0f);
}
inline float2 af_jit_cast_cfloat(float2 value) {
    return value;
}
template<typename T> inline char af_jit_cast_bool(T value) {
    return char(value != T(0));
}
inline char af_jit_cast_bool(float2 value) {
    return char(value.x != 0.0f || value.y != 0.0f);
}

template<typename T> inline T af_jit_sigmoid(T value) {
    return T(1) / (T(1) + exp(-value));
}
template<typename T> inline T af_jit_cbrt(T value) {
    T magnitude = value < T(0) ? -value : value;
    T result = pow(magnitude, T(1) / T(3));
    return value < T(0) ? -result : result;
}
template<typename T> inline T af_jit_expm1(T value) {
    return exp(value) - T(1);
}
template<typename T> inline T af_jit_log1p(T value) {
    return log(T(1) + value);
}
inline float af_jit_erf(float value) {
    float sign = value < 0.0f ? -1.0f : 1.0f;
    float x = value < 0.0f ? -value : value;
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float polynomial =
        (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
          0.284496736f) *
             t +
         0.254829592f) *
        t;
    return sign * (1.0f - polynomial * exp(-x * x));
}
inline half af_jit_erf(half value) {
    return half(af_jit_erf(float(value)));
}
template<typename T> inline T af_jit_erfc(T value) {
    return T(1) - af_jit_erf(value);
}
inline float af_jit_lgamma_positive(float value) {
    float z = value - 1.0f;
    float series = 0.99999994f;
    series += 676.5204f / (z + 1.0f);
    series -= 1259.1392f / (z + 2.0f);
    series += 771.3234f / (z + 3.0f);
    series -= 176.61504f / (z + 4.0f);
    series += 12.507343f / (z + 5.0f);
    series -= 0.1385711f / (z + 6.0f);
    series += 0.00000998437f / (z + 7.0f);
    series += 0.00000015056327f / (z + 8.0f);
    float t = z + 7.5f;
    return 0.9189385332f + (z + 0.5f) * log(t) - t + log(series);
}
inline float af_jit_lgamma(float value) {
    constexpr float pi = 3.14159265358979323846f;
    if (value < 0.5f) {
        return log(pi) - log(abs(sin(pi * value))) -
               af_jit_lgamma_positive(1.0f - value);
    }
    return af_jit_lgamma_positive(value);
}
inline half af_jit_lgamma(half value) {
    return half(af_jit_lgamma(float(value)));
}
inline float af_jit_tgamma(float value) {
    constexpr float pi = 3.14159265358979323846f;
    if (value < 0.5f) {
        return pi /
               (sin(pi * value) *
                exp(af_jit_lgamma_positive(1.0f - value)));
    }
    return exp(af_jit_lgamma_positive(value));
}
inline half af_jit_tgamma(half value) {
    return half(af_jit_tgamma(float(value)));
}
template<typename T> inline char af_jit_iszero(T value) {
    return value == T(0);
}
inline char af_jit_iszero(float2 value) {
    return value.x == 0.0f && value.y == 0.0f;
}
template<typename T> inline char af_jit_isinf(T value) { return isinf(value); }
inline char af_jit_isinf(float2 value) {
    return isinf(value.x) || isinf(value.y);
}
template<typename T> inline char af_jit_isnan(T value) { return isnan(value); }
inline char af_jit_isnan(float2 value) {
    return isnan(value.x) || isnan(value.y);
}
template<typename T> inline T af_jit_noop(T value) { return value; }
template<typename T> inline T af_jit_bitnot(T value) { return ~value; }
template<typename T> inline T af_jit_real(T value) { return value; }
inline float af_jit_real(float2 value) { return value.x; }
template<typename T> inline T af_jit_imag(T) { return T(0); }
inline float af_jit_imag(float2 value) { return value.y; }
template<typename T> inline T af_jit_abs(T value) { return abs(value); }
inline uint af_jit_abs(uint value) { return value; }
inline ulong af_jit_abs(ulong value) { return value; }
inline ushort af_jit_abs(ushort value) { return value; }
inline uchar af_jit_abs(uchar value) { return value; }
inline float af_jit_abs(float2 value) { return af_jit_cabs(value); }
template<typename T> inline T af_jit_conj(T value) { return value; }
inline float2 af_jit_conj(float2 value) {
    return float2(value.x, -value.y);
}
