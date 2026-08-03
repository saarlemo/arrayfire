/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <metal_stdlib>
using namespace metal;

struct CholeskyParams {
    ulong n;
    long stride0;
    long stride1;
    uint step;
    uint upper;
    uint phase;
};

template<typename T>
inline T cholConjugate(const T value) {
    return value;
}

inline float2 cholConjugate(const float2 value) {
    return float2(value.x, -value.y);
}

template<typename T>
inline T cholMultiply(const T lhs, const T rhs) {
    return lhs * rhs;
}

inline float2 cholMultiply(const float2 lhs, const float2 rhs) {
    return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                  lhs.x * rhs.y + lhs.y * rhs.x);
}

template<typename T>
inline T cholDivide(const T lhs, const T rhs) {
    return lhs / rhs;
}

inline float2 cholDivide(const float2 lhs, const float2 rhs) {
    const float denominator = rhs.x * rhs.x + rhs.y * rhs.y;
    return float2((lhs.x * rhs.x + lhs.y * rhs.y) / denominator,
                  (lhs.y * rhs.x - lhs.x * rhs.y) / denominator);
}

template<typename T>
inline float cholDiagonal(const T value) {
    return value;
}

inline float cholDiagonal(const float2 value) {
    return value.x;
}

template<typename T>
inline T cholSqrt(const T value) {
    return sqrt(value);
}

inline float2 cholSqrt(const float2 value) {
    return float2(sqrt(value.x), 0.0f);
}

inline long cholIndex(const ulong row, const ulong col,
                      constant CholeskyParams& p) {
    return long(row) * p.stride0 + long(col) * p.stride1;
}

template<typename T>
inline void choleskyImpl(device T* data, device int* info,
                         constant CholeskyParams& p, const uint gid) {
    if (gid >= p.n || *info != 0) return;

    const ulong step = p.step;
    if (p.upper) {
        const ulong column = gid;
        if (column < step) return;

        if (p.phase == 0) {
            if (column != step) return;
            T diagonal = data[cholIndex(step, step, p)];
            T correction = T(0);
            T compensation = T(0);
            for (ulong j = 0; j < step; ++j) {
                const T term = cholMultiply(
                    cholConjugate(data[cholIndex(j, step, p)]),
                    data[cholIndex(j, step, p)]);
                const T adjusted = term - compensation;
                const T updated = correction + adjusted;
                compensation = (updated - correction) - adjusted;
                correction = updated;
            }
            diagonal -= correction;
            if (!(cholDiagonal(diagonal) > 0.0f)) {
                *info = int(step + 1);
                return;
            }
            data[cholIndex(step, step, p)] = cholSqrt(diagonal);
        } else {
            if (column <= step) return;
            T value = data[cholIndex(step, column, p)];
            T correction = T(0);
            T compensation = T(0);
            for (ulong j = 0; j < step; ++j) {
                const T term = cholMultiply(
                    cholConjugate(data[cholIndex(j, step, p)]),
                    data[cholIndex(j, column, p)]);
                const T adjusted = term - compensation;
                const T updated = correction + adjusted;
                compensation = (updated - correction) - adjusted;
                correction = updated;
            }
            value -= correction;
            value = cholDivide(value, data[cholIndex(step, step, p)]);
            data[cholIndex(step, column, p)] = value;
        }
    } else {
        const ulong row = gid;
        if (row < step) return;

        if (p.phase == 0) {
            if (row != step) return;
            T diagonal = data[cholIndex(step, step, p)];
            T correction = T(0);
            T compensation = T(0);
            for (ulong j = 0; j < step; ++j) {
                const T term = cholMultiply(
                    data[cholIndex(step, j, p)],
                    cholConjugate(data[cholIndex(step, j, p)]));
                const T adjusted = term - compensation;
                const T updated = correction + adjusted;
                compensation = (updated - correction) - adjusted;
                correction = updated;
            }
            diagonal -= correction;
            if (!(cholDiagonal(diagonal) > 0.0f)) {
                *info = int(step + 1);
                return;
            }
            data[cholIndex(step, step, p)] = cholSqrt(diagonal);
        } else {
            if (row <= step) return;
            T value = data[cholIndex(row, step, p)];
            T correction = T(0);
            T compensation = T(0);
            for (ulong j = 0; j < step; ++j) {
                const T term = cholMultiply(
                    data[cholIndex(row, j, p)],
                    cholConjugate(data[cholIndex(step, j, p)]));
                const T adjusted = term - compensation;
                const T updated = correction + adjusted;
                compensation = (updated - correction) - adjusted;
                correction = updated;
            }
            value -= correction;
            value = cholDivide(value, data[cholIndex(step, step, p)]);
            data[cholIndex(row, step, p)] = value;
        }
    }
}

kernel void cholesky_float(device float* data [[buffer(0)]],
                           device int* info [[buffer(1)]],
                           constant CholeskyParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    choleskyImpl(data, info, p, gid);
}

kernel void cholesky_cfloat(device float2* data [[buffer(0)]],
                            device int* info [[buffer(1)]],
                            constant CholeskyParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    choleskyImpl(data, info, p, gid);
}
