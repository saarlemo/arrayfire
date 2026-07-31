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

struct RotateParams {
    ulong odims[4], ostrides[4], idims[4], istrides[4];
    float transform[6];
    uint method;
};

template<typename T>
T linearInterp(T left, T right, float ratio) {
    return fma(T(ratio), right - left, left);
}

template<typename T>
T cubicInterp(T v0, T v1, T v2, T v3, float ratio, bool spline) {
    T a0, a1, a2, a3;
    if (spline) {
        a0 = -0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3;
        a1 = v0 - 2.5f * v1 + 2.0f * v2 - 0.5f * v3;
        a2 = -0.5f * v0 + 0.5f * v2;
        a3 = v1;
    } else {
        a0 = v3 - v2 - v0 + v1;
        a1 = v0 - v1 - a0;
        a2 = v2 - v0;
        a3 = v1;
    }
    const float ratio2 = ratio * ratio;
    return a0 * ratio2 * ratio + a1 * ratio2 + a2 * ratio + a3;
}

#define DEFINE_ROTATE(NAME, TYPE, WORK_TYPE)                                \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant RotateParams& p [[buffer(2)]],                    \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total =                                                       \
        p.odims[0] * p.odims[1] * p.odims[2] * p.odims[3];                  \
    if (gid >= total) return;                                                \
    ulong q = gid;                                                           \
    const ulong x = q % p.odims[0];                                         \
    q /= p.odims[0];                                                         \
    const ulong y = q % p.odims[1];                                         \
    q /= p.odims[1];                                                         \
    const ulong z = q % p.odims[2];                                         \
    const ulong w = q / p.odims[2];                                         \
    volatile float fxx = float(x) * p.transform[0];                          \
    volatile float fxy = float(y) * p.transform[1];                          \
    volatile float fyx = float(x) * p.transform[3];                          \
    volatile float fyy = float(y) * p.transform[4];                          \
    const float fx = fxx + fxy + p.transform[2];                             \
    const float fy = fyx + fyy + p.transform[5];                             \
    const ulong oo = x * p.ostrides[0] + y * p.ostrides[1] +                \
                     z * p.ostrides[2] + w * p.ostrides[3];                  \
    const ulong batchOffset = z * p.istrides[2] + w * p.istrides[3];        \
    /* The transform coefficients are quantized to 1e-3. Account for the     \
       small FMA-vs-scalar rounding difference at an image boundary, then    \
       clamp the interpolation footprint below. */                           \
    const bool inside = fx >= -0.0011f && fy >= -0.0011f &&                 \
                        fx < float(p.idims[0]) + 0.0011f &&                  \
                        fy < float(p.idims[1]) + 0.0011f;                    \
    if (p.method == 0 || p.method == 4) {                                   \
        const long ix = p.method == 4 ? long(floor(fx)) : long(round(fx));  \
        const long iy = p.method == 4 ? long(floor(fy)) : long(round(fy));  \
        if (ix >= 0 && iy >= 0 && ix < long(p.idims[0]) &&                  \
            iy < long(p.idims[1])) {                                        \
            output[oo] = input[ulong(ix) * p.istrides[0] +                  \
                               ulong(iy) * p.istrides[1] + batchOffset];     \
        } else {                                                             \
            output[oo] = TYPE(0);                                            \
        }                                                                    \
        return;                                                              \
    }                                                                        \
    if (!inside) {                                                           \
        output[oo] = TYPE(0);                                                \
        return;                                                              \
    }                                                                        \
    const long gx = long(floor(fx));                                         \
    const long gy = long(floor(fy));                                         \
    const float xfraction = fx - floor(fx);                                  \
    const float yfraction = fy - floor(fy);                                  \
    const float xr0 = xfraction < 0.00001f                                  \
                          ? 0.0f                                             \
                          : (xfraction > 0.99999f ? 1.0f : xfraction);        \
    const float yr0 = yfraction < 0.00001f                                  \
                          ? 0.0f                                             \
                          : (yfraction > 0.99999f ? 1.0f : yfraction);        \
    if (p.method == 2 || p.method == 6) {                                   \
        const long x0 = clamp(gx, 0l, long(p.idims[0]) - 1);                \
        const long x1 = clamp(gx + 1, 0l, long(p.idims[0]) - 1);            \
        const long y0 = clamp(gy, 0l, long(p.idims[1]) - 1);                \
        const long y1 = clamp(gy + 1, 0l, long(p.idims[1]) - 1);            \
        float xr = xr0, yr = yr0;                                            \
        if (p.method == 6) {                                                 \
            xr = (1.0f - cos(xr * M_PI_F)) * 0.5f;                          \
            yr = (1.0f - cos(yr * M_PI_F)) * 0.5f;                          \
        }                                                                    \
        const WORK_TYPE v00 = WORK_TYPE(input[ulong(x0) * p.istrides[0] +   \
                                              ulong(y0) * p.istrides[1] +   \
                                              batchOffset]);                 \
        const WORK_TYPE v01 = WORK_TYPE(input[ulong(x1) * p.istrides[0] +   \
                                              ulong(y0) * p.istrides[1] +   \
                                              batchOffset]);                 \
        const WORK_TYPE v10 = WORK_TYPE(input[ulong(x0) * p.istrides[0] +   \
                                              ulong(y1) * p.istrides[1] +   \
                                              batchOffset]);                 \
        const WORK_TYPE v11 = WORK_TYPE(input[ulong(x1) * p.istrides[0] +   \
                                              ulong(y1) * p.istrides[1] +   \
                                              batchOffset]);                 \
        output[oo] = TYPE(linearInterp(linearInterp(v00, v01, xr),          \
                                       linearInterp(v10, v11, xr), yr));     \
        return;                                                              \
    }                                                                        \
    const bool spline = p.method == 9;                                       \
    WORK_TYPE rows[4];                                                       \
    for (long j = 0; j < 4; ++j) {                                          \
        const long sy = clamp(gy + j - 1, 0l, long(p.idims[1]) - 1);        \
        WORK_TYPE values[4];                                                 \
        for (long i = 0; i < 4; ++i) {                                      \
            const long sx =                                                  \
                clamp(gx + i - 1, 0l, long(p.idims[0]) - 1);                \
            values[i] = WORK_TYPE(input[ulong(sx) * p.istrides[0] +         \
                                          ulong(sy) * p.istrides[1] +       \
                                          batchOffset]);                     \
        }                                                                    \
        rows[j] = cubicInterp(values[0], values[1], values[2], values[3],    \
                              xr0, spline);                                  \
    }                                                                        \
    output[oo] = TYPE(cubicInterp(rows[0], rows[1], rows[2], rows[3], yr0,   \
                                  spline));                                  \
}

DEFINE_ROTATE(rotate_float, float, float)
DEFINE_ROTATE(rotate_cfloat, float2, float2)
DEFINE_ROTATE(rotate_int, int, float)
DEFINE_ROTATE(rotate_uint, uint, float)
DEFINE_ROTATE(rotate_long, long, float)
DEFINE_ROTATE(rotate_ulong, ulong, float)
DEFINE_ROTATE(rotate_char, char, float)
DEFINE_ROTATE(rotate_uchar, uchar, float)
DEFINE_ROTATE(rotate_short, short, float)
DEFINE_ROTATE(rotate_ushort, ushort, float)
