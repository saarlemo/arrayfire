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

struct MedfiltParams {
    ulong dims[4], outputStrides[4], inputStrides[4];
    uint windowLength, windowWidth, padding, oneDimensional;
};

template<typename T>
inline T medfiltValue(const device T* input, constant MedfiltParams& p,
                      const long x, const long y, const ulong z,
                      const ulong w, const uint i, const uint j,
                      const uint height) {
    long ix = x + long(i) - long(p.windowLength / 2);
    long iy = y + long(j) - long(height / 2);
    const bool outside =
        ix < 0 || iy < 0 || ix >= long(p.dims[0]) || iy >= long(p.dims[1]);
    if (outside && p.padding == 0) return T(0);

    if (ix < 0) ix = -ix;
    if (iy < 0) iy = -iy;
    if (ix >= long(p.dims[0])) ix = 2 * (long(p.dims[0]) - 1) - ix;
    if (iy >= long(p.dims[1])) iy = 2 * (long(p.dims[1]) - 1) - iy;

    const ulong offset = ulong(ix) * p.inputStrides[0] +
                         ulong(iy) * p.inputStrides[1] +
                         z * p.inputStrides[2] + w * p.inputStrides[3];
    return input[offset];
}

template<typename T>
inline void medfiltImpl(const device T* input, device T* output,
                        constant MedfiltParams& p, const uint gid) {
    const ulong total = p.dims[0] * p.dims[1] * p.dims[2] * p.dims[3];
    if (gid >= total) return;

    ulong q = gid;
    const long x = long(q % p.dims[0]);
    q /= p.dims[0];
    const long y = long(q % p.dims[1]);
    q /= p.dims[1];
    const ulong z = q % p.dims[2];
    const ulong w = q / p.dims[2];

    const uint height = p.oneDimensional ? 1 : p.windowWidth;
    const uint count = p.windowLength * height;
    const uint lowerRank = (count - 1) / 2;
    const uint upperRank = count / 2;

    T lower = T(0);
    T upper = T(0);
    bool lowerFound = false;
    bool upperFound = false;

    // Determine the two central order statistics without a fixed-size local
    // array. This permits arbitrary API-valid window sizes at the cost of
    // rescanning the window for each distinct candidate.
    for (uint cj = 0; cj < height && !(lowerFound && upperFound); ++cj) {
        for (uint ci = 0; ci < p.windowLength && !(lowerFound && upperFound);
             ++ci) {
            const T candidate =
                medfiltValue(input, p, x, y, z, w, ci, cj, height);
            uint less = 0;
            uint equal = 0;
            for (uint j = 0; j < height; ++j) {
                for (uint i = 0; i < p.windowLength; ++i) {
                    const T value =
                        medfiltValue(input, p, x, y, z, w, i, j, height);
                    less += value < candidate;
                    equal += value == candidate;
                }
            }
            if (!lowerFound && less <= lowerRank &&
                lowerRank < less + equal) {
                lower = candidate;
                lowerFound = true;
            }
            if (!upperFound && less <= upperRank &&
                upperRank < less + equal) {
                upper = candidate;
                upperFound = true;
            }
        }
    }

    const T result =
        (count & 1u) != 0 ? upper : T((lower + upper) / T(2));
    const ulong outputOffset = ulong(x) * p.outputStrides[0] +
                               ulong(y) * p.outputStrides[1] +
                               z * p.outputStrides[2] +
                               w * p.outputStrides[3];
    output[outputOffset] = result;
}

#define DEFINE_MEDFILT(NAME, TYPE)                                         \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 device TYPE* output [[buffer(1)]],                        \
                 constant MedfiltParams& p [[buffer(2)]],                  \
                 uint gid [[thread_position_in_grid]]) {                   \
    medfiltImpl(input, output, p, gid);                                    \
}

DEFINE_MEDFILT(medfilt_float, float)
DEFINE_MEDFILT(medfilt_int, int)
DEFINE_MEDFILT(medfilt_uint, uint)
DEFINE_MEDFILT(medfilt_char, char)
DEFINE_MEDFILT(medfilt_uchar, uchar)
DEFINE_MEDFILT(medfilt_short, short)
DEFINE_MEDFILT(medfilt_ushort, ushort)
