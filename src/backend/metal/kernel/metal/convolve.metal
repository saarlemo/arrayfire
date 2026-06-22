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

struct ConvolveParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong signalDims[4];
    ulong signalStrides[4];
    ulong filterDims[4];
    ulong filterStrides[4];
    uint rank;
    uint expand;
};

kernel void convolve_float(const device float* signal [[buffer(0)]],
                           const device float* filter [[buffer(1)]],
                           device float* output [[buffer(2)]],
                           constant ConvolveParams& p [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;
    ulong q = gid, c[4];
    c[0] = q % p.outputDims[0]; q /= p.outputDims[0];
    c[1] = q % p.outputDims[1]; q /= p.outputDims[1];
    c[2] = q % p.outputDims[2]; c[3] = q / p.outputDims[2];
    ulong oi = 0;
    for (uint d = 0; d < 4; ++d) oi += c[d] * p.outputStrides[d];
    const long sx0 = long(c[0]) + (p.expand ? 0 : long(p.filterDims[0] / 2));
    const long sx1 = long(c[1]) + (p.expand ? 0 : long(p.filterDims[1] / 2));
    const long sx2 = long(c[2]) + (p.expand ? 0 : long(p.filterDims[2] / 2));
    float sum = 0.0f;
    for (ulong f2 = 0; f2 < (p.rank > 2 ? p.filterDims[2] : 1); ++f2) {
        const long z = sx2 - long(f2);
        if (p.rank > 2 && (z < 0 || z >= long(p.signalDims[2]))) continue;
        for (ulong f1 = 0; f1 < (p.rank > 1 ? p.filterDims[1] : 1); ++f1) {
            const long y = sx1 - long(f1);
            if (p.rank > 1 && (y < 0 || y >= long(p.signalDims[1]))) continue;
            for (ulong f0 = 0; f0 < p.filterDims[0]; ++f0) {
                const long x = sx0 - long(f0);
                if (x < 0 || x >= long(p.signalDims[0])) continue;
                ulong si = ulong(x) * p.signalStrides[0];
                if (p.rank > 1) si += ulong(y) * p.signalStrides[1];
                if (p.rank > 2) si += ulong(z) * p.signalStrides[2];
                ulong fi = f0 * p.filterStrides[0] +
                           f1 * p.filterStrides[1] +
                           f2 * p.filterStrides[2];
                sum += signal[si] * filter[fi];
            }
        }
    }
    output[oi] = sum;
}
