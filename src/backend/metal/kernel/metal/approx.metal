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

struct ApproxParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    ulong xDims[4];
    ulong xStrides[4];
    ulong yStrides[4];
    uint xDimension;
    uint yDimension;
    uint method;
    float xBegin;
    float xStep;
    float yBegin;
    float yStep;
    float offGrid;
};

kernel void approx1_float(const device float* input [[buffer(0)]],
                          const device float* xPositions [[buffer(1)]],
                          device float* output [[buffer(2)]],
                          constant ApproxParams& p [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;
    ulong q = gid;
    ulong c[4];
    c[0] = q % p.outputDims[0]; q /= p.outputDims[0];
    c[1] = q % p.outputDims[1]; q /= p.outputDims[1];
    c[2] = q % p.outputDims[2]; c[3] = q / p.outputDims[2];
    ulong oi = 0, ii = 0, xi = 0;
    for (uint d = 0; d < 4; ++d) {
        oi += c[d] * p.outputStrides[d];
        if (d != p.xDimension) ii += c[d] * p.inputStrides[d];
        if (p.xDims[d] > 1) xi += c[d] * p.xStrides[d];
    }
    const float x = (xPositions[xi] - p.xBegin) / p.xStep;
    if (x < 0.0f || float(p.inputDims[p.xDimension]) < x + 1.0f) {
        output[oi] = p.offGrid;
        return;
    }
    const long index = p.method == 1 ? long(floor(x)) : long(floor(x + 0.5f));
    output[oi] = input[ii + ulong(index) * p.inputStrides[p.xDimension]];
}

kernel void approx2_float(const device float* input [[buffer(0)]],
                          const device float* xPositions [[buffer(1)]],
                          const device float* yPositions [[buffer(2)]],
                          device float* output [[buffer(3)]],
                          constant ApproxParams& p [[buffer(4)]],
                          uint gid [[thread_position_in_grid]]) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;
    ulong q = gid;
    ulong c[4];
    c[0] = q % p.outputDims[0]; q /= p.outputDims[0];
    c[1] = q % p.outputDims[1]; q /= p.outputDims[1];
    c[2] = q % p.outputDims[2]; c[3] = q / p.outputDims[2];
    ulong oi = 0, ii = 0, xi = 0, yi = 0;
    for (uint d = 0; d < 4; ++d) {
        oi += c[d] * p.outputStrides[d];
        if (d != p.xDimension && d != p.yDimension)
            ii += c[d] * p.inputStrides[d];
        if (p.xDims[d] > 1) {
            xi += c[d] * p.xStrides[d];
            yi += c[d] * p.yStrides[d];
        }
    }
    const float x = (xPositions[xi] - p.xBegin) / p.xStep;
    const float y = (yPositions[yi] - p.yBegin) / p.yStep;
    if (x < 0.0f || float(p.inputDims[p.xDimension]) < x + 1.0f ||
        y < 0.0f || float(p.inputDims[p.yDimension]) < y + 1.0f) {
        output[oi] = p.offGrid;
        return;
    }
    const long ix = p.method == 1 ? long(floor(x)) : long(floor(x + 0.5f));
    const long iy = p.method == 1 ? long(floor(y)) : long(floor(y + 0.5f));
    ii += ulong(ix) * p.inputStrides[p.xDimension];
    ii += ulong(iy) * p.inputStrides[p.yDimension];
    output[oi] = input[ii];
}
