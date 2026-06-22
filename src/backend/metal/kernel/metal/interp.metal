/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

inline float interpLinearFloat(float first, float second, float ratio) {
    if (ratio == 0.0f) return first;
    if (ratio == 1.0f) return second;
    if (isinf(first) && !isinf(second)) return first;
    if (!isinf(first) && isinf(second)) return second;
    if (isinf(first) && isinf(second) && signbit(first) == signbit(second))
        return first;
    return (1.0f - ratio) * first + ratio * second;
}

kernel void interp1_linear_float(
    const device float* input [[buffer(0)]],
    const device float* xPositions [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ApproxParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;
    ulong q = gid, c[4];
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
    if (!isfinite(x) || x < 0.0f ||
        float(p.inputDims[p.xDimension]) < x + 1.0f) {
        output[oi] = p.offGrid;
        return;
    }
    const ulong grid = ulong(floor(x));
    const ulong next = min(grid + 1, p.inputDims[p.xDimension] - 1);
    const float ratio = x - float(grid);
    const float first = input[ii + grid * p.inputStrides[p.xDimension]];
    const float second = input[ii + next * p.inputStrides[p.xDimension]];
    output[oi] = interpLinearFloat(first, second, ratio);
}

kernel void interp2_linear_float(
    const device float* input [[buffer(0)]],
    const device float* xPositions [[buffer(1)]],
    const device float* yPositions [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant ApproxParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;
    ulong q = gid, c[4];
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
    if (!isfinite(x) || !isfinite(y) || x < 0.0f ||
        float(p.inputDims[p.xDimension]) < x + 1.0f || y < 0.0f ||
        float(p.inputDims[p.yDimension]) < y + 1.0f) {
        output[oi] = p.offGrid;
        return;
    }
    const ulong gx = ulong(floor(x)), gy = ulong(floor(y));
    const ulong nx = min(gx + 1, p.inputDims[p.xDimension] - 1);
    const ulong ny = min(gy + 1, p.inputDims[p.yDimension] - 1);
    const float xr = x - float(gx), yr = y - float(gy);
    const ulong xs = p.inputStrides[p.xDimension];
    const ulong ys = p.inputStrides[p.yDimension];
    const float v00 = input[ii + gx * xs + gy * ys];
    const float v10 = input[ii + nx * xs + gy * ys];
    const float v01 = input[ii + gx * xs + ny * ys];
    const float v11 = input[ii + nx * xs + ny * ys];
    const float top = interpLinearFloat(v00, v10, xr);
    const float bottom = interpLinearFloat(v01, v11, xr);
    output[oi] = interpLinearFloat(top, bottom, yr);
}
