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

struct HsvRgbParams {
    ulong dims[4], outputStrides[4], inputStrides[4];
};

kernel void hsv2rgb_float(const device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant HsvRgbParams& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    const ulong total = p.dims[0] * p.dims[1] * p.dims[3];
    if (gid >= total) return;
    ulong q = gid;
    const ulong x = q % p.dims[0]; q /= p.dims[0];
    const ulong y = q % p.dims[1];
    const ulong w = q / p.dims[1];
    const ulong ii = x*p.inputStrides[0] + y*p.inputStrides[1] +
                     w*p.inputStrides[3];
    const ulong oo = x*p.outputStrides[0] + y*p.outputStrides[1] +
                     w*p.outputStrides[3];
    const float h = input[ii];
    const float s = input[ii + p.inputStrides[2]];
    const float v = input[ii + 2*p.inputStrides[2]];
    const int m = int(h * 6.0f);
    const float f = h * 6.0f - float(m);
    const float a = v * (1.0f - s);
    const float b = v * (1.0f - f*s);
    const float c = v * (1.0f - (1.0f-f)*s);
    float3 rgb;
    switch (m % 6) {
        case 0: rgb = float3(v, c, a); break;
        case 1: rgb = float3(b, v, a); break;
        case 2: rgb = float3(a, v, c); break;
        case 3: rgb = float3(a, b, v); break;
        case 4: rgb = float3(c, a, v); break;
        default: rgb = float3(v, a, b); break;
    }
    output[oo] = rgb.x;
    output[oo + p.outputStrides[2]] = rgb.y;
    output[oo + 2*p.outputStrides[2]] = rgb.z;
}

kernel void rgb2hsv_float(const device float* input [[buffer(0)]],
                          device float* output [[buffer(1)]],
                          constant HsvRgbParams& p [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    const ulong total = p.dims[0] * p.dims[1] * p.dims[3];
    if (gid >= total) return;
    ulong q = gid;
    const ulong x = q % p.dims[0]; q /= p.dims[0];
    const ulong y = q % p.dims[1];
    const ulong w = q / p.dims[1];
    const ulong ii = x*p.inputStrides[0] + y*p.inputStrides[1] +
                     w*p.inputStrides[3];
    const ulong oo = x*p.outputStrides[0] + y*p.outputStrides[1] +
                     w*p.outputStrides[3];
    const float r = input[ii];
    const float g = input[ii + p.inputStrides[2]];
    const float b = input[ii + 2*p.inputStrides[2]];
    const float hi = max(max(r, g), b);
    const float lo = min(min(r, g), b);
    const float delta = hi - lo;
    float h = 0.0f;
    if (hi != lo) {
        if (hi == r) h = (g-b)/delta + (g < b ? 6.0f : 0.0f);
        if (hi == g) h = (b-r)/delta + 2.0f;
        if (hi == b) h = (r-g)/delta + 4.0f;
        h /= 6.0f;
    }
    output[oo] = h;
    output[oo + p.outputStrides[2]] = hi == 0.0f ? 0.0f : delta/hi;
    output[oo + 2*p.outputStrides[2]] = hi;
}
