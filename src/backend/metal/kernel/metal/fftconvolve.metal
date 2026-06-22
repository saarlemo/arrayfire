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

struct FFTConvolveParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong signalDims[4];
    ulong signalStrides[4];
    ulong filterDims[4];
    ulong filterStrides[4];
    ulong offset;
    uint kind;
};

kernel void fftconvolve_multiply_float(
    device float* packed [[buffer(0)]],
    constant FFTConvolveParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong complexDim0 = p.outputDims[0] / 2;
    const ulong total = complexDim0 * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;
    ulong q = gid;
    const ulong d0 = q % complexDim0; q /= complexDim0;
    const ulong d1 = q % p.outputDims[1]; q /= p.outputDims[1];
    const ulong d2 = q % p.outputDims[2];
    const ulong d3 = q / p.outputDims[2];
    const ulong index = d3 * p.outputStrides[3] +
                        d2 * p.outputStrides[2] +
                        d1 * p.outputStrides[1] + d0 * 2;
    ulong firstIndex = index;
    ulong secondIndex = p.offset + index;
    ulong outputIndex = index;
    if (p.kind == 1) {
        const ulong filterBatch = p.filterStrides[3] * p.filterDims[3];
        secondIndex = p.offset + index % filterBatch;
    } else if (p.kind == 2) {
        const ulong signalBatch = p.signalStrides[3] * p.signalDims[3];
        firstIndex = index % signalBatch;
        outputIndex = p.offset + index;
    }
    const float ar = packed[firstIndex];
    const float ai = packed[firstIndex + 1];
    const float br = packed[secondIndex];
    const float bi = packed[secondIndex + 1];
    packed[outputIndex] = ar * br - ai * bi;
    packed[outputIndex + 1] = ar * bi + ai * br;
}
