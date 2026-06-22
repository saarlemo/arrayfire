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

struct ExampleFunctionParams {
    ulong dims[4];
    ulong outputStrides[4];
    ulong leftStrides[4];
    ulong rightStrides[4];
};

kernel void example_function_float(
    const device float* left [[buffer(0)]],
    const device float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ExampleFunctionParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong total = p.dims[0] * p.dims[1] * p.dims[2] * p.dims[3];
    if (gid >= total) return;

    ulong q = gid;
    const ulong d0 = q % p.dims[0]; q /= p.dims[0];
    const ulong d1 = q % p.dims[1]; q /= p.dims[1];
    const ulong d2 = q % p.dims[2];
    const ulong d3 = q / p.dims[2];
    const ulong outputIndex = d0 * p.outputStrides[0] +
                              d1 * p.outputStrides[1] +
                              d2 * p.outputStrides[2] +
                              d3 * p.outputStrides[3];
    const ulong leftIndex = d0 * p.leftStrides[0] +
                            d1 * p.leftStrides[1] +
                            d2 * p.leftStrides[2] +
                            d3 * p.leftStrides[3];
    const ulong rightIndex = d0 * p.rightStrides[0] +
                             d1 * p.rightStrides[1] +
                             d2 * p.rightStrides[2] +
                             d3 * p.rightStrides[3];
    output[outputIndex] = left[leftIndex] + right[rightIndex];
}
