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

struct SparseArithParams {
    uint nonzeros;
    uint rows;
    uint rhsStride;
    uint csr;
    uint reverse;
    uint operation;
};

kernel void sparse_arith_float(device float* values [[buffer(0)]],
                               const device int* rowIndices [[buffer(1)]],
                               const device int* columnIndices [[buffer(2)]],
                               const device float* rhs [[buffer(3)]],
                               constant SparseArithParams& p [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= p.nonzeros) return;
    uint row = uint(rowIndices[gid]);
    if (p.csr) {
        uint low = 0, high = p.rows;
        while (low + 1 < high) {
            const uint middle = (low + high) / 2;
            if (uint(rowIndices[middle]) <= gid) low = middle;
            else high = middle;
        }
        row = low;
    }
    const uint column = uint(columnIndices[gid]);
    const float dense = rhs[row + column * p.rhsStride];
    const float sparse = values[gid];
    const float left = p.reverse ? dense : sparse;
    const float right = p.reverse ? sparse : dense;
    switch (p.operation) {
        case 0: values[gid] = left + right; break;
        case 1: values[gid] = left - right; break;
        case 2: values[gid] = left * right; break;
        default: values[gid] = left / right; break;
    }
}
