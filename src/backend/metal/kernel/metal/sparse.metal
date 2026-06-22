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

struct SparseParams {
    uint rows;
    uint columns;
    uint outputStride;
    uint nonzeros;
};

kernel void sparse_zero_float(device float* output [[buffer(0)]],
                              constant SparseParams& p [[buffer(1)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid < p.rows * p.columns) output[gid] = 0.0f;
}

kernel void sparse_csr_to_dense_float(
    const device float* values [[buffer(0)]], const device int* rows [[buffer(1)]],
    const device int* columns [[buffer(2)]], device float* output [[buffer(3)]],
    constant SparseParams& p [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    for (int i = rows[row]; i < rows[row + 1]; ++i)
        output[uint(columns[i]) * p.outputStride + row] = values[i];
}

kernel void sparse_coo_to_dense_float(
    const device float* values [[buffer(0)]], const device int* rows [[buffer(1)]],
    const device int* columns [[buffer(2)]], device float* output [[buffer(3)]],
    constant SparseParams& p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.nonzeros) return;
    output[uint(columns[gid]) * p.outputStride + uint(rows[gid])] = values[gid];
}
