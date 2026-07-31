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

struct SparseMatmulParams {
    ulong sourceRows;
    ulong sourceColumns;
    ulong outputRows;
    ulong rhsColumns;
    ulong outputStrides[4];
    ulong rhsStrides[4];
    uint operation;
};

kernel void sparse_matmul_float(
    device float* output [[buffer(0)]],
    const device float* values [[buffer(1)]],
    const device int* rowIndex [[buffer(2)]],
    const device int* columnIndex [[buffer(3)]],
    const device float* rhs [[buffer(4)]],
    constant SparseMatmulParams& params [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong row = position.x;
    const ulong column = position.y;
    if (row >= params.outputRows || column >= params.rhsColumns) return;

    float result = 0.0f;
    if (params.operation == 0) {
        for (int element = rowIndex[row]; element < rowIndex[row + 1];
             ++element) {
            result += values[element] *
                      rhs[ulong(columnIndex[element]) * params.rhsStrides[0] +
                          column * params.rhsStrides[1]];
        }
    } else {
        for (ulong sourceRow = 0; sourceRow < params.sourceRows; ++sourceRow) {
            for (int element = rowIndex[sourceRow];
                 element < rowIndex[sourceRow + 1]; ++element) {
                if (ulong(columnIndex[element]) == row) {
                    result += values[element] *
                              rhs[sourceRow * params.rhsStrides[0] +
                                  column * params.rhsStrides[1]];
                }
            }
        }
    }
    output[row * params.outputStrides[0] +
           column * params.outputStrides[1]] = result;
}

inline float2 sparseComplexMultiply(const float2 lhs, const float2 rhs) {
    return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                  lhs.x * rhs.y + lhs.y * rhs.x);
}

kernel void sparse_matmul_cfloat(
    device float2* output [[buffer(0)]],
    const device float2* values [[buffer(1)]],
    const device int* rowIndex [[buffer(2)]],
    const device int* columnIndex [[buffer(3)]],
    const device float2* rhs [[buffer(4)]],
    constant SparseMatmulParams& params [[buffer(5)]],
    uint2 position [[thread_position_in_grid]]) {
    const ulong row = position.x;
    const ulong column = position.y;
    if (row >= params.outputRows || column >= params.rhsColumns) return;

    float2 result = float2(0.0f, 0.0f);
    if (params.operation == 0) {
        for (int element = rowIndex[row]; element < rowIndex[row + 1];
             ++element) {
            result += sparseComplexMultiply(
                values[element],
                rhs[ulong(columnIndex[element]) * params.rhsStrides[0] +
                    column * params.rhsStrides[1]]);
        }
    } else {
        for (ulong sourceRow = 0; sourceRow < params.sourceRows; ++sourceRow) {
            for (int element = rowIndex[sourceRow];
                 element < rowIndex[sourceRow + 1]; ++element) {
                if (ulong(columnIndex[element]) == row) {
                    float2 value = values[element];
                    if (params.operation == 2) value.y = -value.y;
                    result += sparseComplexMultiply(
                        value, rhs[sourceRow * params.rhsStrides[0] +
                                   column * params.rhsStrides[1]]);
                }
            }
        }
    }
    output[row * params.outputStrides[0] +
           column * params.outputStrides[1]] = result;
}
