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

struct SparseDenseToCsrParams {
    uint rows;
    uint columns;
    uint inputStride;
    uint nonzeros;
};

struct SparseCooToCsrParams {
    uint rows;
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

kernel void sparse_zero_cfloat(device float2* output [[buffer(0)]],
                               constant SparseParams& p [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid < p.rows * p.columns) output[gid] = float2(0.0f);
}

kernel void sparse_csr_to_dense_cfloat(
    const device float2* values [[buffer(0)]], const device int* rows [[buffer(1)]],
    const device int* columns [[buffer(2)]], device float2* output [[buffer(3)]],
    constant SparseParams& p [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    for (int i = rows[row]; i < rows[row + 1]; ++i)
        output[uint(columns[i]) * p.outputStride + row] = values[i];
}

kernel void sparse_coo_to_dense_cfloat(
    const device float2* values [[buffer(0)]], const device int* rows [[buffer(1)]],
    const device int* columns [[buffer(2)]], device float2* output [[buffer(3)]],
    constant SparseParams& p [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= p.nonzeros) return;
    output[uint(columns[gid]) * p.outputStride + uint(rows[gid])] = values[gid];
}

#define SPARSE_IS_ZERO_float(value) ((value) == 0.0f)
#define SPARSE_IS_ZERO_cfloat(value) ((value).x == 0.0f && (value).y == 0.0f)

#define DEFINE_SPARSE_CONVERSION(NAME, TYPE, ZERO)                           \
kernel void sparse_dense_to_csr_count_##NAME(                                 \
    const device TYPE* input [[buffer(0)]], device int* rowIdx [[buffer(1)]], \
    constant SparseDenseToCsrParams& p [[buffer(2)]],                        \
    uint row [[thread_position_in_grid]]) {                                   \
    if (row >= p.rows) return;                                                \
    uint count = 0;                                                           \
    for (uint column = 0; column < p.columns; ++column) {                     \
        if (!ZERO(input[row + column * p.inputStride])) ++count;              \
    }                                                                         \
    rowIdx[row] = int(count);                                                  \
}                                                                             \
kernel void sparse_dense_to_csr_prefix_##NAME(                                \
    device int* rowIdx [[buffer(0)]],                                         \
    constant SparseDenseToCsrParams& p [[buffer(1)]],                         \
    uint gid [[thread_position_in_grid]]) {                                   \
    if (gid != 0) return;                                                      \
    int offset = 0;                                                           \
    for (uint row = 0; row < p.rows; ++row) {                                  \
        const int count = rowIdx[row];                                        \
        rowIdx[row] = offset;                                                 \
        offset += count;                                                       \
    }                                                                         \
    rowIdx[p.rows] = offset;                                                   \
}                                                                             \
kernel void sparse_dense_to_csr_scatter_##NAME(                               \
    const device TYPE* input [[buffer(0)]], device TYPE* values [[buffer(1)]],\
    device int* columns [[buffer(2)]], const device int* rowIdx [[buffer(3)]],\
    constant SparseDenseToCsrParams& p [[buffer(4)]],                         \
    uint row [[thread_position_in_grid]]) {                                   \
    if (row >= p.rows) return;                                                 \
    uint output = uint(rowIdx[row]);                                          \
    for (uint column = 0; column < p.columns; ++column) {                      \
        const TYPE value = input[row + column * p.inputStride];               \
        if (!ZERO(value)) {                                                    \
            if (output < p.nonzeros) {                                        \
                values[output] = value;                                       \
                columns[output] = int(column);                                \
            }                                                                   \
            ++output;                                                          \
        }                                                                       \
    }                                                                           \
}                                                                               \
kernel void sparse_csr_to_coo_##NAME(                                          \
    const device TYPE* inputValues [[buffer(0)]],                              \
    const device int* inputRows [[buffer(1)]],                                 \
    const device int* inputColumns [[buffer(2)]],                              \
    device TYPE* outputValues [[buffer(3)]], device int* outputRows [[buffer(4)]],\
    device int* outputColumns [[buffer(5)]],                                   \
    constant SparseParams& p [[buffer(6)]],                                    \
    uint gid [[thread_position_in_grid]]) {                                    \
    if (gid >= p.nonzeros) return;                                             \
    outputValues[gid] = inputValues[gid];                                      \
    outputColumns[gid] = inputColumns[gid];                                    \
    uint low = 0;                                                              \
    uint high = p.rows;                                                        \
    while (low + 1 < high) {                                                   \
        const uint middle = (low + high) / 2;                                 \
        if (uint(inputRows[middle]) <= gid) low = middle;                     \
        else high = middle;                                                    \
    }                                                                           \
    outputRows[gid] = int(low);                                                \
}                                                                               \
kernel void sparse_coo_to_csr_clear_##NAME(                                    \
    device int* rowIdx [[buffer(0)]],                                          \
    constant SparseCooToCsrParams& p [[buffer(1)]],                             \
    uint row [[thread_position_in_grid]]) {                                    \
    if (row < p.rows) rowIdx[row] = 0;                                         \
}                                                                               \
kernel void sparse_coo_to_csr_count_##NAME(                                    \
    const device int* inputRows [[buffer(0)]],                                 \
    device atomic_int* rowIdx [[buffer(1)]],                                   \
    constant SparseCooToCsrParams& p [[buffer(2)]],                             \
    uint gid [[thread_position_in_grid]]) {                                    \
    if (gid >= p.nonzeros) return;                                             \
    const int row = inputRows[gid];                                             \
    if (row >= 0 && uint(row) < p.rows)                                        \
        atomic_fetch_add_explicit(&rowIdx[row], 1, memory_order_relaxed);      \
}                                                                               \
kernel void sparse_coo_to_csr_prefix_##NAME(                                   \
    device int* rowIdx [[buffer(0)]],                                          \
    constant SparseCooToCsrParams& p [[buffer(1)]],                             \
    uint gid [[thread_position_in_grid]]) {                                    \
    if (gid != 0) return;                                                       \
    int offset = 0;                                                             \
    for (uint row = 0; row < p.rows; ++row) {                                   \
        const int count = rowIdx[row];                                         \
        rowIdx[row] = offset;                                                  \
        offset += count;                                                        \
    }                                                                           \
    rowIdx[p.rows] = offset;                                                    \
}                                                                               \
kernel void sparse_coo_to_csr_cursor_##NAME(                                   \
    const device int* rowIdx [[buffer(0)]], device int* cursor [[buffer(1)]],  \
    constant SparseCooToCsrParams& p [[buffer(2)]],                             \
    uint row [[thread_position_in_grid]]) {                                    \
    if (row < p.rows) cursor[row] = rowIdx[row];                                \
}                                                                               \
kernel void sparse_coo_to_csr_scatter_##NAME(                                  \
    const device TYPE* inputValues [[buffer(0)]],                              \
    const device int* inputRows [[buffer(1)]],                                 \
    const device int* inputColumns [[buffer(2)]],                              \
    device TYPE* outputValues [[buffer(3)]],                                   \
    device int* outputColumns [[buffer(4)]],                                   \
    device atomic_int* cursor [[buffer(5)]],                                   \
    constant SparseCooToCsrParams& p [[buffer(6)]],                             \
    uint gid [[thread_position_in_grid]]) {                                    \
    if (gid >= p.nonzeros) return;                                             \
    const int row = inputRows[gid];                                             \
    if (row < 0 || uint(row) >= p.rows) return;                                \
    const int output = atomic_fetch_add_explicit(                              \
        &cursor[row], 1, memory_order_relaxed);                                \
    outputValues[output] = inputValues[gid];                                    \
    outputColumns[output] = inputColumns[gid];                                  \
}                                                                               \
kernel void sparse_coo_to_csr_scatter_sorted_##NAME(                            \
    const device TYPE* inputValues [[buffer(0)]],                              \
    const device int* inputColumns [[buffer(1)]],                              \
    device TYPE* outputValues [[buffer(2)]],                                   \
    device int* outputColumns [[buffer(3)]],                                   \
    const device int* outputRows [[buffer(4)]],                                \
    constant SparseCooToCsrParams& p [[buffer(5)]],                             \
    uint row [[thread_position_in_grid]]) {                                    \
    if (row >= p.rows) return;                                                  \
    for (int index = outputRows[row]; index < outputRows[row + 1]; ++index) {  \
        outputValues[index] = inputValues[index];                              \
        outputColumns[index] = inputColumns[index];                            \
    }                                                                           \
}

DEFINE_SPARSE_CONVERSION(float, float, SPARSE_IS_ZERO_float)
DEFINE_SPARSE_CONVERSION(cfloat, float2, SPARSE_IS_ZERO_cfloat)

#undef DEFINE_SPARSE_CONVERSION
#undef SPARSE_IS_ZERO_float
#undef SPARSE_IS_ZERO_cfloat
