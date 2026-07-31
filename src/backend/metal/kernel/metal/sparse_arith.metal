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
    uint columns;
    uint rhsStride;
    uint csr;
    uint reverse;
    uint operation;
};

struct SparseCsrArithParams {
    uint rows;
    uint lhsNonzeros;
    uint rhsNonzeros;
    uint operation;
};

#define DEFINE_SPARSE_ARITH(NAME, TYPE, ZERO, MUL, DIV)                       \
inline TYPE sparse_arith_apply_##NAME(TYPE left, TYPE right, uint operation) { \
    switch (operation) {                                                       \
        case 0: return left + right;                                           \
        case 1: return left - right;                                           \
        case 2: return MUL(left, right);                                       \
        default: return DIV(left, right);                                      \
    }                                                                          \
}                                                                              \
kernel void sparse_arith_dense_##NAME(                                         \
    device TYPE* output [[buffer(0)]], const device TYPE* values [[buffer(1)]],\
    const device int* rowIndices [[buffer(2)]],                                \
    const device int* columnIndices [[buffer(3)]],                             \
    const device TYPE* rhs [[buffer(4)]],                                      \
    constant SparseArithParams& p [[buffer(5)]],                               \
    uint gid [[thread_position_in_grid]]) {                                    \
    if (gid >= p.nonzeros) return;                                              \
    uint row = uint(rowIndices[gid]);                                           \
    if (p.csr) {                                                                \
        uint low = 0;                                                           \
        uint high = p.rows;                                                     \
        while (low + 1 < high) {                                                \
            const uint middle = (low + high) / 2;                              \
            if (uint(rowIndices[middle]) <= gid) low = middle;                 \
            else high = middle;                                                 \
        }                                                                        \
        row = low;                                                              \
    }                                                                            \
    const uint column = uint(columnIndices[gid]);                               \
    if (row >= p.rows || column >= p.columns) return;                           \
    const TYPE dense = rhs[row + column * p.rhsStride];                         \
    const TYPE sparse = values[gid];                                            \
    const TYPE left = p.reverse ? dense : sparse;                               \
    const TYPE right = p.reverse ? sparse : dense;                              \
    output[row + column * p.rhsStride] =                                       \
        sparse_arith_apply_##NAME(left, right, p.operation);                    \
}                                                                               \
kernel void sparse_arith_values_##NAME(                                        \
    device TYPE* values [[buffer(0)]], const device int* rowIndices [[buffer(1)]],\
    const device int* columnIndices [[buffer(2)]], const device TYPE* rhs [[buffer(3)]],\
    constant SparseArithParams& p [[buffer(4)]],                               \
    uint gid [[thread_position_in_grid]]) {                                    \
    if (gid >= p.nonzeros) return;                                              \
    uint row = uint(rowIndices[gid]);                                           \
    if (p.csr) {                                                                \
        uint low = 0;                                                           \
        uint high = p.rows;                                                     \
        while (low + 1 < high) {                                                \
            const uint middle = (low + high) / 2;                              \
            if (uint(rowIndices[middle]) <= gid) low = middle;                 \
            else high = middle;                                                 \
        }                                                                        \
        row = low;                                                              \
    }                                                                            \
    const uint column = uint(columnIndices[gid]);                               \
    if (row >= p.rows || column >= p.columns) return;                           \
    const TYPE dense = rhs[row + column * p.rhsStride];                         \
    const TYPE sparse = values[gid];                                            \
    const TYPE left = p.reverse ? dense : sparse;                               \
    const TYPE right = p.reverse ? sparse : dense;                              \
    values[gid] = sparse_arith_apply_##NAME(left, right, p.operation);          \
}                                                                               \
kernel void sparse_csr_arith_count_##NAME(                                      \
    device int* outputRows [[buffer(0)]], const device int* lhsRows [[buffer(1)]],\
    const device int* lhsColumns [[buffer(2)]], const device int* rhsRows [[buffer(3)]],\
    const device int* rhsColumns [[buffer(4)]],                                 \
    constant SparseCsrArithParams& p [[buffer(5)]],                             \
    uint row [[thread_position_in_grid]]) {                                     \
    if (row >= p.rows) return;                                                   \
    const int lhsEnd = lhsRows[row + 1];                                        \
    const int rhsEnd = rhsRows[row + 1];                                        \
    int lhs = lhsRows[row];                                                       \
    int rhs = rhsRows[row];                                                       \
    uint count = 0;                                                             \
    while (lhs < lhsEnd && rhs < rhsEnd) {                                      \
        const int lhsColumn = lhsColumns[lhs];                                  \
        const int rhsColumn = rhsColumns[rhs];                                  \
        lhs += lhsColumn <= rhsColumn;                                          \
        rhs += lhsColumn >= rhsColumn;                                          \
        ++count;                                                                 \
    }                                                                            \
    count += uint(lhsEnd - lhs) + uint(rhsEnd - rhs);                            \
    outputRows[row] = int(count);                                                \
}                                                                               \
kernel void sparse_csr_arith_prefix_##NAME(                                     \
    device int* outputRows [[buffer(0)]], constant SparseCsrArithParams& p [[buffer(1)]],\
    uint gid [[thread_position_in_grid]]) {                                      \
    if (gid != 0) return;                                                        \
    int offset = 0;                                                              \
    for (uint row = 0; row < p.rows; ++row) {                                    \
        const int count = outputRows[row];                                       \
        outputRows[row] = offset;                                                \
        offset += count;                                                         \
    }                                                                            \
    outputRows[p.rows] = offset;                                                 \
}                                                                               \
kernel void sparse_csr_arith_merge_##NAME(                                      \
    device TYPE* outputValues [[buffer(0)]], device int* outputColumns [[buffer(1)]],\
    const device int* outputRows [[buffer(2)]],                                 \
    const device TYPE* lhsValues [[buffer(3)]], const device int* lhsRows [[buffer(4)]],\
    const device int* lhsColumns [[buffer(5)]], const device TYPE* rhsValues [[buffer(6)]],\
    const device int* rhsRows [[buffer(7)]], const device int* rhsColumns [[buffer(8)]],\
    constant SparseCsrArithParams& p [[buffer(9)]],                              \
    uint row [[thread_position_in_grid]]) {                                     \
    if (row >= p.rows) return;                                                   \
    const int lhsEnd = lhsRows[row + 1];                                        \
    const int rhsEnd = rhsRows[row + 1];                                        \
    int lhs = lhsRows[row];                                                       \
    int rhs = rhsRows[row];                                                       \
    int output = outputRows[row];                                                \
    while (lhs < lhsEnd && rhs < rhsEnd) {                                      \
        const int lhsColumn = lhsColumns[lhs];                                  \
        const int rhsColumn = rhsColumns[rhs];                                  \
        const bool takeLhs = lhsColumn <= rhsColumn;                            \
        const bool takeRhs = lhsColumn >= rhsColumn;                            \
        const TYPE lhsValue = takeLhs ? lhsValues[lhs] : ZERO;                  \
        const TYPE rhsValue = takeRhs ? rhsValues[rhs] : ZERO;                  \
        outputValues[output] = sparse_arith_apply_##NAME(                      \
            lhsValue, rhsValue, p.operation);                                   \
        outputColumns[output] = takeLhs ? lhsColumn : rhsColumn;                \
        lhs += takeLhs;                                                          \
        rhs += takeRhs;                                                          \
        ++output;                                                                \
    }                                                                            \
    while (lhs < lhsEnd) {                                                       \
        outputValues[output] = sparse_arith_apply_##NAME(                       \
            lhsValues[lhs], ZERO, p.operation);                                  \
        outputColumns[output++] = lhsColumns[lhs++];                             \
    }                                                                            \
    while (rhs < rhsEnd) {                                                       \
        outputValues[output] = sparse_arith_apply_##NAME(                       \
            ZERO, rhsValues[rhs], p.operation);                                  \
        outputColumns[output++] = rhsColumns[rhs++];                             \
    }                                                                            \
}

inline float sparse_mul_float(float lhs, float rhs) { return lhs * rhs; }
inline float sparse_div_float(float lhs, float rhs) { return lhs / rhs; }
DEFINE_SPARSE_ARITH(float, float, 0.0f, sparse_mul_float, sparse_div_float)

inline float2 sparse_mul_cfloat(float2 lhs, float2 rhs) {
    return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                  lhs.x * rhs.y + lhs.y * rhs.x);
}
inline float2 sparse_div_cfloat(float2 lhs, float2 rhs) {
    const float scale = rhs.x * rhs.x + rhs.y * rhs.y;
    return float2((lhs.x * rhs.x + lhs.y * rhs.y) / scale,
                  (lhs.y * rhs.x - lhs.x * rhs.y) / scale);
}
DEFINE_SPARSE_ARITH(cfloat, float2, float2(0.0f), sparse_mul_cfloat,
                    sparse_div_cfloat)

#undef DEFINE_SPARSE_ARITH
