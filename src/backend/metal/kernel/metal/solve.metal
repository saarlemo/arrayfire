/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/
#include <metal_stdlib>
using namespace metal;

struct SolveParams {
    ulong rows;
    ulong columns;
    ulong rhsColumns;
    ulong aStrides[4];
    ulong bStrides[4];
    ulong pivotStrides[4];
    uint batch;
    uint batchZ;
    uint upper;
    uint unit;
};

float solveDivide(float a, float b) { return a / b; }
float2 solveDivide(float2 a, float2 b) {
    const float d = dot(b, b);
    return float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / d;
}

float solveMultiply(float a, float b) { return a * b; }
float2 solveMultiply(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline ulong solveBatchOffset(const ulong stride2, const ulong stride3,
                              const uint batch, const uint batchZ) {
    const ulong z = ulong(batch % batchZ);
    const ulong w = ulong(batch / batchZ);
    return z * stride2 + w * stride3;
}

#define DEFINE_TRIANGULAR(NAME, TYPE)                                      \
kernel void NAME(const device TYPE* a [[buffer(0)]],                       \
                 device TYPE* b [[buffer(1)]],                            \
                 constant SolveParams& p [[buffer(2)]],                   \
                 uint gid [[thread_position_in_grid]]) {                  \
    if (gid >= p.batch * p.rhsColumns) return;                              \
    const uint batch = gid / uint(p.rhsColumns);                            \
    const ulong rhs = ulong(gid % uint(p.rhsColumns));                     \
    const ulong abase = solveBatchOffset(p.aStrides[2], p.aStrides[3],      \
                                         batch, p.batchZ);                  \
    const ulong bbase = solveBatchOffset(p.bStrides[2], p.bStrides[3],      \
                                         batch, p.batchZ) +                 \
                        rhs * p.bStrides[1];                               \
    if (!p.upper) {                                                         \
        for (ulong i = 0; i < p.rows; ++i) {                                \
            TYPE value = b[bbase + i * p.bStrides[0]];                     \
            for (ulong j = 0; j < i; ++j)                                   \
                value -= solveMultiply(a[abase + i * p.aStrides[0] +       \
                                         j * p.aStrides[1]],               \
                                       b[bbase + j * p.bStrides[0]]);        \
            if (!p.unit)                                                    \
                value = solveDivide(                                        \
                    value, a[abase + i * p.aStrides[0] +                   \
                              i * p.aStrides[1]]);                          \
            b[bbase + i * p.bStrides[0]] = value;                           \
        }                                                                   \
    } else {                                                                \
        for (ulong ii = 0; ii < p.rows; ++ii) {                              \
            const ulong i = p.rows - 1 - ii;                                \
            TYPE value = b[bbase + i * p.bStrides[0]];                      \
            for (ulong j = i + 1; j < p.rows; ++j)                          \
                value -= solveMultiply(a[abase + i * p.aStrides[0] +       \
                                         j * p.aStrides[1]],               \
                                       b[bbase + j * p.bStrides[0]]);        \
            if (!p.unit)                                                    \
                value = solveDivide(                                        \
                    value, a[abase + i * p.aStrides[0] +                   \
                              i * p.aStrides[1]]);                          \
            b[bbase + i * p.bStrides[0]] = value;                           \
        }                                                                   \
    }                                                                       \
}

#define DEFINE_LU_SOLVE(NAME, TYPE)                                        \
kernel void NAME(const device TYPE* a [[buffer(0)]],                       \
                 const device int* pivot [[buffer(1)]],                    \
                 device TYPE* b [[buffer(2)]],                             \
                 constant SolveParams& p [[buffer(3)]],                   \
                 uint gid [[thread_position_in_grid]]) {                  \
    if (gid >= p.batch * p.rhsColumns) return;                              \
    const uint batch = gid / uint(p.rhsColumns);                            \
    const ulong rhs = ulong(gid % uint(p.rhsColumns));                     \
    const ulong abase = solveBatchOffset(p.aStrides[2], p.aStrides[3],      \
                                         batch, p.batchZ);                  \
    const ulong bbase = solveBatchOffset(p.bStrides[2], p.bStrides[3],      \
                                         batch, p.batchZ) +                 \
                        rhs * p.bStrides[1];                               \
    const ulong pbase = solveBatchOffset(p.pivotStrides[2],                 \
                                         p.pivotStrides[3], batch, p.batchZ);\
    for (ulong k = 0; k < p.rows; ++k) {                                    \
        const int pv = pivot[pbase + k * p.pivotStrides[0]] - 1;            \
        if (pv >= 0 && ulong(pv) != k) {                                    \
            const ulong x = bbase + k * p.bStrides[0];                     \
            const ulong y = bbase + ulong(pv) * p.bStrides[0];              \
            const TYPE tmp = b[x]; b[x] = b[y]; b[y] = tmp;                 \
        }                                                                       \
    }                                                                       \
    for (ulong i = 0; i < p.rows; ++i) {                                    \
        TYPE value = b[bbase + i * p.bStrides[0]];                          \
        for (ulong j = 0; j < i; ++j)                                       \
            value -= solveMultiply(a[abase + i * p.aStrides[0] +            \
                                      j * p.aStrides[1]],                   \
                                   b[bbase + j * p.bStrides[0]]);            \
        b[bbase + i * p.bStrides[0]] = value;                               \
    }                                                                       \
    for (ulong ii = 0; ii < p.rows; ++ii) {                                 \
        const ulong i = p.rows - 1 - ii;                                    \
        TYPE value = b[bbase + i * p.bStrides[0]];                          \
        for (ulong j = i + 1; j < p.rows; ++j)                              \
            value -= solveMultiply(a[abase + i * p.aStrides[0] +            \
                                      j * p.aStrides[1]],                   \
                                   b[bbase + j * p.bStrides[0]]);            \
        value = solveDivide(value,                                         \
                            a[abase + i * p.aStrides[0] +                   \
                               i * p.aStrides[1]]);                         \
        b[bbase + i * p.bStrides[0]] = value;                               \
    }                                                                       \
}

DEFINE_TRIANGULAR(solve_triangular_float, float)
DEFINE_TRIANGULAR(solve_triangular_cfloat, float2)
DEFINE_LU_SOLVE(solve_lu_float, float)
DEFINE_LU_SOLVE(solve_lu_cfloat, float2)
