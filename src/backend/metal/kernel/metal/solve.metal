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

struct SolveGramParams {
    ulong rows;
    ulong columns;
    ulong rank;
    ulong rhsColumns;
    ulong aStrides[4];
    ulong bStrides[4];
    ulong gramStrides[4];
    ulong rhsStrides[4];
    uint batch;
    uint batchZ;
    uint underdetermined;
};

struct SolveExpandParams {
    ulong rows;
    ulong columns;
    ulong rhsColumns;
    ulong aStrides[4];
    ulong yStrides[4];
    ulong outputStrides[4];
    uint batch;
    uint batchZ;
};

float solveAbsSquared(float x) { return x * x; }
float solveAbsSquared(float2 x) { return dot(x, x); }
float2 solveConj(float2 x) { return float2(x.x, -x.y); }
float solveConj(float x) { return x; }
float solveRealPart(float x) { return x; }
float solveRealPart(float2 x) { return x.x; }

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

#define DEFINE_GENERAL(NAME, TYPE)                                         \
kernel void NAME(device TYPE* a [[buffer(0)]],                             \
                 device TYPE* b [[buffer(1)]],                             \
                 constant SolveParams& p [[buffer(2)]],                   \
                 uint batch [[thread_position_in_grid]]) {                \
    if (batch >= p.batch) return;                                           \
    const ulong abase = solveBatchOffset(p.aStrides[2], p.aStrides[3],       \
                                         batch, p.batchZ);                   \
    const ulong bbase = solveBatchOffset(p.bStrides[2], p.bStrides[3],       \
                                         batch, p.batchZ);                   \
    for (ulong k = 0; k < p.rows; ++k) {                                    \
        ulong pivot = k;                                                    \
        float best = solveAbsSquared(                                       \
            a[abase + k * p.aStrides[0] + k * p.aStrides[1]]);              \
        for (ulong i = k + 1; i < p.rows; ++i) {                            \
            const float candidate = solveAbsSquared(                       \
                a[abase + i * p.aStrides[0] + k * p.aStrides[1]]);          \
            if (candidate > best) { best = candidate; pivot = i; }          \
        }                                                                   \
        if (pivot != k) {                                                    \
            for (ulong j = 0; j < p.columns; ++j) {                          \
                const ulong x = abase + k * p.aStrides[0] +                 \
                                 j * p.aStrides[1];                          \
                const ulong y = abase + pivot * p.aStrides[0] +             \
                                 j * p.aStrides[1];                          \
                const TYPE tmp = a[x]; a[x] = a[y]; a[y] = tmp;              \
            }                                                               \
            for (ulong j = 0; j < p.rhsColumns; ++j) {                        \
                const ulong x = bbase + k * p.bStrides[0] +                 \
                                 j * p.bStrides[1];                          \
                const ulong y = bbase + pivot * p.bStrides[0] +             \
                                 j * p.bStrides[1];                          \
                const TYPE tmp = b[x]; b[x] = b[y]; b[y] = tmp;              \
            }                                                               \
        }                                                                   \
        const TYPE diagonal =                                               \
            a[abase + k * p.aStrides[0] + k * p.aStrides[1]];                \
        for (ulong i = k + 1; i < p.rows; ++i) {                             \
            const ulong ik = abase + i * p.aStrides[0] + k * p.aStrides[1];  \
            a[ik] = solveDivide(a[ik], diagonal);                           \
            for (ulong j = k + 1; j < p.columns; ++j)                       \
                a[abase + i * p.aStrides[0] + j * p.aStrides[1]] -=          \
                    solveMultiply(a[ik],                                      \
                                  a[abase + k * p.aStrides[0] +              \
                                     j * p.aStrides[1]]);                    \
        }                                                                   \
    }                                                                       \
    for (ulong rhs = 0; rhs < p.rhsColumns; ++rhs) {                        \
        for (ulong i = 0; i < p.rows; ++i) {                                 \
            TYPE value = b[bbase + i * p.bStrides[0] +                      \
                           rhs * p.bStrides[1]];                             \
            for (ulong j = 0; j < i; ++j)                                   \
                value -= solveMultiply(a[abase + i * p.aStrides[0] +       \
                                         j * p.aStrides[1]],               \
                                       b[bbase + j * p.bStrides[0] +         \
                                         rhs * p.bStrides[1]]);              \
            b[bbase + i * p.bStrides[0] + rhs * p.bStrides[1]] = value;      \
        }                                                                   \
        for (ulong ii = 0; ii < p.rows; ++ii) {                              \
            const ulong i = p.rows - 1 - ii;                                \
            TYPE value = b[bbase + i * p.bStrides[0] +                       \
                           rhs * p.bStrides[1]];                              \
            for (ulong j = i + 1; j < p.rows; ++j)                           \
                value -= solveMultiply(a[abase + i * p.aStrides[0] +        \
                                         j * p.aStrides[1]],                \
                                       b[bbase + j * p.bStrides[0] +          \
                                         rhs * p.bStrides[1]]);               \
            value = solveDivide(value,                                       \
                                a[abase + i * p.aStrides[0] +                \
                                   i * p.aStrides[1]]);                      \
            b[bbase + i * p.bStrides[0] + rhs * p.bStrides[1]] = value;      \
        }                                                                   \
    }                                                                       \
}

#define DEFINE_GRAM(NAME, TYPE)                                            \
kernel void NAME(const device TYPE* a [[buffer(0)]],                       \
                 const device TYPE* b [[buffer(1)]],                       \
                 device TYPE* gram [[buffer(2)]],                          \
                 device TYPE* rhs [[buffer(3)]],                           \
                 constant SolveGramParams& p [[buffer(4)]],               \
                 uint batch [[thread_position_in_grid]]) {                \
    if (batch >= p.batch) return;                                           \
    const ulong abase = solveBatchOffset(p.aStrides[2], p.aStrides[3],       \
                                         batch, p.batchZ);                   \
    const ulong bbase = solveBatchOffset(p.bStrides[2], p.bStrides[3],       \
                                         batch, p.batchZ);                   \
    const ulong gbase = solveBatchOffset(p.gramStrides[2],                  \
                                         p.gramStrides[3], batch, p.batchZ); \
    const ulong rbase = solveBatchOffset(p.rhsStrides[2],                    \
                                         p.rhsStrides[3], batch, p.batchZ);  \
    if (!p.underdetermined) {                                               \
        for (ulong i = 0; i < p.rank; ++i) {                                \
            for (ulong j = 0; j < p.rank; ++j) {                            \
                TYPE sum = TYPE(0);                                         \
                for (ulong t = 0; t < p.rows; ++t)                          \
                    sum += solveMultiply(                                   \
                        solveConj(a[abase + t * p.aStrides[0] +             \
                                      i * p.aStrides[1]]),                  \
                        a[abase + t * p.aStrides[0] + j * p.aStrides[1]]);   \
                gram[gbase + i * p.gramStrides[0] +                          \
                     j * p.gramStrides[1]] = sum;                            \
            }                                                               \
            for (ulong k = 0; k < p.rhsColumns; ++k) {                       \
                TYPE sum = TYPE(0);                                         \
                for (ulong t = 0; t < p.rows; ++t)                          \
                    sum += solveMultiply(                                   \
                        solveConj(a[abase + t * p.aStrides[0] +             \
                                      i * p.aStrides[1]]),                  \
                        b[bbase + t * p.bStrides[0] + k * p.bStrides[1]]);   \
                rhs[rbase + i * p.rhsStrides[0] + k * p.rhsStrides[1]] = sum; \
            }                                                               \
        }                                                                   \
    } else {                                                                \
        for (ulong i = 0; i < p.rank; ++i) {                                \
            for (ulong j = 0; j < p.rank; ++j) {                            \
                TYPE sum = TYPE(0);                                         \
                for (ulong t = 0; t < p.columns; ++t)                       \
                    sum += solveMultiply(                                   \
                        a[abase + i * p.aStrides[0] + t * p.aStrides[1]],    \
                        solveConj(a[abase + j * p.aStrides[0] +              \
                                      t * p.aStrides[1]]));                 \
                gram[gbase + i * p.gramStrides[0] +                          \
                     j * p.gramStrides[1]] = sum;                            \
            }                                                               \
            for (ulong k = 0; k < p.rhsColumns; ++k)                        \
                rhs[rbase + i * p.rhsStrides[0] + k * p.rhsStrides[1]] =    \
                    b[bbase + i * p.bStrides[0] + k * p.bStrides[1]];        \
        }                                                                   \
    }                                                                       \
}

#define DEFINE_EXPAND(NAME, TYPE)                                           \
kernel void NAME(const device TYPE* a [[buffer(0)]],                       \
                 const device TYPE* y [[buffer(1)]],                       \
                 device TYPE* out [[buffer(2)]],                           \
                 constant SolveExpandParams& p [[buffer(3)]],              \
                 uint gid [[thread_position_in_grid]]) {                  \
    if (gid >= p.batch * p.rhsColumns) return;                              \
    const uint batch = gid / uint(p.rhsColumns);                            \
    const ulong rhs = ulong(gid % uint(p.rhsColumns));                     \
    const ulong abase = solveBatchOffset(p.aStrides[2], p.aStrides[3],       \
                                         batch, p.batchZ);                   \
    const ulong ybase = solveBatchOffset(p.yStrides[2], p.yStrides[3],       \
                                         batch, p.batchZ);                   \
    const ulong obase = solveBatchOffset(p.outputStrides[2],                \
                                         p.outputStrides[3], batch, p.batchZ) + \
                        rhs * p.outputStrides[1];                          \
    for (ulong i = 0; i < p.columns; ++i) {                                 \
        TYPE sum = TYPE(0);                                                 \
        for (ulong t = 0; t < p.rows; ++t)                                   \
            sum += solveMultiply(                                           \
                solveConj(a[abase + t * p.aStrides[0] + i * p.aStrides[1]]),\
                y[ybase + t * p.yStrides[0] + rhs * p.yStrides[1]]);         \
        out[obase + i * p.outputStrides[0]] = sum;                           \
    }                                                                       \
}

DEFINE_TRIANGULAR(solve_triangular_float, float)
DEFINE_TRIANGULAR(solve_triangular_cfloat, float2)
DEFINE_LU_SOLVE(solve_lu_float, float)
DEFINE_LU_SOLVE(solve_lu_cfloat, float2)
DEFINE_GENERAL(solve_general_float, float)
DEFINE_GENERAL(solve_general_cfloat, float2)
DEFINE_GRAM(solve_gram_float, float)
DEFINE_GRAM(solve_gram_cfloat, float2)
DEFINE_EXPAND(solve_expand_float, float)
DEFINE_EXPAND(solve_expand_cfloat, float2)
