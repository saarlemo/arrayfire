/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/

#include <metal_stdlib>
using namespace metal;

struct SvdInitParams {
    uint inputRows;
    uint inputColumns;
    uint workRows;
    uint workColumns;
    uint transpose;
};

struct SvdStageParams {
    uint rows;
    uint columns;
    uint stage;
    uint threads;
};

struct SvdFinalizeParams {
    uint inputRows;
    uint inputColumns;
    uint workRows;
    uint workColumns;
    uint transpose;
};

float svdNormSquared(const float value) { return value * value; }
float svdNormSquared(const float2 value) { return dot(value, value); }
float2 svdConjugate(const float2 value) { return float2(value.x, -value.y); }
float2 svdMultiply(const float2 lhs, const float2 rhs) {
    return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                  lhs.x * rhs.y + lhs.y * rhs.x);
}

kernel void svd_init_float(const device float* input [[buffer(0)]],
                            device float* work [[buffer(1)]],
                            device float* vectors [[buffer(2)]],
                            constant SvdInitParams& p [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
    const uint workElements = p.workRows * p.workColumns;
    if (gid < workElements) {
        const uint row = gid % p.workRows;
        const uint column = gid / p.workRows;
        const uint inputIndex = p.transpose
                                    ? column + row * p.inputRows
                                    : row + column * p.inputRows;
        work[gid] = input[inputIndex];
    }
    const uint vectorElements = p.workColumns * p.workColumns;
    if (gid < vectorElements) {
        const uint row = gid % p.workColumns;
        const uint column = gid / p.workColumns;
        vectors[gid] = row == column ? 1.0f : 0.0f;
    }
}

kernel void svd_stage_float(device float* work [[buffer(0)]],
                             device float* vectors [[buffer(1)]],
                             constant SvdStageParams& p [[buffer(2)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    const uint pairCount = p.columns / 2;
    if (group >= pairCount) return;

    const uint fixed = p.columns - 1;
    uint first, second;
    if (group == 0) {
        first = fixed;
        second = p.stage % (p.columns - 1);
    } else {
        first = (p.stage + group) % (p.columns - 1);
        second = (p.stage + p.columns - 1 - group) % (p.columns - 1);
    }

    threadgroup float alpha[256];
    threadgroup float beta[256];
    threadgroup float gamma[256];
    threadgroup float cosine;
    threadgroup float sine;

    float localAlpha = 0.0f;
    float localBeta = 0.0f;
    float localGamma = 0.0f;
    for (uint row = tid; row < p.rows; row += p.threads) {
        const float lhs = work[row + first * p.rows];
        const float rhs = work[row + second * p.rows];
        localAlpha += lhs * lhs;
        localBeta += rhs * rhs;
        localGamma += lhs * rhs;
    }
    alpha[tid] = localAlpha;
    beta[tid] = localBeta;
    gamma[tid] = localGamma;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float a = 0.0f, b = 0.0f, g = 0.0f;
        for (uint i = 0; i < p.threads; ++i) {
            a += alpha[i];
            b += beta[i];
            g += gamma[i];
        }
        if (g == 0.0f) {
            cosine = 1.0f;
            sine = 0.0f;
        } else {
            const float tau = (b - a) / (2.0f * g);
            const float t = tau >= 0.0f
                                ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                                : -1.0f / (-tau + sqrt(1.0f + tau * tau));
            cosine = 1.0f / sqrt(1.0f + t * t);
            sine = t * cosine;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = tid; row < p.rows; row += p.threads) {
        const uint firstIndex = row + first * p.rows;
        const uint secondIndex = row + second * p.rows;
        const float lhs = work[firstIndex];
        const float rhs = work[secondIndex];
        work[firstIndex] = cosine * lhs + sine * rhs;
        work[secondIndex] = -sine * lhs + cosine * rhs;
    }
    for (uint row = tid; row < p.columns; row += p.threads) {
        const uint firstIndex = row + first * p.columns;
        const uint secondIndex = row + second * p.columns;
        const float lhs = vectors[firstIndex];
        const float rhs = vectors[secondIndex];
        vectors[firstIndex] = cosine * lhs + sine * rhs;
        vectors[secondIndex] = -sine * lhs + cosine * rhs;
    }
}

kernel void svd_sort_float(device float* work [[buffer(0)]],
                            device float* vectors [[buffer(1)]],
                            constant SvdStageParams& p [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    for (uint target = 0; target < p.columns; ++target) {
        uint best = target;
        float bestNorm = -1.0f;
        for (uint column = target; column < p.columns; ++column) {
            float norm = 0.0f;
            for (uint row = 0; row < p.rows; ++row) {
                const float value = work[row + column * p.rows];
                norm += value * value;
            }
            if (norm > bestNorm) {
                bestNorm = norm;
                best = column;
            }
        }
        if (best == target) continue;
        for (uint row = 0; row < p.rows; ++row) {
            const uint lhs = row + target * p.rows;
            const uint rhs = row + best * p.rows;
            const float tmp = work[lhs];
            work[lhs] = work[rhs];
            work[rhs] = tmp;
        }
        for (uint row = 0; row < p.columns; ++row) {
            const uint lhs = row + target * p.columns;
            const uint rhs = row + best * p.columns;
            const float tmp = vectors[lhs];
            vectors[lhs] = vectors[rhs];
            vectors[rhs] = tmp;
        }
    }
}

kernel void svd_finalize_float(const device float* work [[buffer(0)]],
                                const device float* vectors [[buffer(1)]],
                                device float* singularValues [[buffer(2)]],
                                device float* u [[buffer(3)]],
                                device float* vt [[buffer(4)]],
                                constant SvdFinalizeParams& p [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid < p.workColumns) {
        float norm = 0.0f;
        for (uint row = 0; row < p.workRows; ++row) {
            const float value = work[row + gid * p.workRows];
            norm += value * value;
        }
        singularValues[gid] = sqrt(norm);
    }

    const uint uElements = p.inputRows * p.inputRows;
    if (gid < uElements) {
        const uint row = gid % p.inputRows;
        const uint column = gid / p.inputRows;
        if (!p.transpose) {
            if (column < p.workColumns) {
                float norm = 0.0f;
                for (uint i = 0; i < p.workRows; ++i) {
                    const float value = work[i + column * p.workRows];
                    norm += value * value;
                }
                u[gid] = norm > 0.0f ? work[row + column * p.workRows] /
                                          sqrt(norm)
                                      : 0.0f;
            } else {
                u[gid] = row == column ? 1.0f : 0.0f;
            }
        } else {
            u[gid] = vectors[row + column * p.workColumns];
        }
    }

    const uint vtElements = p.inputColumns * p.inputColumns;
    if (gid < vtElements) {
        const uint row = gid % p.inputColumns;
        const uint column = gid / p.inputColumns;
        if (!p.transpose) {
            vt[gid] = vectors[column + row * p.workColumns];
        } else if (row < p.workColumns) {
            float norm = 0.0f;
            for (uint i = 0; i < p.workRows; ++i) {
                const float value = work[i + row * p.workRows];
                norm += value * value;
            }
            vt[gid] = norm > 0.0f ? work[column + row * p.workRows] /
                                       sqrt(norm)
                                   : 0.0f;
        } else {
            vt[gid] = row == column ? 1.0f : 0.0f;
        }
    }
}

kernel void svd_init_cfloat(const device float2* input [[buffer(0)]],
                             device float2* work [[buffer(1)]],
                             device float2* vectors [[buffer(2)]],
                             constant SvdInitParams& p [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    const uint workElements = p.workRows * p.workColumns;
    if (gid < workElements) {
        const uint row = gid % p.workRows;
        const uint column = gid / p.workRows;
        const uint inputIndex = p.transpose
                                    ? column + row * p.inputRows
                                    : row + column * p.inputRows;
        work[gid] = p.transpose ? svdConjugate(input[inputIndex])
                                : input[inputIndex];
    }
    const uint vectorElements = p.workColumns * p.workColumns;
    if (gid < vectorElements) {
        const uint row = gid % p.workColumns;
        const uint column = gid / p.workColumns;
        vectors[gid] = row == column ? float2(1.0f, 0.0f) : float2(0.0f);
    }
}

kernel void svd_stage_cfloat(device float2* work [[buffer(0)]],
                              device float2* vectors [[buffer(1)]],
                              constant SvdStageParams& p [[buffer(2)]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint group [[threadgroup_position_in_grid]]) {
    const uint pairCount = p.columns / 2;
    if (group >= pairCount) return;

    const uint fixed = p.columns - 1;
    uint first, second;
    if (group == 0) {
        first = fixed;
        second = p.stage % (p.columns - 1);
    } else {
        first = (p.stage + group) % (p.columns - 1);
        second = (p.stage + p.columns - 1 - group) % (p.columns - 1);
    }

    threadgroup float alpha[256];
    threadgroup float beta[256];
    threadgroup float2 gamma[256];
    threadgroup float cosine;
    threadgroup float sine;
    threadgroup float2 phase;

    float localAlpha = 0.0f;
    float localBeta = 0.0f;
    float2 localGamma = float2(0.0f);
    for (uint row = tid; row < p.rows; row += p.threads) {
        const float2 lhs = work[row + first * p.rows];
        const float2 rhs = work[row + second * p.rows];
        localAlpha += dot(lhs, lhs);
        localBeta += dot(rhs, rhs);
        localGamma += svdMultiply(svdConjugate(lhs), rhs);
    }
    alpha[tid] = localAlpha;
    beta[tid] = localBeta;
    gamma[tid] = localGamma;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float a = 0.0f, b = 0.0f;
        float2 g = float2(0.0f);
        for (uint i = 0; i < p.threads; ++i) {
            a += alpha[i];
            b += beta[i];
            g += gamma[i];
        }
        const float magnitude = length(g);
        if (magnitude == 0.0f) {
            cosine = 1.0f;
            sine = 0.0f;
            phase = float2(1.0f, 0.0f);
        } else {
            phase = g / magnitude;
            const float tau = (b - a) / (2.0f * magnitude);
            const float t = tau >= 0.0f
                                ? 1.0f / (tau + sqrt(1.0f + tau * tau))
                                : -1.0f / (-tau + sqrt(1.0f + tau * tau));
            cosine = 1.0f / sqrt(1.0f + t * t);
            sine = t * cosine;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float2 conjugatePhase = svdConjugate(phase);
    for (uint row = tid; row < p.rows; row += p.threads) {
        const uint firstIndex = row + first * p.rows;
        const uint secondIndex = row + second * p.rows;
        const float2 lhs = work[firstIndex];
        const float2 rhs = work[secondIndex];
        work[firstIndex] = cosine * lhs + sine * svdMultiply(conjugatePhase, rhs);
        work[secondIndex] = -sine * svdMultiply(phase, lhs) + cosine * rhs;
    }
    for (uint row = tid; row < p.columns; row += p.threads) {
        const uint firstIndex = row + first * p.columns;
        const uint secondIndex = row + second * p.columns;
        const float2 lhs = vectors[firstIndex];
        const float2 rhs = vectors[secondIndex];
        vectors[firstIndex] = cosine * lhs + sine * svdMultiply(conjugatePhase, rhs);
        vectors[secondIndex] = -sine * svdMultiply(phase, lhs) + cosine * rhs;
    }
}

kernel void svd_sort_cfloat(device float2* work [[buffer(0)]],
                             device float2* vectors [[buffer(1)]],
                             constant SvdStageParams& p [[buffer(2)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    for (uint target = 0; target < p.columns; ++target) {
        uint best = target;
        float bestNorm = -1.0f;
        for (uint column = target; column < p.columns; ++column) {
            float norm = 0.0f;
            for (uint row = 0; row < p.rows; ++row)
                norm += svdNormSquared(work[row + column * p.rows]);
            if (norm > bestNorm) {
                bestNorm = norm;
                best = column;
            }
        }
        if (best == target) continue;
        for (uint row = 0; row < p.rows; ++row) {
            const uint lhs = row + target * p.rows;
            const uint rhs = row + best * p.rows;
            const float2 tmp = work[lhs];
            work[lhs] = work[rhs];
            work[rhs] = tmp;
        }
        for (uint row = 0; row < p.columns; ++row) {
            const uint lhs = row + target * p.columns;
            const uint rhs = row + best * p.columns;
            const float2 tmp = vectors[lhs];
            vectors[lhs] = vectors[rhs];
            vectors[rhs] = tmp;
        }
    }
}

kernel void svd_finalize_cfloat(const device float2* work [[buffer(0)]],
                                 const device float2* vectors [[buffer(1)]],
                                 device float* singularValues [[buffer(2)]],
                                 device float2* u [[buffer(3)]],
                                 device float2* vt [[buffer(4)]],
                                 constant SvdFinalizeParams& p [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid < p.workColumns) {
        float norm = 0.0f;
        for (uint row = 0; row < p.workRows; ++row)
            norm += svdNormSquared(work[row + gid * p.workRows]);
        singularValues[gid] = sqrt(norm);
    }

    const uint uElements = p.inputRows * p.inputRows;
    if (gid < uElements) {
        const uint row = gid % p.inputRows;
        const uint column = gid / p.inputRows;
        if (!p.transpose) {
            if (column < p.workColumns) {
                float norm = 0.0f;
                for (uint i = 0; i < p.workRows; ++i)
                    norm += svdNormSquared(work[i + column * p.workRows]);
                u[gid] = norm > 0.0f
                              ? work[row + column * p.workRows] / sqrt(norm)
                              : float2(0.0f);
            } else {
                u[gid] = row == column ? float2(1.0f, 0.0f)
                                       : float2(0.0f);
            }
        } else {
            u[gid] = vectors[row + column * p.workColumns];
        }
    }

    const uint vtElements = p.inputColumns * p.inputColumns;
    if (gid < vtElements) {
        const uint row = gid % p.inputColumns;
        const uint column = gid / p.inputColumns;
        if (!p.transpose) {
            vt[gid] = svdConjugate(vectors[column + row * p.workColumns]);
        } else if (row < p.workColumns) {
            float norm = 0.0f;
            for (uint i = 0; i < p.workRows; ++i)
                norm += svdNormSquared(work[i + row * p.workRows]);
            vt[gid] = norm > 0.0f
                          ? work[column + row * p.workRows] / sqrt(norm)
                          : float2(0.0f);
            vt[gid] = svdConjugate(vt[gid]);
        } else {
            vt[gid] = row == column ? float2(1.0f, 0.0f)
                                   : float2(0.0f);
        }
    }
}
