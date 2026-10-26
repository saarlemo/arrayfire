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

struct MeanParams {
    ulong outputDims[4];
    long outputStrides[4];
    ulong inputDims[4];
    long inputStrides[4];
    long weightStrides[4];
    uint dimension;
    uint reduceAll;
};

void meanCompensatedAdd(thread float& sum, thread float& correction,
                        const float value) {
    if (!isfinite(sum) || !isfinite(value)) {
        sum += value;
        correction = 0.0f;
        return;
    }
    const float updated = sum + value;
    if (!isfinite(updated)) {
        sum = updated;
        correction = 0.0f;
        return;
    }
    correction += abs(sum) >= abs(value) ? (sum - updated) + value
                                        : (value - updated) + sum;
    sum = updated;
}

void meanCompensatedAdd(thread float2& sum, thread float2& correction,
                        const float2 value) {
    float realSum = sum.x, realCorrection = correction.x;
    float imagSum = sum.y, imagCorrection = correction.y;
    meanCompensatedAdd(realSum, realCorrection, value.x);
    meanCompensatedAdd(imagSum, imagCorrection, value.y);
    sum = float2(realSum, imagSum);
    correction = float2(realCorrection, imagCorrection);
}
void meanCompensatedAdd(threadgroup float& sum,
                        threadgroup float& correction, const float value) {
    if (!isfinite(sum) || !isfinite(value)) {
        sum += value;
        correction = 0.0f;
        return;
    }
    const float updated = sum + value;
    if (!isfinite(updated)) {
        sum = updated;
        correction = 0.0f;
        return;
    }
    correction += abs(sum) >= abs(value) ? (sum - updated) + value
                                        : (value - updated) + sum;
    sum = updated;
}
void meanCompensatedAdd(threadgroup float2& sum,
                        threadgroup float2& correction,
                        const float2 value) {
    float realSum = sum.x, realCorrection = correction.x;
    float imagSum = sum.y, imagCorrection = correction.y;
    meanCompensatedAdd(realSum, realCorrection, value.x);
    meanCompensatedAdd(imagSum, imagCorrection, value.y);
    sum = float2(realSum, imagSum);
    correction = float2(realCorrection, imagCorrection);
}

#define DEFINE_MEAN_PARALLEL(NAME, INPUT_TYPE, OUTPUT_TYPE, COMPUTE_TYPE)   \
kernel void NAME##_partial(                                                  \
    const device INPUT_TYPE* input [[buffer(0)]],                            \
    device COMPUTE_TYPE* partials [[buffer(1)]],                             \
    device COMPUTE_TYPE* partialCorrections [[buffer(2)]],                   \
    constant MeanParams& p [[buffer(3)]],                                    \
    uint3 group [[threadgroup_position_in_grid]],                            \
    uint lane [[thread_index_in_threadgroup]]) {                             \
    threadgroup COMPUTE_TYPE values[256];                                    \
    threadgroup COMPUTE_TYPE corrections[256];                               \
    const ulong outputTotal = p.outputDims[0] * p.outputDims[1] *            \
                              p.outputDims[2] * p.outputDims[3];              \
    const ulong inputTotal = p.inputDims[0] * p.inputDims[1] *               \
                             p.inputDims[2] * p.inputDims[3];                 \
    const ulong length = p.reduceAll ? inputTotal : p.inputDims[p.dimension];\
    const ulong chunks = max((length + 4095) / 4096, 1UL);                   \
    const ulong outputIndex = ulong(group.x) / chunks;                       \
    const ulong chunk = ulong(group.x) % chunks;                             \
    if (outputIndex >= outputTotal) return;                                  \
    COMPUTE_TYPE runningSum = COMPUTE_TYPE(0);                               \
    COMPUTE_TYPE sumCorrection = COMPUTE_TYPE(0);                            \
    const ulong begin = chunk * 4096 + lane;                                 \
    const ulong end = min((chunk + 1) * 4096, length);                       \
    for (ulong i = begin; i < end; i += 256) {                               \
        long inputIndex = 0;                                                 \
        if (p.reduceAll) {                                                   \
            ulong q = i;                                                     \
            for (uint d = 0; d < 4; ++d) {                                  \
                const ulong coordinate = q % p.inputDims[d];                 \
                q /= p.inputDims[d];                                         \
                inputIndex += long(coordinate) * p.inputStrides[d];          \
            }                                                                \
        } else {                                                             \
            ulong q = outputIndex;                                           \
            for (uint d = 0; d < 4; ++d) {                                  \
                const ulong coordinate = q % p.outputDims[d];                \
                q /= p.outputDims[d];                                         \
                inputIndex += long(coordinate) * p.inputStrides[d];          \
            }                                                                \
            inputIndex += long(i) * p.inputStrides[p.dimension];             \
        }                                                                    \
        meanCompensatedAdd(runningSum, sumCorrection,                        \
                           COMPUTE_TYPE(input[inputIndex]));                 \
    }                                                                        \
    values[lane] = runningSum;                                               \
    corrections[lane] = sumCorrection;                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    for (uint stride = 128; stride > 0; stride >>= 1) {                     \
        if (lane < stride) {                                                 \
            meanCompensatedAdd(values[lane], corrections[lane],              \
                               values[lane + stride]);                       \
            meanCompensatedAdd(values[lane], corrections[lane],              \
                               corrections[lane + stride]);                 \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    }                                                                        \
    if (lane == 0) {                                                         \
        partials[group.x] = values[0];                                       \
        partialCorrections[group.x] = corrections[0];                        \
    }                                                                        \
}                                                                            \
kernel void NAME##_final(                                                   \
    const device COMPUTE_TYPE* partials [[buffer(0)]],                      \
    const device COMPUTE_TYPE* partialCorrections [[buffer(1)]],            \
    device OUTPUT_TYPE* output [[buffer(2)]],                                \
    constant MeanParams& p [[buffer(3)]],                                    \
    uint outputLinear [[threadgroup_position_in_grid]],                      \
    uint lane [[thread_index_in_threadgroup]]) {                             \
    threadgroup COMPUTE_TYPE values[256];                                    \
    threadgroup COMPUTE_TYPE corrections[256];                               \
    const ulong outputTotal = p.outputDims[0] * p.outputDims[1] *            \
                              p.outputDims[2] * p.outputDims[3];              \
    const ulong inputTotal = p.inputDims[0] * p.inputDims[1] *               \
                             p.inputDims[2] * p.inputDims[3];                 \
    const ulong length = p.reduceAll ? inputTotal : p.inputDims[p.dimension];\
    const ulong chunks = max((length + 4095) / 4096, 1UL);                   \
    if (ulong(outputLinear) >= outputTotal) return;                         \
    COMPUTE_TYPE value = COMPUTE_TYPE(0);                                    \
    COMPUTE_TYPE correction = COMPUTE_TYPE(0);                               \
    for (ulong i = lane; i < chunks; i += 256) {                             \
        const ulong partialIndex = ulong(outputLinear) * chunks + i;         \
        meanCompensatedAdd(value, correction, partials[partialIndex]);        \
        meanCompensatedAdd(value, correction,                                \
                           partialCorrections[partialIndex]);                \
    }                                                                        \
    values[lane] = value;                                                    \
    corrections[lane] = correction;                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    for (uint stride = 128; stride > 0; stride >>= 1) {                     \
        if (lane < stride) {                                                 \
            meanCompensatedAdd(values[lane], corrections[lane],              \
                               values[lane + stride]);                       \
            meanCompensatedAdd(values[lane], corrections[lane],              \
                               corrections[lane + stride]);                 \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    }                                                                        \
    if (lane == 0) {                                                         \
        ulong q = outputLinear;                                              \
        long outputOffset = 0;                                               \
        if (!p.reduceAll) {                                                  \
            for (uint d = 0; d < 4; ++d) {                                  \
                const ulong coordinate = q % p.outputDims[d];                \
                q /= p.outputDims[d];                                         \
                outputOffset += long(coordinate) * p.outputStrides[d];       \
            }                                                                \
        }                                                                    \
        output[outputOffset] = length != 0                                   \
                                   ? OUTPUT_TYPE((values[0] + corrections[0]) / float(length)) \
                                   : OUTPUT_TYPE(0);                          \
    }                                                                        \
}

#define DEFINE_MEAN(NAME, INPUT_TYPE, OUTPUT_TYPE, COMPUTE_TYPE)             \
    DEFINE_MEAN_PARALLEL(NAME, INPUT_TYPE, OUTPUT_TYPE, COMPUTE_TYPE)

#define DEFINE_WEIGHTED_MEAN_PARALLEL(NAME, VALUE_TYPE, OUTPUT_TYPE,        \
                                      COMPUTE_TYPE)                          \
kernel void NAME##_partial(                                                  \
    const device VALUE_TYPE* input [[buffer(0)]],                            \
    const device float* weights [[buffer(1)]],                               \
    device COMPUTE_TYPE* partialValues [[buffer(2)]],                       \
    device COMPUTE_TYPE* partialValueCorrections [[buffer(3)]],             \
    device float* partialWeights [[buffer(4)]],                              \
    device float* partialWeightCorrections [[buffer(5)]],                    \
    constant MeanParams& p [[buffer(6)]],                                    \
    uint3 group [[threadgroup_position_in_grid]],                            \
    uint lane [[thread_index_in_threadgroup]]) {                             \
    threadgroup COMPUTE_TYPE sums[256];                                      \
    threadgroup COMPUTE_TYPE sumCorrections[256];                            \
    threadgroup float weightSums[256];                                       \
    threadgroup float weightCorrections[256];                                \
    const ulong outputTotal = p.outputDims[0] * p.outputDims[1] *            \
                              p.outputDims[2] * p.outputDims[3];              \
    const ulong inputTotal = p.inputDims[0] * p.inputDims[1] *               \
                             p.inputDims[2] * p.inputDims[3];                 \
    const ulong length = p.reduceAll ? inputTotal : p.inputDims[p.dimension];\
    const ulong chunks = max((length + 4095) / 4096, 1UL);                   \
    const ulong outputIndex = ulong(group.x) / chunks;                       \
    const ulong chunk = ulong(group.x) % chunks;                             \
    if (outputIndex >= outputTotal) return;                                  \
    COMPUTE_TYPE runningSum = COMPUTE_TYPE(0);                               \
    COMPUTE_TYPE sumCorrection = COMPUTE_TYPE(0);                            \
    float runningWeight = 0.0f;                                              \
    float weightCorrection = 0.0f;                                           \
    const ulong begin = chunk * 4096 + lane;                                 \
    const ulong end = min((chunk + 1) * 4096, length);                       \
    for (ulong i = begin; i < end; i += 256) {                               \
        long inputIndex = 0;                                                 \
        long weightIndex = 0;                                                \
        if (p.reduceAll) {                                                   \
            ulong q = i;                                                     \
            for (uint d = 0; d < 4; ++d) {                                  \
                const ulong coordinate = q % p.inputDims[d];                 \
                q /= p.inputDims[d];                                         \
                inputIndex += long(coordinate) * p.inputStrides[d];          \
                weightIndex += long(coordinate) * p.weightStrides[d];        \
            }                                                                \
        } else {                                                             \
            ulong q = outputIndex;                                           \
            for (uint d = 0; d < 4; ++d) {                                  \
                const ulong coordinate = q % p.outputDims[d];                \
                q /= p.outputDims[d];                                         \
                inputIndex += long(coordinate) * p.inputStrides[d];          \
                weightIndex += long(coordinate) * p.weightStrides[d];        \
            }                                                                \
            inputIndex += long(i) * p.inputStrides[p.dimension];             \
            weightIndex += long(i) * p.weightStrides[p.dimension];           \
        }                                                                    \
        const float weight = weights[weightIndex];                           \
        if (weight != 0.0f) {                                                 \
            meanCompensatedAdd(                                               \
                runningSum, sumCorrection,                                   \
                COMPUTE_TYPE(input[inputIndex]) * weight);                   \
            meanCompensatedAdd(runningWeight, weightCorrection, weight);     \
        }                                                                    \
    }                                                                        \
    sums[lane] = runningSum;                                                 \
    sumCorrections[lane] = sumCorrection;                                    \
    weightSums[lane] = runningWeight;                                        \
    weightCorrections[lane] = weightCorrection;                              \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    for (uint stride = 128; stride > 0; stride >>= 1) {                     \
        if (lane < stride) {                                                 \
            meanCompensatedAdd(sums[lane], sumCorrections[lane],             \
                               sums[lane + stride]);                         \
            meanCompensatedAdd(sums[lane], sumCorrections[lane],             \
                               sumCorrections[lane + stride]);               \
            meanCompensatedAdd(weightSums[lane],                             \
                               weightCorrections[lane],                      \
                               weightSums[lane + stride]);                   \
            meanCompensatedAdd(weightSums[lane],                             \
                               weightCorrections[lane],                      \
                               weightCorrections[lane + stride]);             \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    }                                                                        \
    if (lane == 0) {                                                         \
        partialValues[group.x] = sums[0];                                    \
        partialValueCorrections[group.x] = sumCorrections[0];                \
        partialWeights[group.x] = weightSums[0];                             \
        partialWeightCorrections[group.x] = weightCorrections[0];            \
    }                                                                        \
}                                                                            \
kernel void NAME##_final(                                                   \
    const device COMPUTE_TYPE* partialValues [[buffer(0)]],                 \
    const device COMPUTE_TYPE* partialValueCorrections [[buffer(1)]],       \
    const device float* partialWeights [[buffer(2)]],                       \
    const device float* partialWeightCorrections [[buffer(3)]],              \
    device OUTPUT_TYPE* output [[buffer(4)]],                                \
    constant MeanParams& p [[buffer(5)]],                                    \
    uint outputLinear [[threadgroup_position_in_grid]],                      \
    uint lane [[thread_index_in_threadgroup]]) {                             \
    threadgroup COMPUTE_TYPE sums[256];                                      \
    threadgroup COMPUTE_TYPE sumCorrections[256];                            \
    threadgroup float weightSums[256];                                       \
    threadgroup float weightCorrections[256];                                \
    const ulong outputTotal = p.outputDims[0] * p.outputDims[1] *            \
                              p.outputDims[2] * p.outputDims[3];              \
    const ulong inputTotal = p.inputDims[0] * p.inputDims[1] *               \
                             p.inputDims[2] * p.inputDims[3];                 \
    const ulong length = p.reduceAll ? inputTotal : p.inputDims[p.dimension];\
    const ulong chunks = max((length + 4095) / 4096, 1UL);                   \
    if (ulong(outputLinear) >= outputTotal) return;                         \
    COMPUTE_TYPE value = COMPUTE_TYPE(0);                                    \
    COMPUTE_TYPE valueCorrection = COMPUTE_TYPE(0);                          \
    float weight = 0.0f;                                                     \
    float weightCorrection = 0.0f;                                           \
    for (ulong i = lane; i < chunks; i += 256) {                             \
        const ulong partialIndex = ulong(outputLinear) * chunks + i;         \
        meanCompensatedAdd(value, valueCorrection,                           \
                           partialValues[partialIndex]);                    \
        meanCompensatedAdd(value, valueCorrection,                           \
                           partialValueCorrections[partialIndex]);           \
        meanCompensatedAdd(weight, weightCorrection,                         \
                           partialWeights[partialIndex]);                   \
        meanCompensatedAdd(weight, weightCorrection,                         \
                           partialWeightCorrections[partialIndex]);          \
    }                                                                        \
    sums[lane] = value;                                                      \
    sumCorrections[lane] = valueCorrection;                                  \
    weightSums[lane] = weight;                                               \
    weightCorrections[lane] = weightCorrection;                              \
    threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    for (uint stride = 128; stride > 0; stride >>= 1) {                     \
        if (lane < stride) {                                                 \
            meanCompensatedAdd(sums[lane], sumCorrections[lane],             \
                               sums[lane + stride]);                         \
            meanCompensatedAdd(sums[lane], sumCorrections[lane],             \
                               sumCorrections[lane + stride]);               \
            meanCompensatedAdd(weightSums[lane],                             \
                               weightCorrections[lane],                      \
                               weightSums[lane + stride]);                   \
            meanCompensatedAdd(weightSums[lane],                             \
                               weightCorrections[lane],                      \
                               weightCorrections[lane + stride]);             \
        }                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                     \
    }                                                                        \
    if (lane == 0) {                                                         \
        ulong q = outputLinear;                                              \
        long outputOffset = 0;                                               \
        if (!p.reduceAll) {                                                  \
            for (uint d = 0; d < 4; ++d) {                                  \
                const ulong coordinate = q % p.outputDims[d];                \
                q /= p.outputDims[d];                                         \
                outputOffset += long(coordinate) * p.outputStrides[d];       \
            }                                                                \
        }                                                                    \
        const float totalWeight = weightSums[0] + weightCorrections[0];      \
        const COMPUTE_TYPE totalValue = sums[0] + sumCorrections[0];         \
        output[outputOffset] = totalWeight != 0.0f                           \
                                   ? OUTPUT_TYPE(totalValue / totalWeight)   \
                                   : OUTPUT_TYPE(0);                          \
    }                                                                        \
}

#define DEFINE_WEIGHTED_MEAN(NAME, VALUE_TYPE, OUTPUT_TYPE, COMPUTE_TYPE)    \
    DEFINE_WEIGHTED_MEAN_PARALLEL(NAME, VALUE_TYPE, OUTPUT_TYPE, COMPUTE_TYPE)

DEFINE_MEAN(mean_float_float, float, float, float)
DEFINE_MEAN(mean_cfloat_cfloat, float2, float2, float2)
DEFINE_MEAN(mean_int_float, int, float, float)
DEFINE_MEAN(mean_uint_float, uint, float, float)
DEFINE_MEAN(mean_char_float, char, float, float)
DEFINE_MEAN(mean_uchar_float, uchar, float, float)
DEFINE_MEAN(mean_short_float, short, float, float)
DEFINE_MEAN(mean_ushort_float, ushort, float, float)
DEFINE_MEAN(mean_half_half, half, half, float)
DEFINE_MEAN(mean_half_float, half, float, float)

DEFINE_WEIGHTED_MEAN(mean_weighted_float, float, float, float)
DEFINE_WEIGHTED_MEAN(mean_weighted_cfloat, float2, float2, float2)
DEFINE_WEIGHTED_MEAN(mean_weighted_half, half, half, float)
