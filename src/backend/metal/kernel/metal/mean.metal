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

template<typename T>
void meanAccumulate(thread T& runningSum, thread T& sumCorrection,
                    thread float& runningWeight,
                    thread float& weightCorrection, const T value,
                    const float weight) {
    if (weight == 0.0f) return;

    const T weightedValue = value * weight;
    const T correctedValue = weightedValue - sumCorrection;
    const T updatedSum = runningSum + correctedValue;
    sumCorrection = (updatedSum - runningSum) - correctedValue;
    runningSum = updatedSum;

    const float correctedWeight = weight - weightCorrection;
    const float updatedWeight = runningWeight + correctedWeight;
    weightCorrection = (updatedWeight - runningWeight) - correctedWeight;
    runningWeight = updatedWeight;
}

#define DEFINE_MEAN(NAME, INPUT_TYPE, OUTPUT_TYPE, COMPUTE_TYPE)             \
kernel void NAME(const device INPUT_TYPE* input [[buffer(0)]],               \
                 device OUTPUT_TYPE* output [[buffer(1)]],                   \
                 constant MeanParams& params [[buffer(2)]],                  \
                 uint gid [[thread_position_in_grid]]) {                     \
    const ulong outputTotal = params.outputDims[0] * params.outputDims[1] *  \
                              params.outputDims[2] * params.outputDims[3];    \
    if (gid >= outputTotal) return;                                           \
    COMPUTE_TYPE runningSum = COMPUTE_TYPE(0);                               \
    COMPUTE_TYPE sumCorrection = COMPUTE_TYPE(0);                            \
    float runningWeight = 0.0f;                                              \
    float weightCorrection = 0.0f;                                           \
    if (params.reduceAll) {                                                   \
        const ulong inputTotal = params.inputDims[0] * params.inputDims[1] * \
                                 params.inputDims[2] * params.inputDims[3];   \
        for (ulong linear = 0; linear < inputTotal; ++linear) {              \
            ulong q = linear;                                                \
            long inputIndex = 0;                                             \
            for (uint d = 0; d < 4; ++d) {                                  \
                const ulong coordinate = q % params.inputDims[d];            \
                q /= params.inputDims[d];                                    \
                inputIndex += long(coordinate) * params.inputStrides[d];     \
            }                                                                \
            meanAccumulate(runningSum, sumCorrection, runningWeight,          \
                           weightCorrection,                                  \
                           COMPUTE_TYPE(input[inputIndex]), 1.0f);            \
        }                                                                    \
        output[0] = runningWeight != 0.0f                                     \
                        ? OUTPUT_TYPE(runningSum / runningWeight)             \
                        : OUTPUT_TYPE(0);                                     \
        return;                                                              \
    }                                                                        \
    ulong q = gid;                                                           \
    long outputIndex = 0;                                                    \
    long inputBase = 0;                                                      \
    for (uint d = 0; d < 4; ++d) {                                          \
        const ulong coordinate = q % params.outputDims[d];                   \
        q /= params.outputDims[d];                                           \
        outputIndex += long(coordinate) * params.outputStrides[d];           \
        inputBase += long(coordinate) * params.inputStrides[d];              \
    }                                                                        \
    const ulong length = params.inputDims[params.dimension];                 \
    const long stride = params.inputStrides[params.dimension];               \
    for (ulong i = 0; i < length; ++i)                                      \
        meanAccumulate(runningSum, sumCorrection, runningWeight,              \
                       weightCorrection,                                     \
                       COMPUTE_TYPE(input[inputBase + long(i) * stride]),     \
                       1.0f);                                                  \
    output[outputIndex] = runningWeight != 0.0f                               \
                              ? OUTPUT_TYPE(runningSum / runningWeight)       \
                              : OUTPUT_TYPE(0);                               \
}

#define DEFINE_WEIGHTED_MEAN(NAME, VALUE_TYPE, OUTPUT_TYPE, COMPUTE_TYPE)    \
kernel void NAME(const device VALUE_TYPE* input [[buffer(0)]],               \
                 const device float* weights [[buffer(1)]],                  \
                 device OUTPUT_TYPE* output [[buffer(2)]],                   \
                 constant MeanParams& params [[buffer(3)]],                  \
                 uint gid [[thread_position_in_grid]]) {                     \
    const ulong outputTotal = params.outputDims[0] * params.outputDims[1] *  \
                              params.outputDims[2] * params.outputDims[3];    \
    if (gid >= outputTotal) return;                                           \
    COMPUTE_TYPE runningSum = COMPUTE_TYPE(0);                               \
    COMPUTE_TYPE sumCorrection = COMPUTE_TYPE(0);                            \
    float runningWeight = 0.0f;                                              \
    float weightCorrection = 0.0f;                                           \
    if (params.reduceAll) {                                                   \
        const ulong inputTotal = params.inputDims[0] * params.inputDims[1] * \
                                 params.inputDims[2] * params.inputDims[3];   \
        for (ulong linear = 0; linear < inputTotal; ++linear) {              \
            ulong q = linear;                                                \
            long inputIndex = 0;                                             \
            long weightIndex = 0;                                            \
            for (uint d = 0; d < 4; ++d) {                                  \
                const ulong coordinate = q % params.inputDims[d];            \
                q /= params.inputDims[d];                                    \
                inputIndex += long(coordinate) * params.inputStrides[d];     \
                weightIndex += long(coordinate) * params.weightStrides[d];   \
            }                                                                \
            meanAccumulate(runningSum, sumCorrection, runningWeight,          \
                           weightCorrection,                                  \
                           COMPUTE_TYPE(input[inputIndex]),                    \
                           weights[weightIndex]);                              \
        }                                                                    \
        output[0] = runningWeight != 0.0f                                     \
                        ? OUTPUT_TYPE(runningSum / runningWeight)             \
                        : OUTPUT_TYPE(0);                                     \
        return;                                                              \
    }                                                                        \
    ulong q = gid;                                                           \
    long outputIndex = 0;                                                    \
    long inputBase = 0;                                                      \
    long weightBase = 0;                                                     \
    for (uint d = 0; d < 4; ++d) {                                          \
        const ulong coordinate = q % params.outputDims[d];                   \
        q /= params.outputDims[d];                                           \
        outputIndex += long(coordinate) * params.outputStrides[d];           \
        inputBase += long(coordinate) * params.inputStrides[d];              \
        weightBase += long(coordinate) * params.weightStrides[d];            \
    }                                                                        \
    const ulong length = params.inputDims[params.dimension];                 \
    const long inputStride = params.inputStrides[params.dimension];          \
    const long weightStride = params.weightStrides[params.dimension];        \
    for (ulong i = 0; i < length; ++i) {                                    \
        const long inputIndex = inputBase + long(i) * inputStride;           \
        const long weightIndex = weightBase + long(i) * weightStride;        \
        meanAccumulate(runningSum, sumCorrection, runningWeight,              \
                       weightCorrection, COMPUTE_TYPE(input[inputIndex]),    \
                       weights[weightIndex]);                                 \
    }                                                                        \
    output[outputIndex] = runningWeight != 0.0f                               \
                              ? OUTPUT_TYPE(runningSum / runningWeight)       \
                              : OUTPUT_TYPE(0);                               \
}

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
