// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct GradientParams {
    ulong dims[4];
    ulong inputStrides[4];
    ulong gradient0Strides[4];
    ulong gradient1Strides[4];
};

#define DEFINE_GRADIENT_KERNEL(NAME, TYPE)                                  \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* gradient0 [[buffer(1)]],                      \
                 device TYPE* gradient1 [[buffer(2)]],                      \
                 constant GradientParams& params [[buffer(3)]],             \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total = params.dims[0] * params.dims[1] *                   \
                        params.dims[2] * params.dims[3];                     \
    if (gid >= total) return;                                                \
    ulong position = gid;                                                    \
    const ulong x = position % params.dims[0];                               \
    position /= params.dims[0];                                              \
    const ulong y = position % params.dims[1];                               \
    position /= params.dims[1];                                              \
    const ulong z = position % params.dims[2];                               \
    const ulong w = position / params.dims[2];                               \
    const ulong inputOffset = x + y * params.inputStrides[1] +               \
        z * params.inputStrides[2] + w * params.inputStrides[3];             \
    const ulong output0Offset = x + y * params.gradient0Strides[1] +         \
        z * params.gradient0Strides[2] + w * params.gradient0Strides[3];     \
    const ulong output1Offset = x + y * params.gradient1Strides[1] +         \
        z * params.gradient1Strides[2] + w * params.gradient1Strides[3];     \
    if (params.dims[0] == 1) {                                               \
        gradient0[output0Offset] = TYPE(0);                                  \
    } else {                                                                 \
        const ulong left = x == 0 ? inputOffset : inputOffset - 1;           \
        const ulong right = x == params.dims[0] - 1                          \
            ? inputOffset : inputOffset + 1;                                \
        const float scale = (x == 0 || x == params.dims[0] - 1)             \
            ? 1.0f : 0.5f;                                                  \
        gradient0[output0Offset] = scale * (input[right] - input[left]);     \
    }                                                                        \
    if (params.dims[1] == 1) {                                               \
        gradient1[output1Offset] = TYPE(0);                                  \
    } else {                                                                 \
        const ulong down = y == 0 ? inputOffset                             \
            : inputOffset - params.inputStrides[1];                         \
        const ulong up = y == params.dims[1] - 1 ? inputOffset              \
            : inputOffset + params.inputStrides[1];                         \
        const float scale = (y == 0 || y == params.dims[1] - 1)             \
            ? 1.0f : 0.5f;                                                  \
        gradient1[output1Offset] = scale * (input[up] - input[down]);        \
    }                                                                        \
}

DEFINE_GRADIENT_KERNEL(gradient_float, float)
DEFINE_GRADIENT_KERNEL(gradient_cfloat, float2)

