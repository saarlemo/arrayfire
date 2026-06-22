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

struct MomentsParams {
    ulong inputDims[4], outputStrides[4], inputStrides[4];
    uint moment;
};

#define DEFINE_MOMENTS(NAME, TYPE)                                         \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 device float* output [[buffer(1)]],                       \
                 constant MomentsParams& p [[buffer(2)]],                  \
                 uint gid [[thread_position_in_grid]]) {                   \
    const ulong batches = p.inputDims[2] * p.inputDims[3];                 \
    if (gid >= batches) return;                                            \
    const ulong z = gid % p.inputDims[2], w = gid / p.inputDims[2];        \
    const ulong io = z*p.inputStrides[2] + w*p.inputStrides[3];            \
    const ulong oo = z*p.outputStrides[2] + w*p.outputStrides[3];          \
    float m00=0.0f, m01=0.0f, m10=0.0f, m11=0.0f;                         \
    for (ulong y=0; y<p.inputDims[1]; ++y) {                               \
        for (ulong x=0; x<p.inputDims[0]; ++x) {                           \
            const float v=float(input[io+x*p.inputStrides[0]+              \
                                      y*p.inputStrides[1]]);                \
            m00+=v; m01+=float(x)*v; m10+=float(y)*v;                      \
            m11+=float(x)*float(y)*v;                                      \
        }                                                                  \
    }                                                                      \
    ulong m=0;                                                             \
    if (p.moment&1u) output[oo+m++]=m00;                                   \
    if (p.moment&2u) output[oo+m++]=m01;                                   \
    if (p.moment&4u) output[oo+m++]=m10;                                   \
    if (p.moment&8u) output[oo+m]=m11;                                     \
}

DEFINE_MOMENTS(moments_float, float)
DEFINE_MOMENTS(moments_int, int)
DEFINE_MOMENTS(moments_uint, uint)
DEFINE_MOMENTS(moments_char, char)
DEFINE_MOMENTS(moments_uchar, uchar)
DEFINE_MOMENTS(moments_short, short)
DEFINE_MOMENTS(moments_ushort, ushort)
