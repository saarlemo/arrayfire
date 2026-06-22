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

struct HistogramParams {
    ulong inputDims[4], outputStrides[4], inputStrides[4];
    uint bins;
    float minValue, maxValue;
    uint linear;
};

#define DEFINE_HISTOGRAM(NAME, TYPE)                                       \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 device uint* output [[buffer(1)]],                        \
                 constant HistogramParams& p [[buffer(2)]],                \
                 uint gid [[thread_position_in_grid]]) {                   \
    const ulong batches=p.inputDims[2]*p.inputDims[3];                     \
    if (gid>=batches) return;                                              \
    const ulong z=gid%p.inputDims[2], w=gid/p.inputDims[2];                \
    const ulong ii=z*p.inputStrides[2]+w*p.inputStrides[3];                \
    const ulong oo=z*p.outputStrides[2]+w*p.outputStrides[3];              \
    for (uint bin=0; bin<p.bins; ++bin) output[oo+bin]=0;                  \
    const ulong count=p.inputDims[0]*p.inputDims[1];                       \
    for (ulong i=0; i<count; ++i) {                                        \
        const ulong index=p.linear ? i : (i%p.inputDims[0])*               \
            p.inputStrides[0]+(i/p.inputDims[0])*p.inputStrides[1];        \
        float scaled=(float(input[ii+index])-p.minValue)*                  \
            float(p.bins)/(p.maxValue-p.minValue);                         \
        int bin=int(floor(nextafter(scaled, INFINITY)));                   \
        bin=clamp(bin,0,int(p.bins)-1);                                    \
        output[oo+ulong(bin)]++;                                           \
    }                                                                      \
}

DEFINE_HISTOGRAM(histogram_float, float)
DEFINE_HISTOGRAM(histogram_int, int)
DEFINE_HISTOGRAM(histogram_uint, uint)
DEFINE_HISTOGRAM(histogram_long, long)
DEFINE_HISTOGRAM(histogram_ulong, ulong)
DEFINE_HISTOGRAM(histogram_char, char)
DEFINE_HISTOGRAM(histogram_uchar, uchar)
DEFINE_HISTOGRAM(histogram_short, short)
DEFINE_HISTOGRAM(histogram_ushort, ushort)
DEFINE_HISTOGRAM(histogram_half, half)
