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

struct NearestNeighbourParams {
    ulong outputDims[4], outputStrides[4];
    ulong queryDims[4], queryStrides[4];
    ulong trainDims[4], trainStrides[4];
    uint distanceDimension, distanceType;
};

#define DEFINE_NEAREST(NAME, TYPE, RESULT)                                 \
kernel void NAME(const device TYPE* query [[buffer(0)]],                   \
                 const device TYPE* train [[buffer(1)]],                   \
                 device RESULT* output [[buffer(2)]],                      \
                 constant NearestNeighbourParams& p [[buffer(3)]],         \
                 uint gid [[thread_position_in_grid]]) {                   \
    const ulong total=p.outputDims[0]*p.outputDims[1];                     \
    if (gid>=total) return;                                                \
    const ulong t=gid%p.outputDims[0], q=gid/p.outputDims[0];              \
    const ulong length=p.queryDims[p.distanceDimension];                   \
    RESULT result=RESULT(0);                                               \
    for (ulong k=0; k<length; ++k) {                                       \
        const ulong qi=p.distanceDimension==0 ?                            \
            k*p.queryStrides[0]+q*p.queryStrides[1] :                     \
            q*p.queryStrides[0]+k*p.queryStrides[1];                      \
        const ulong ti=p.distanceDimension==0 ?                            \
            k*p.trainStrides[0]+t*p.trainStrides[1] :                     \
            t*p.trainStrides[0]+k*p.trainStrides[1];                      \
        const RESULT delta=RESULT(query[qi])-RESULT(train[ti]);            \
        const RESULT sad=query[qi]>=train[ti] ?                            \
            RESULT(query[qi])-RESULT(train[ti]) :                         \
            RESULT(train[ti])-RESULT(query[qi]);                          \
        result += p.distanceType==0 ? sad : delta*delta;                  \
    }                                                                      \
    output[t*p.outputStrides[0]+q*p.outputStrides[1]]=result;              \
}

#define DEFINE_HAMMING(NAME, TYPE)                                        \
kernel void NAME(const device TYPE* query [[buffer(0)]],                   \
                 const device TYPE* train [[buffer(1)]],                   \
                 device uint* output [[buffer(2)]],                        \
                 constant NearestNeighbourParams& p [[buffer(3)]],         \
                 uint gid [[thread_position_in_grid]]) {                   \
    const ulong total=p.outputDims[0]*p.outputDims[1];                     \
    if (gid>=total) return;                                                \
    const ulong t=gid%p.outputDims[0], q=gid/p.outputDims[0];              \
    const ulong length=p.queryDims[p.distanceDimension];                   \
    uint result=0;                                                         \
    for (ulong k=0; k<length; ++k) {                                       \
        const ulong qi=p.distanceDimension==0 ?                            \
            k*p.queryStrides[0]+q*p.queryStrides[1] :                     \
            q*p.queryStrides[0]+k*p.queryStrides[1];                      \
        const ulong ti=p.distanceDimension==0 ?                            \
            k*p.trainStrides[0]+t*p.trainStrides[1] :                     \
            t*p.trainStrides[0]+k*p.trainStrides[1];                      \
        result+=uint(popcount(query[qi]^train[ti]));                       \
    }                                                                      \
    output[t*p.outputStrides[0]+q*p.outputStrides[1]]=result;              \
}

DEFINE_NEAREST(nearest_float, float, float)
DEFINE_NEAREST(nearest_int, int, int)
DEFINE_NEAREST(nearest_uint, uint, uint)
DEFINE_NEAREST(nearest_long, long, long)
DEFINE_NEAREST(nearest_ulong, ulong, ulong)
DEFINE_NEAREST(nearest_char_int, char, int)
DEFINE_NEAREST(nearest_uchar_uint, uchar, uint)
DEFINE_NEAREST(nearest_short_int, short, int)
DEFINE_NEAREST(nearest_ushort_uint, ushort, uint)
DEFINE_HAMMING(nearest_hamming_uchar, uchar)
DEFINE_HAMMING(nearest_hamming_ushort, ushort)
DEFINE_HAMMING(nearest_hamming_uint, uint)
DEFINE_HAMMING(nearest_hamming_ulong, ulong)
