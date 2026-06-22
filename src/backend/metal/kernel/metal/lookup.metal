// Copyright (c) 2026, ArrayFire
// SPDX-License-Identifier: BSD-3-Clause

#include <metal_stdlib>
using namespace metal;

struct LookupParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    uint dimension;
    uint indexType;
};

long readIndex(const device uchar* indices, const ulong index,
               const uint type) {
    switch (type) {
        case 0:
            return static_cast<long>(
                reinterpret_cast<const device float*>(indices)[index]);
        case 1:
            return static_cast<long>(
                reinterpret_cast<const device int*>(indices)[index]);
        case 2:
            return static_cast<long>(
                reinterpret_cast<const device uint*>(indices)[index]);
        case 3:
            return reinterpret_cast<const device long*>(indices)[index];
        case 4:
            return static_cast<long>(
                reinterpret_cast<const device ulong*>(indices)[index]);
        case 5:
            return static_cast<long>(
                reinterpret_cast<const device char*>(indices)[index]);
        case 6:
            return static_cast<long>(
                reinterpret_cast<const device uchar*>(indices)[index]);
        case 7:
            return static_cast<long>(
                reinterpret_cast<const device short*>(indices)[index]);
        case 8:
            return static_cast<long>(
                reinterpret_cast<const device ushort*>(indices)[index]);
        case 9:
            return static_cast<long>(
                reinterpret_cast<const device half*>(indices)[index]);
        default:
            return 0;
    }
}

ulong trimLookupIndex(long index, const ulong length) {
    if (index < 0) {
        return static_cast<ulong>((-index - 1) %
                                  static_cast<long>(length));
    }
    if (static_cast<ulong>(index) >= length) {
        const ulong offset = static_cast<ulong>(index) % length;
        return length - offset - 1;
    }
    return static_cast<ulong>(index);
}

#define DEFINE_LOOKUP_KERNEL(NAME, TYPE)                                    \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 const device uchar* indices [[buffer(1)]],                 \
                 device TYPE* output [[buffer(2)]],                         \
                 constant LookupParams& params [[buffer(3)]],               \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total = params.outputDims[0] * params.outputDims[1] *        \
                        params.outputDims[2] * params.outputDims[3];          \
    if (gid >= total) return;                                                \
    ulong position = gid;                                                    \
    const ulong x = position % params.outputDims[0];                         \
    position /= params.outputDims[0];                                        \
    const ulong y = position % params.outputDims[1];                         \
    position /= params.outputDims[1];                                        \
    const ulong z = position % params.outputDims[2];                         \
    const ulong w = position / params.outputDims[2];                         \
    ulong coordinates[4] = {x, y, z, w};                                    \
    coordinates[params.dimension] = trimLookupIndex(                         \
        readIndex(indices, coordinates[params.dimension], params.indexType), \
        params.inputDims[params.dimension]);                                 \
    const ulong inputOffset = coordinates[0] * params.inputStrides[0] +      \
        coordinates[1] * params.inputStrides[1] +                            \
        coordinates[2] * params.inputStrides[2] +                            \
        coordinates[3] * params.inputStrides[3];                             \
    const ulong outputOffset = x * params.outputStrides[0] +                 \
        y * params.outputStrides[1] + z * params.outputStrides[2] +          \
        w * params.outputStrides[3];                                         \
    output[outputOffset] = input[inputOffset];                               \
}

DEFINE_LOOKUP_KERNEL(lookup_float, float)
DEFINE_LOOKUP_KERNEL(lookup_cfloat, float2)
DEFINE_LOOKUP_KERNEL(lookup_int, int)
DEFINE_LOOKUP_KERNEL(lookup_uint, uint)
DEFINE_LOOKUP_KERNEL(lookup_long, long)
DEFINE_LOOKUP_KERNEL(lookup_ulong, ulong)
DEFINE_LOOKUP_KERNEL(lookup_char, char)
DEFINE_LOOKUP_KERNEL(lookup_uchar, uchar)
DEFINE_LOOKUP_KERNEL(lookup_short, short)
DEFINE_LOOKUP_KERNEL(lookup_ushort, ushort)
DEFINE_LOOKUP_KERNEL(lookup_half, half)
