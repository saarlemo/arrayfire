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

// Metal does not support pointer reinterpret_cast. Read the index bytes
// explicitly, then use as_type only for scalar bit reinterpretation.
inline ushort readIndexU16(const device uchar* indices,
                           const ulong offset) {
    return static_cast<ushort>(indices[offset]) |
           (static_cast<ushort>(indices[offset + 1]) << 8);
}

inline uint readIndexU32(const device uchar* indices, const ulong offset) {
    return static_cast<uint>(indices[offset]) |
           (static_cast<uint>(indices[offset + 1]) << 8) |
           (static_cast<uint>(indices[offset + 2]) << 16) |
           (static_cast<uint>(indices[offset + 3]) << 24);
}

inline ulong readIndexU64(const device uchar* indices,
                          const ulong offset) {
    return static_cast<ulong>(indices[offset]) |
           (static_cast<ulong>(indices[offset + 1]) << 8) |
           (static_cast<ulong>(indices[offset + 2]) << 16) |
           (static_cast<ulong>(indices[offset + 3]) << 24) |
           (static_cast<ulong>(indices[offset + 4]) << 32) |
           (static_cast<ulong>(indices[offset + 5]) << 40) |
           (static_cast<ulong>(indices[offset + 6]) << 48) |
           (static_cast<ulong>(indices[offset + 7]) << 56);
}

long readIndex(const device uchar* indices, const ulong index,
               const uint type) {
    switch (type) {
        case 0:
            return static_cast<long>(as_type<float>(
                readIndexU32(indices, index * 4)));
        case 1:
            return static_cast<long>(as_type<int>(
                readIndexU32(indices, index * 4)));
        case 2:
            return static_cast<long>(readIndexU32(indices, index * 4));
        case 3:
            return as_type<long>(readIndexU64(indices, index * 8));
        case 4:
            return static_cast<long>(readIndexU64(indices, index * 8));
        case 5: {
            const uint value = static_cast<uint>(indices[index]);
            return (value & 0x80u) != 0
                       ? static_cast<long>(value) - 256
                       : static_cast<long>(value);
        }
        case 6:
            return static_cast<long>(indices[index]);
        case 7: {
            const uint bits = static_cast<uint>(
                readIndexU16(indices, index * 2));
            return (bits & 0x8000u) != 0
                       ? static_cast<long>(bits) - 65536
                       : static_cast<long>(bits);
        }
        case 8:
            return static_cast<long>(readIndexU16(indices, index * 2));
        case 9:
            return static_cast<long>(as_type<half>(
                readIndexU16(indices, index * 2)));
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
