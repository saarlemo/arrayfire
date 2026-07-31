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

struct FFTConvolveParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong signalDims[4];
    ulong signalStrides[4];
    ulong filterDims[4];
    ulong filterStrides[4];
    ulong offset;
    uint kind;
};

struct FFTConvolvePackParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
};

struct FFTConvolvePadParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    ulong offset;
};

struct FFTConvolveReorderParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong inputDims[4];
    ulong inputStrides[4];
    ulong filterDims[4];
    ulong filterOffset;
    ulong signalHalfDim0;
    ulong fftScale;
    uint kind;
};

kernel void fftconvolve_multiply_float(
    device float* packed [[buffer(0)]],
    constant FFTConvolveParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong complexDim0 = p.outputDims[0] / 2;
    const ulong total = complexDim0 * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;
    ulong q = gid;
    const ulong d0 = q % complexDim0; q /= complexDim0;
    const ulong d1 = q % p.outputDims[1]; q /= p.outputDims[1];
    const ulong d2 = q % p.outputDims[2];
    const ulong d3 = q / p.outputDims[2];
    const ulong index = d3 * p.outputStrides[3] +
                        d2 * p.outputStrides[2] +
                        d1 * p.outputStrides[1] + d0 * 2;
    ulong firstIndex = index;
    ulong secondIndex = p.offset + index;
    ulong outputIndex = index;
    if (p.kind == 1) {
        const ulong filterBatch = p.filterStrides[3] * p.filterDims[3];
        secondIndex = p.offset + index % filterBatch;
    } else if (p.kind == 2) {
        const ulong signalBatch = p.signalStrides[3] * p.signalDims[3];
        firstIndex = index % signalBatch;
        outputIndex = p.offset + index;
    }
    const float ar = packed[firstIndex];
    const float ai = packed[firstIndex + 1];
    const float br = packed[secondIndex];
    const float bi = packed[secondIndex + 1];
    packed[outputIndex] = ar * br - ai * bi;
    packed[outputIndex + 1] = ar * bi + ai * br;
}

#define DEFINE_FFTCONVOLVE_PACK(NAME, TYPE)                                  \
kernel void NAME(const device TYPE* input [[buffer(0)]],                     \
                 device float* output [[buffer(1)]],                        \
                 constant FFTConvolvePackParams& p [[buffer(2)]],            \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong complexDim0 = p.outputDims[0] / 2;                            \
    const ulong total = complexDim0 * p.outputDims[1] *                       \
                        p.outputDims[2] * p.outputDims[3];                    \
    if (gid >= total) return;                                                  \
    ulong q = gid;                                                             \
    const ulong d0 = q % complexDim0; q /= complexDim0;                        \
    const ulong d1 = q % p.outputDims[1]; q /= p.outputDims[1];                \
    const ulong d2 = q % p.outputDims[2];                                       \
    const ulong d3 = q / p.outputDims[2];                                       \
    const ulong outIndex = d3 * p.outputStrides[3] +                          \
                           d2 * p.outputStrides[2] +                          \
                           d1 * p.outputStrides[1] + d0 * 2;                  \
    const ulong inputHalf = (p.inputDims[0] + 1) / 2;                         \
    if (d0 < inputHalf && d1 < p.inputDims[1] &&                              \
        d2 < p.inputDims[2] && d3 < p.inputDims[3]) {                          \
        const ulong inputIndex = d3 * p.inputStrides[3] +                     \
                                  d2 * p.inputStrides[2] +                     \
                                  d1 * p.inputStrides[1] + d0;                 \
        output[outIndex] = float(input[inputIndex]);                           \
        output[outIndex + 1] = (d0 + 1 == inputHalf &&                        \
                                (p.inputDims[0] & 1) != 0)                      \
                                   ? 0.0f                                      \
                                   : float(input[inputIndex + inputHalf]);     \
    } else {                                                                   \
        output[outIndex] = 0.0f;                                               \
        output[outIndex + 1] = 0.0f;                                           \
    }                                                                          \
}

#define DEFINE_FFTCONVOLVE_PAD(NAME, TYPE)                                   \
kernel void NAME(const device TYPE* input [[buffer(0)]],                     \
                 device float* output [[buffer(1)]],                        \
                 constant FFTConvolvePadParams& p [[buffer(2)]],              \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong complexDim0 = p.outputDims[0] / 2;                            \
    const ulong total = complexDim0 * p.outputDims[1] *                       \
                        p.outputDims[2] * p.outputDims[3];                    \
    if (gid >= total) return;                                                  \
    ulong q = gid;                                                             \
    const ulong d0 = q % complexDim0; q /= complexDim0;                        \
    const ulong d1 = q % p.outputDims[1]; q /= p.outputDims[1];                \
    const ulong d2 = q % p.outputDims[2];                                       \
    const ulong d3 = q / p.outputDims[2];                                       \
    const ulong outIndex = p.offset + d3 * p.outputStrides[3] +                \
                           d2 * p.outputStrides[2] +                           \
                           d1 * p.outputStrides[1] + d0 * 2;                  \
    if (d0 < p.inputDims[0] && d1 < p.inputDims[1] &&                          \
        d2 < p.inputDims[2] && d3 < p.inputDims[3]) {                          \
        const ulong inputIndex = d3 * p.inputStrides[3] +                     \
                                  d2 * p.inputStrides[2] +                     \
                                  d1 * p.inputStrides[1] + d0;                 \
        output[outIndex] = float(input[inputIndex]);                           \
    } else {                                                                   \
        output[outIndex] = 0.0f;                                               \
    }                                                                          \
    output[outIndex + 1] = 0.0f;                                               \
}

#define DEFINE_FFTCONVOLVE_REORDER(NAME, TYPE, RANK, EXPAND, ROUND_RESULT)    \
kernel void NAME(const device float* input [[buffer(0)]],                     \
                 device TYPE* output [[buffer(1)]],                           \
                 constant FFTConvolveReorderParams& p [[buffer(2)]],           \
                 uint gid [[thread_position_in_grid]]) {                     \
    const ulong total = p.outputDims[0] * p.outputDims[1] *                    \
                        p.outputDims[2] * p.outputDims[3];                     \
    if (gid >= total) return;                                                   \
    ulong q = gid;                                                              \
    const ulong d0 = q % p.outputDims[0]; q /= p.outputDims[0];                 \
    const ulong d1 = q % p.outputDims[1]; q /= p.outputDims[1];                 \
    const ulong d2 = q % p.outputDims[2];                                       \
    const ulong d3 = q / p.outputDims[2];                                       \
    ulong id0 = EXPAND ? d0 : d0 + p.filterDims[0] / 2;                        \
    ulong id1 = EXPAND ? d1 * p.inputStrides[1] :                              \
                (d1 + (RANK > 1 ? p.filterDims[1] / 2 : 0)) *                 \
                    p.inputStrides[1];                                        \
    ulong id2 = EXPAND ? d2 * p.inputStrides[2] :                              \
                (d2 + (RANK > 2 ? p.filterDims[2] / 2 : 0)) *                 \
                    p.inputStrides[2];                                        \
    const ulong id3 = d3 * p.inputStrides[3];                                  \
    const ulong inputBase = p.kind == 2 ? p.filterOffset : 0;                 \
    const ulong outputIndex = d3 * p.outputStrides[3] +                        \
                              d2 * p.outputStrides[2] +                        \
                              d1 * p.outputStrides[1] + d0;                    \
    float value;                                                               \
    if (id0 < p.signalHalfDim0) {                                              \
        value = input[inputBase + id3 + id2 + id1 + id0 * 2] /                 \
                float(p.fftScale);                                            \
    } else if (id0 < p.signalHalfDim0 + p.filterDims[0] - 1) {                 \
        const ulong first = inputBase + id3 + id2 + id1 + id0 * 2;            \
        const ulong second = inputBase + id3 + id2 + id1 +                     \
                             (id0 - p.signalHalfDim0) * 2 + 1;                 \
        value = (input[first] + input[second]) / float(p.fftScale);            \
    } else {                                                                    \
        const ulong index = inputBase + id3 + id2 + id1 +                      \
                            (id0 - p.signalHalfDim0) * 2 + 1;                 \
        value = input[index] / float(p.fftScale);                              \
    }                                                                         \
    output[outputIndex] = ROUND_RESULT ? TYPE(round(value)) : TYPE(value);     \
}

DEFINE_FFTCONVOLVE_PACK(fftconvolve_pack_float, float)
DEFINE_FFTCONVOLVE_PACK(fftconvolve_pack_int, int)
DEFINE_FFTCONVOLVE_PACK(fftconvolve_pack_uint, uint)
DEFINE_FFTCONVOLVE_PACK(fftconvolve_pack_long, long)
DEFINE_FFTCONVOLVE_PACK(fftconvolve_pack_ulong, ulong)
DEFINE_FFTCONVOLVE_PACK(fftconvolve_pack_char, char)
DEFINE_FFTCONVOLVE_PACK(fftconvolve_pack_uchar, uchar)
DEFINE_FFTCONVOLVE_PACK(fftconvolve_pack_short, short)
DEFINE_FFTCONVOLVE_PACK(fftconvolve_pack_ushort, ushort)

DEFINE_FFTCONVOLVE_PAD(fftconvolve_pad_float, float)
DEFINE_FFTCONVOLVE_PAD(fftconvolve_pad_int, int)
DEFINE_FFTCONVOLVE_PAD(fftconvolve_pad_uint, uint)
DEFINE_FFTCONVOLVE_PAD(fftconvolve_pad_long, long)
DEFINE_FFTCONVOLVE_PAD(fftconvolve_pad_ulong, ulong)
DEFINE_FFTCONVOLVE_PAD(fftconvolve_pad_char, char)
DEFINE_FFTCONVOLVE_PAD(fftconvolve_pad_uchar, uchar)
DEFINE_FFTCONVOLVE_PAD(fftconvolve_pad_short, short)
DEFINE_FFTCONVOLVE_PAD(fftconvolve_pad_ushort, ushort)

#define DEFINE_REORDER_FOR_TYPE(TYPE_NAME, TYPE, ROUND_RESULT)                \
DEFINE_FFTCONVOLVE_REORDER(fftconvolve_reorder_##TYPE_NAME##_1_expand, TYPE, 1, true, ROUND_RESULT) \
DEFINE_FFTCONVOLVE_REORDER(fftconvolve_reorder_##TYPE_NAME##_2_expand, TYPE, 2, true, ROUND_RESULT) \
DEFINE_FFTCONVOLVE_REORDER(fftconvolve_reorder_##TYPE_NAME##_3_expand, TYPE, 3, true, ROUND_RESULT) \
DEFINE_FFTCONVOLVE_REORDER(fftconvolve_reorder_##TYPE_NAME##_1_same, TYPE, 1, false, ROUND_RESULT) \
DEFINE_FFTCONVOLVE_REORDER(fftconvolve_reorder_##TYPE_NAME##_2_same, TYPE, 2, false, ROUND_RESULT) \
DEFINE_FFTCONVOLVE_REORDER(fftconvolve_reorder_##TYPE_NAME##_3_same, TYPE, 3, false, ROUND_RESULT)

DEFINE_REORDER_FOR_TYPE(float, float, false)
DEFINE_REORDER_FOR_TYPE(int, int, true)
DEFINE_REORDER_FOR_TYPE(uint, uint, true)
DEFINE_REORDER_FOR_TYPE(long, long, true)
DEFINE_REORDER_FOR_TYPE(ulong, ulong, true)
DEFINE_REORDER_FOR_TYPE(char, char, true)
DEFINE_REORDER_FOR_TYPE(uchar, uchar, true)
DEFINE_REORDER_FOR_TYPE(short, short, true)
DEFINE_REORDER_FOR_TYPE(ushort, ushort, true)
