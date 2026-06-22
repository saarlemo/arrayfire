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

struct ScanParams {
    ulong dims[4];
    ulong outputStrides[4];
    ulong inputStrides[4];
    uint dimension;
};

#define DEFINE_SCAN(NAME, TYPE)                                               \
kernel void NAME(const device TYPE* input [[buffer(0)]],                      \
                 device TYPE* output [[buffer(1)]],                           \
                 constant ScanParams& params [[buffer(2)]],                   \
                 uint gid [[thread_position_in_grid]]) {                      \
    const ulong total = params.dims[0] * params.dims[1] * params.dims[2] *    \
                        params.dims[3];                                       \
    if (gid >= total) return;                                                  \
    ulong q = gid;                                                             \
    ulong coordinates[4];                                                      \
    coordinates[0] = q % params.dims[0];                                     \
    q /= params.dims[0];                                                       \
    coordinates[1] = q % params.dims[1];                                     \
    q /= params.dims[1];                                                       \
    coordinates[2] = q % params.dims[2];                                     \
    coordinates[3] = q / params.dims[2];                                     \
    if (coordinates[params.dimension] != 0) return;                           \
    ulong inputBase = 0;                                                       \
    ulong outputBase = 0;                                                      \
    for (uint d = 0; d < 4; ++d) {                                            \
        inputBase += coordinates[d] * params.inputStrides[d];                 \
        outputBase += coordinates[d] * params.outputStrides[d];               \
    }                                                                          \
    TYPE accumulated = TYPE(0);                                                \
    for (ulong i = 0; i < params.dims[params.dimension]; ++i) {               \
        accumulated += input[inputBase + i *                                  \
                             params.inputStrides[params.dimension]];          \
        output[outputBase + i * params.outputStrides[params.dimension]] =     \
            accumulated;                                                       \
    }                                                                          \
}

DEFINE_SCAN(scan_add_float, float)
DEFINE_SCAN(scan_add_int, int)
DEFINE_SCAN(scan_add_uint, uint)
DEFINE_SCAN(scan_add_long, long)
DEFINE_SCAN(scan_add_ulong, ulong)
DEFINE_SCAN(scan_add_char, char)
DEFINE_SCAN(scan_add_uchar, uchar)
DEFINE_SCAN(scan_add_short, short)
DEFINE_SCAN(scan_add_ushort, ushort)
