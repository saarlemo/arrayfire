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

struct ScanByKeyParams {
    ulong dims[4];
    ulong outputStrides[4];
    ulong keyStrides[4];
    ulong inputStrides[4];
    uint dimension;
};

#define DEFINE_SCAN_BY_KEY(NAME, VALUE_TYPE)                                 \
kernel void NAME(const device int* keys [[buffer(0)]],                       \
                 const device VALUE_TYPE* input [[buffer(1)]],               \
                 device VALUE_TYPE* output [[buffer(2)]],                    \
                 constant ScanByKeyParams& params [[buffer(3)]],             \
                 uint gid [[thread_position_in_grid]]) {                     \
    const ulong total = params.dims[0] * params.dims[1] * params.dims[2] *   \
                        params.dims[3];                                      \
    if (gid >= total) return;                                                 \
    ulong q = gid;                                                            \
    ulong coordinates[4];                                                     \
    coordinates[0] = q % params.dims[0];                                    \
    q /= params.dims[0];                                                      \
    coordinates[1] = q % params.dims[1];                                    \
    q /= params.dims[1];                                                      \
    coordinates[2] = q % params.dims[2];                                    \
    coordinates[3] = q / params.dims[2];                                    \
    if (coordinates[params.dimension] != 0) return;                          \
    ulong outputBase = 0;                                                     \
    ulong keyBase = 0;                                                        \
    ulong inputBase = 0;                                                      \
    for (uint d = 0; d < 4; ++d) {                                           \
        outputBase += coordinates[d] * params.outputStrides[d];              \
        keyBase += coordinates[d] * params.keyStrides[d];                    \
        inputBase += coordinates[d] * params.inputStrides[d];                \
    }                                                                         \
    int currentKey = keys[keyBase];                                           \
    VALUE_TYPE accumulated = input[inputBase];                               \
    output[outputBase] = accumulated;                                         \
    for (ulong i = 1; i < params.dims[params.dimension]; ++i) {              \
        const ulong keyIndex =                                               \
            keyBase + i * params.keyStrides[params.dimension];               \
        const ulong inputIndex =                                             \
            inputBase + i * params.inputStrides[params.dimension];           \
        const int nextKey = keys[keyIndex];                                  \
        if (nextKey != currentKey) {                                          \
            currentKey = nextKey;                                             \
            accumulated = input[inputIndex];                                 \
        } else {                                                              \
            accumulated += input[inputIndex];                                \
        }                                                                     \
        output[outputBase + i * params.outputStrides[params.dimension]] =    \
            accumulated;                                                      \
    }                                                                         \
}

DEFINE_SCAN_BY_KEY(scan_by_key_add_int_int, int)
DEFINE_SCAN_BY_KEY(scan_by_key_add_int_float, float)
