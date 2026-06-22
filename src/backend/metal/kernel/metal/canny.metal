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

struct CannyParams {
    ulong dims[4];
    ulong outputStrides[4];
    ulong firstStrides[4];
    ulong secondStrides[4];
    ulong thirdStrides[4];
};

kernel void canny_nonmax_float(
    const device float* magnitude [[buffer(0)]],
    const device float* derivativeX [[buffer(1)]],
    const device float* derivativeY [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant CannyParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong total = params.dims[0] * params.dims[1] * params.dims[2] *
                        params.dims[3];
    if (gid >= total) return;

    ulong q = gid;
    const ulong x = q % params.dims[0];
    q /= params.dims[0];
    const ulong y = q % params.dims[1];
    q /= params.dims[1];
    const ulong z = q % params.dims[2];
    const ulong w = q / params.dims[2];
    const ulong outputOffset = x * params.outputStrides[0] +
                               y * params.outputStrides[1] +
                               z * params.outputStrides[2] +
                               w * params.outputStrides[3];
    output[outputOffset] = 0.0f;
    if (x == 0 || y == 0 || x + 1 >= params.dims[0] ||
        y + 1 >= params.dims[1]) return;

    const ulong magnitudeOffset = x * params.firstStrides[0] +
                                  y * params.firstStrides[1] +
                                  z * params.firstStrides[2] +
                                  w * params.firstStrides[3];
    const float current = magnitude[magnitudeOffset];
    if (current == 0.0f) return;

    const ulong dxOffset = x * params.secondStrides[0] +
                           y * params.secondStrides[1] +
                           z * params.secondStrides[2] +
                           w * params.secondStrides[3];
    const ulong dyOffset = x * params.thirdStrides[0] +
                           y * params.thirdStrides[1] +
                           z * params.thirdStrides[2] +
                           w * params.thirdStrides[3];
    const float dx = derivativeX[dxOffset];
    const float dy = derivativeY[dyOffset];
    const long strideX = long(params.firstStrides[0]);
    const long strideY = long(params.firstStrides[1]);
    const long center = long(magnitudeOffset);
    const float southeast = magnitude[center + strideY + strideX];
    const float northwest = magnitude[center - strideY - strideX];
    const float east = magnitude[center + strideX];
    const float west = magnitude[center - strideX];
    const float northeast = magnitude[center - strideY + strideX];
    const float southwest = magnitude[center + strideY - strideX];
    const float north = magnitude[center - strideY];
    const float south = magnitude[center + strideY];

    float a1;
    float a2;
    float b1;
    float b2;
    float alpha;
    if (dx >= 0.0f) {
        if (dy >= 0.0f) {
            const bool dxIsGreater = dx - dy >= 0.0f;
            a1 = dxIsGreater ? east : south;
            a2 = dxIsGreater ? west : north;
            b1 = southeast;
            b2 = northwest;
            alpha = dxIsGreater ? dy / dx : dx / dy;
        } else {
            const bool dxIsGreater = dx + dy >= 0.0f;
            a1 = dxIsGreater ? east : north;
            a2 = dxIsGreater ? west : south;
            b1 = northeast;
            b2 = southwest;
            alpha = dxIsGreater ? -dy / dx : dx / -dy;
        }
    } else if (dy >= 0.0f) {
        const bool dyIsGreater = dx + dy >= 0.0f;
        a1 = dyIsGreater ? south : west;
        a2 = dyIsGreater ? north : east;
        b1 = southwest;
        b2 = northeast;
        alpha = dyIsGreater ? -dx / dy : dy / -dx;
    } else {
        const bool dxIsGreater = -dx + dy >= 0.0f;
        a1 = dxIsGreater ? west : north;
        a2 = dxIsGreater ? east : south;
        b1 = northwest;
        b2 = southeast;
        alpha = dxIsGreater ? dy / dx : dx / dy;
    }

    const float magnitude1 = (1.0f - alpha) * a1 + alpha * b1;
    const float magnitude2 = (1.0f - alpha) * a2 + alpha * b2;
    if (current > magnitude1 && current > magnitude2)
        output[outputOffset] = current;
}

kernel void canny_hysteresis_init_char(
    const device char* strong [[buffer(0)]],
    device char* output [[buffer(1)]],
    constant CannyParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong total = params.dims[0] * params.dims[1] * params.dims[2] *
                        params.dims[3];
    if (gid >= total) return;
    ulong q = gid;
    const ulong x = q % params.dims[0];
    q /= params.dims[0];
    const ulong y = q % params.dims[1];
    q /= params.dims[1];
    const ulong z = q % params.dims[2];
    const ulong w = q / params.dims[2];
    const ulong outputOffset = x * params.outputStrides[0] +
                               y * params.outputStrides[1] +
                               z * params.outputStrides[2] +
                               w * params.outputStrides[3];
    const ulong strongOffset = x * params.firstStrides[0] +
                               y * params.firstStrides[1] +
                               z * params.firstStrides[2] +
                               w * params.firstStrides[3];
    output[outputOffset] = strong[strongOffset] > 0 ? char(1) : char(0);
}

kernel void canny_hysteresis_step_char(
    const device char* weak [[buffer(0)]], device char* output [[buffer(1)]],
    device atomic_uint* changed [[buffer(2)]],
    constant CannyParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong total = params.dims[0] * params.dims[1] * params.dims[2] *
                        params.dims[3];
    if (gid >= total) return;
    ulong q = gid;
    const ulong x = q % params.dims[0];
    q /= params.dims[0];
    const ulong y = q % params.dims[1];
    q /= params.dims[1];
    const ulong z = q % params.dims[2];
    const ulong w = q / params.dims[2];
    if (x == 0 || y == 0 || x + 1 >= params.dims[0] ||
        y + 1 >= params.dims[1]) return;
    const ulong outputOffset = x * params.outputStrides[0] +
                               y * params.outputStrides[1] +
                               z * params.outputStrides[2] +
                               w * params.outputStrides[3];
    if (output[outputOffset] != 0) return;
    const ulong weakOffset = x * params.secondStrides[0] +
                             y * params.secondStrides[1] +
                             z * params.secondStrides[2] +
                             w * params.secondStrides[3];
    if (weak[weakOffset] <= 0) return;
    bool connected = false;
    for (long ny = long(y) - 1; ny <= long(y) + 1; ++ny) {
        for (long nx = long(x) - 1; nx <= long(x) + 1; ++nx) {
            const ulong neighbor = ulong(nx) * params.outputStrides[0] +
                                   ulong(ny) * params.outputStrides[1] +
                                   z * params.outputStrides[2] +
                                   w * params.outputStrides[3];
            connected = connected || output[neighbor] != 0;
        }
    }
    if (connected) {
        output[outputOffset] = 1;
        atomic_store_explicit(changed, 1u, memory_order_relaxed);
    }
}
