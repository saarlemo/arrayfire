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

struct DiffusionParams {
    ulong dims[4];
    ulong inputStrides[4];
    ulong outputStrides[4];
    float dt;
    float mct;
    uint flux;
    uint curvature;
};

float gradient_update(float mct, float nw, float n, float ne, float w,
                      float c, float e, float sw, float s, float se,
                      uint flux) {
    const float dx = (e - w) * 0.5f;
    const float dy = (s - n) * 0.5f;
    float df = e - c;
    float db = c - w;
    float forward = (df * df +
                     0.25f * pow(dy + 0.5f * (se - ne), 2.0f)) * mct;
    float backward = (db * db +
                      0.25f * pow(dy + 0.5f * (sw - nw), 2.0f)) * mct;
    float cf = flux == 2 ? exp(forward) : 1.0f / (1.0f + forward);
    float cb = flux == 2 ? exp(backward) : 1.0f / (1.0f + backward);
    float delta = cf * df - cb * db;

    df = s - c;
    db = c - n;
    forward = (df * df +
               0.25f * pow(dx + 0.5f * (se - sw), 2.0f)) * mct;
    backward = (db * db +
                0.25f * pow(dx + 0.5f * (ne - nw), 2.0f)) * mct;
    cf = flux == 2 ? exp(forward) : 1.0f / (1.0f + forward);
    cb = flux == 2 ? exp(backward) : 1.0f / (1.0f + backward);
    return delta + cf * df - cb * db;
}

float curvature_update(float mct, float nw, float n, float ne, float w,
                       float c, float e, float sw, float s, float se) {
    const float dx = (e - w) * 0.5f;
    const float dy = (s - n) * 0.5f;
    float df = e - c;
    float db = c - w;
    const float df0 = df;
    const float db0 = db;
    float forward = df * df +
                    0.25f * pow(dy + 0.5f * (se - ne), 2.0f);
    float backward = db * db +
                     0.25f * pow(dy + 0.5f * (sw - nw), 2.0f);
    float delta = (df / sqrt(1.0e-10f + forward)) * exp(forward * mct) -
                  (db / sqrt(1.0e-10f + backward)) * exp(backward * mct);

    df = s - c;
    db = c - n;
    forward = df * df + 0.25f * pow(dx + 0.5f * (se - sw), 2.0f);
    backward = db * db + 0.25f * pow(dx + 0.5f * (ne - nw), 2.0f);
    delta += (df / sqrt(1.0e-10f + forward)) * exp(forward * mct) -
             (db / sqrt(1.0e-10f + backward)) * exp(backward * mct);

    float propagated = 0.0f;
    if (delta > 0.0f) {
        propagated += pow(min(db0, 0.0f), 2.0f) + pow(max(df0, 0.0f), 2.0f);
        propagated += pow(min(db, 0.0f), 2.0f) + pow(max(df, 0.0f), 2.0f);
    } else {
        propagated += pow(max(db0, 0.0f), 2.0f) + pow(min(df0, 0.0f), 2.0f);
        propagated += pow(max(db, 0.0f), 2.0f) + pow(min(df, 0.0f), 2.0f);
    }
    return sqrt(propagated) * delta;
}

kernel void anisotropic_diffusion_float(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant DiffusionParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong width = params.dims[0];
    const ulong height = params.dims[1];
    const ulong plane = width * height;
    const ulong total = plane * params.dims[2] * params.dims[3];
    if (gid >= total) return;

    const ulong x = gid % width;
    const ulong y = (gid / width) % height;
    const ulong z = (gid / plane) % params.dims[2];
    const ulong w = gid / (plane * params.dims[2]);
    const ulong inputBase = z * params.inputStrides[2] +
                            w * params.inputStrides[3];
    const ulong outputBase = z * params.outputStrides[2] +
                             w * params.outputStrides[3];
    const ulong inputCenter = inputBase + x * params.inputStrides[0] +
                              y * params.inputStrides[1];
    const ulong outputCenter = outputBase + x * params.outputStrides[0] +
                               y * params.outputStrides[1];

    // Keep the existing fixed-border behavior. Every pixel reads only from
    // the immutable input, so the update is race-free and parallel over pixels.
    if (x == 0 || y == 0 || x + 1 >= width || y + 1 >= height) {
        output[outputCenter] = input[inputCenter];
        return;
    }

    const long sx = long(params.inputStrides[0]);
    const long sy = long(params.inputStrides[1]);
    const long c = long(inputCenter);
    float delta;
    if (params.curvature != 0) {
        delta = curvature_update(
            params.mct, input[c - sy - sx], input[c - sy],
            input[c - sy + sx], input[c - sx], input[c], input[c + sx],
            input[c + sy - sx], input[c + sy], input[c + sy + sx]);
    } else {
        delta = gradient_update(
            params.mct, input[c - sy - sx], input[c - sy],
            input[c - sy + sx], input[c - sx], input[c], input[c + sx],
            input[c + sy - sx], input[c + sy], input[c + sy + sx], params.flux);
    }
    output[outputCenter] = input[inputCenter] + delta * params.dt;
}
