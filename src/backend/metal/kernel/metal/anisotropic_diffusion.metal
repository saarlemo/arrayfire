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
    ulong strides[4];
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
    if (gid != 0) return;
    const ulong total = params.dims[0] * params.dims[1] * params.dims[2] *
                        params.dims[3];
    for (ulong index = 0; index < total; ++index) output[index] = input[index];
    for (ulong w = 0; w < params.dims[3]; ++w) {
        for (ulong z = 0; z < params.dims[2]; ++z) {
            const ulong base = z * params.strides[2] + w * params.strides[3];
            for (ulong y = 1; y + 1 < params.dims[1]; ++y) {
                for (ulong x = 1; x + 1 < params.dims[0]; ++x) {
                    const ulong center = base + x * params.strides[0] +
                                         y * params.strides[1];
                    const long sx = long(params.strides[0]);
                    const long sy = long(params.strides[1]);
                    const long c = long(center);
                    float delta;
                    if (params.curvature != 0) {
                        delta = curvature_update(
                            params.mct, output[c - sy - sx], output[c - sy],
                            output[c - sy + sx], output[c - sx], output[c],
                            output[c + sx], output[c + sy - sx],
                            output[c + sy], output[c + sy + sx]);
                    } else {
                        delta = gradient_update(
                            params.mct, output[c - sy - sx], output[c - sy],
                            output[c - sy + sx], output[c - sx], output[c],
                            output[c + sx], output[c + sy - sx],
                            output[c + sy], output[c + sy + sx], params.flux);
                    }
                    output[center] += delta * params.dt;
                }
            }
        }
    }
}
