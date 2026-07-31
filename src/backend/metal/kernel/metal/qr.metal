/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/

#include <metal_stdlib>
using namespace metal;

struct QRFactorParams {
    ulong rows;
    ulong columns;
    ulong stride0;
    ulong stride1;
    uint step;
};

struct QRGenerateParams {
    ulong rows;
    ulong columns;
    ulong outputStride0;
    ulong outputStride1;
    ulong packedStride0;
    ulong packedStride1;
    uint step;
};

float qrAbsSquared(const float value) { return value * value; }
float qrAbsSquared(const float2 value) { return dot(value, value); }

float qrRealPart(const float value) { return value; }
float qrRealPart(const float2 value) { return value.x; }

float qrMagnitude(const float value) { return fabs(value); }
float qrMagnitude(const float2 value) { return length(value); }

float qrConjugate(const float value) { return value; }
float2 qrConjugate(const float2 value) { return float2(value.x, -value.y); }

float qrMultiply(const float lhs, const float rhs) { return lhs * rhs; }
float2 qrMultiply(const float2 lhs, const float2 rhs) {
    return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                  lhs.x * rhs.y + lhs.y * rhs.x);
}

float qrOne(const float) { return 1.0f; }
float2 qrOne(const float2) { return float2(1.0f, 0.0f); }

float qrDivide(const float lhs, const float rhs) { return lhs / rhs; }
float2 qrDivide(const float2 lhs, const float2 rhs) {
    const float denominator = dot(rhs, rhs);
    return float2(lhs.x * rhs.x + lhs.y * rhs.y,
                  lhs.y * rhs.x - lhs.x * rhs.y) /
           denominator;
}

float qrBeta(const float alpha, const float norm) {
    return qrRealPart(alpha) >= 0.0f ? -norm : norm;
}
float2 qrBeta(const float2 alpha, const float norm) {
    const float magnitude = qrMagnitude(alpha);
    return magnitude == 0.0f ? float2(-norm, 0.0f)
                            : -alpha * (norm / magnitude);
}

#define DEFINE_QR(NAME, TYPE)                                                \
kernel void NAME##_factor(device TYPE* input [[buffer(0)]],                  \
                          device TYPE* tau [[buffer(1)]],                   \
                          constant QRFactorParams& p [[buffer(2)]],          \
                          uint gid [[thread_position_in_grid]]) {            \
    if (gid != 0 || p.step >= p.rows || p.step >= p.columns) return;         \
    const ulong diagonal = p.step * p.stride0 + p.step * p.stride1;          \
    const TYPE alpha = input[diagonal];                                      \
    float normSquared = 0.0f;                                                \
    for (ulong row = p.step + 1; row < p.rows; ++row) {                      \
        normSquared += qrAbsSquared(                                         \
            input[row * p.stride0 + p.step * p.stride1]);                    \
    }                                                                        \
    const float norm = sqrt(normSquared + qrAbsSquared(alpha));              \
    if (normSquared == 0.0f) {                                                \
        tau[p.step] = TYPE(0);                                               \
        return;                                                               \
    }                                                                        \
    const TYPE beta = qrBeta(alpha, norm);                                   \
    tau[p.step] = qrDivide(beta - alpha, beta);                              \
    input[diagonal] = beta;                                                  \
    const TYPE scale = qrDivide(qrOne(alpha), alpha - beta);                \
    for (ulong row = p.step + 1; row < p.rows; ++row) {                      \
        const ulong index = row * p.stride0 + p.step * p.stride1;             \
        input[index] = qrMultiply(input[index], scale);                      \
    }                                                                        \
}                                                                            \
kernel void NAME##_apply(device TYPE* input [[buffer(0)]],                   \
                         device TYPE* tau [[buffer(1)]],                    \
                         constant QRFactorParams& p [[buffer(2)]],           \
                         uint gid [[thread_position_in_grid]]) {             \
    const ulong column = gid;                                                 \
    if (column <= p.step || column >= p.columns || p.step >= p.rows) return;  \
    TYPE dot = input[p.step * p.stride0 + column * p.stride1];                \
    for (ulong row = p.step + 1; row < p.rows; ++row) {                      \
        const TYPE v = input[row * p.stride0 + p.step * p.stride1];          \
        dot += qrMultiply(qrConjugate(v),                                    \
                          input[row * p.stride0 + column * p.stride1]);      \
    }                                                                        \
    const TYPE scale = qrMultiply(tau[p.step], dot);                         \
    input[p.step * p.stride0 + column * p.stride1] -= scale;                 \
    for (ulong row = p.step + 1; row < p.rows; ++row) {                      \
        const ulong index = row * p.stride0 + column * p.stride1;             \
        input[index] -= qrMultiply(                                         \
            input[row * p.stride0 + p.step * p.stride1], scale);             \
    }                                                                        \
}                                                                            \
kernel void NAME##_generate(device TYPE* output [[buffer(0)]],               \
                            const device TYPE* packed [[buffer(1)]],         \
                            const device TYPE* tau [[buffer(2)]],             \
                            constant QRGenerateParams& p [[buffer(3)]],     \
                            uint gid [[thread_position_in_grid]]) {          \
    const ulong column = gid;                                                 \
    if (column >= p.rows || p.step >= p.columns || p.step >= p.rows) return; \
    TYPE dot = output[p.step * p.outputStride0 + column * p.outputStride1];  \
    for (ulong row = p.step + 1; row < p.rows; ++row) {                       \
        const TYPE v = packed[row * p.packedStride0 + p.step * p.packedStride1]; \
        dot += qrMultiply(qrConjugate(v),                                    \
                          output[row * p.outputStride0 + column * p.outputStride1]); \
    }                                                                        \
    const TYPE scale = qrMultiply(tau[p.step], dot);                         \
    output[p.step * p.outputStride0 + column * p.outputStride1] -= scale;    \
    for (ulong row = p.step + 1; row < p.rows; ++row) {                      \
        const ulong index = row * p.outputStride0 + column * p.outputStride1; \
        output[index] -= qrMultiply(                                         \
            packed[row * p.packedStride0 + p.step * p.packedStride1], scale); \
    }                                                                        \
}

DEFINE_QR(qr_float, float)
DEFINE_QR(qr_cfloat, float2)
