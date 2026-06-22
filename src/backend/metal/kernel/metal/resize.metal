#include <metal_stdlib>
using namespace metal;

struct ResizeParams {
    ulong outputDims[4], outputStrides[4], inputDims[4], inputStrides[4];
    uint method;
};

#define DEFINE_RESIZE(NAME, TYPE)                                           \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant ResizeParams& p [[buffer(2)]],                    \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total = p.outputDims[0] * p.outputDims[1] *                 \
                        p.outputDims[2] * p.outputDims[3];                   \
    if (gid >= total) return;                                               \
    ulong q = gid; const ulong x = q % p.outputDims[0];                     \
    q /= p.outputDims[0]; const ulong y = q % p.outputDims[1];              \
    q /= p.outputDims[1]; const ulong z = q % p.outputDims[2];              \
    const ulong w = q / p.outputDims[2];                                    \
    const float sx = float(x) / (float(p.outputDims[0]) /                   \
                                  float(p.inputDims[0]));                    \
    const float sy = float(y) / (float(p.outputDims[1]) /                   \
                                  float(p.inputDims[1]));                    \
    ulong x0 = p.method == 0 ? ulong(sx + .5f) : ulong(floor(sx));          \
    ulong y0 = p.method == 0 ? ulong(sy + .5f) : ulong(floor(sy));          \
    x0 = min(x0, p.inputDims[0] - 1);                                       \
    y0 = min(y0, p.inputDims[1] - 1);                                       \
    const ulong oo = x * p.outputStrides[0] + y * p.outputStrides[1] +      \
                     z * p.outputStrides[2] + w * p.outputStrides[3];        \
    const ulong base = z * p.inputStrides[2] + w * p.inputStrides[3];       \
    if (p.method != 2) {                                                    \
        output[oo] = input[base + x0 * p.inputStrides[0] +                  \
                           y0 * p.inputStrides[1]];                          \
        return;                                                             \
    }                                                                       \
    const ulong x1 = min(x0 + 1, p.inputDims[0] - 1);                       \
    const ulong y1 = min(y0 + 1, p.inputDims[1] - 1);                       \
    const float bx = sx - float(x0), ay = sy - float(y0);                   \
    const TYPE v00 = input[base + x0*p.inputStrides[0]+y0*p.inputStrides[1]];\
    const TYPE v01 = input[base + x0*p.inputStrides[0]+y1*p.inputStrides[1]];\
    const TYPE v10 = input[base + x1*p.inputStrides[0]+y0*p.inputStrides[1]];\
    const TYPE v11 = input[base + x1*p.inputStrides[0]+y1*p.inputStrides[1]];\
    output[oo] = TYPE((1-ay)*(1-bx)*v00 + ay*(1-bx)*v01 +                 \
                      (1-ay)*bx*v10 + ay*bx*v11);                            \
}

DEFINE_RESIZE(resize_float, float)
DEFINE_RESIZE(resize_cfloat, float2)
DEFINE_RESIZE(resize_int, int)
DEFINE_RESIZE(resize_uint, uint)
DEFINE_RESIZE(resize_long, long)
DEFINE_RESIZE(resize_ulong, ulong)
DEFINE_RESIZE(resize_char, char)
DEFINE_RESIZE(resize_uchar, uchar)
DEFINE_RESIZE(resize_short, short)
DEFINE_RESIZE(resize_ushort, ushort)
