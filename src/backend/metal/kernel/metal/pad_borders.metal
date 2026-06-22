#include <metal_stdlib>
using namespace metal;

struct PadParams {
    ulong outputDims[4], outputStrides[4], inputDims[4], inputStrides[4];
    long lower[4];
    uint borderType;
};

long mapIndex(long x, ulong length, uint type) {
    const long n = long(length);
    if (x >= 0 && x < n) return x;
    if (type == 1) return x < 0 ? (-x - 1) % n : n - (x % n) - 1;
    if (type == 2) return clamp(x, long(0), n - 1);
    if (type == 3) { const long r = x % n; return r < 0 ? r + n : r; }
    return -1;
}

#define DEFINE_PAD(NAME, TYPE, ZERO)                                        \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant PadParams& p [[buffer(2)]],                       \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total = p.outputDims[0]*p.outputDims[1]*                    \
                        p.outputDims[2]*p.outputDims[3];                     \
    if (gid >= total) return;                                               \
    ulong q=gid; const ulong x=q%p.outputDims[0]; q/=p.outputDims[0];       \
    const ulong y=q%p.outputDims[1]; q/=p.outputDims[1];                    \
    const ulong z=q%p.outputDims[2]; const ulong w=q/p.outputDims[2];       \
    const ulong c[4]={x,y,z,w}; ulong oo=0, ii=0;                           \
    for (uint d=0; d<4; ++d) {                                             \
        oo += c[d]*p.outputStrides[d];                                      \
        const long ix=mapIndex(long(c[d])-p.lower[d],p.inputDims[d],        \
                               p.borderType);                               \
        if (ix < 0) { output[oo]=ZERO; return; }                            \
        ii += ulong(ix)*p.inputStrides[d];                                  \
    }                                                                       \
    output[oo]=input[ii];                                                   \
}

DEFINE_PAD(pad_float, float, float(0))
DEFINE_PAD(pad_cfloat, float2, float2(0))
DEFINE_PAD(pad_int, int, int(0))
DEFINE_PAD(pad_uint, uint, uint(0))
DEFINE_PAD(pad_long, long, long(0))
DEFINE_PAD(pad_ulong, ulong, ulong(0))
DEFINE_PAD(pad_char, char, char(0))
DEFINE_PAD(pad_uchar, uchar, uchar(0))
DEFINE_PAD(pad_short, short, short(0))
DEFINE_PAD(pad_ushort, ushort, ushort(0))
