#include <metal_stdlib>
using namespace metal;

struct CopyParams {
    ulong dims[4], outputStrides[4], inputStrides[4];
};

#define DEFINE_COPY(NAME, TYPE)                                             \
kernel void NAME(const device TYPE* input [[buffer(0)]],                    \
                 device TYPE* output [[buffer(1)]],                         \
                 constant CopyParams& p [[buffer(2)]],                      \
                 uint gid [[thread_position_in_grid]]) {                    \
    const ulong total=p.dims[0]*p.dims[1]*p.dims[2]*p.dims[3];             \
    if (gid>=total) return;                                                 \
    ulong q=gid; const ulong x=q%p.dims[0]; q/=p.dims[0];                  \
    const ulong y=q%p.dims[1]; q/=p.dims[1];                              \
    const ulong z=q%p.dims[2]; const ulong w=q/p.dims[2];                  \
    const ulong oo=x*p.outputStrides[0]+y*p.outputStrides[1]+              \
                   z*p.outputStrides[2]+w*p.outputStrides[3];              \
    const ulong ii=x*p.inputStrides[0]+y*p.inputStrides[1]+                \
                   z*p.inputStrides[2]+w*p.inputStrides[3];                \
    output[oo]=input[ii];                                                   \
}

DEFINE_COPY(copy_float, float)
DEFINE_COPY(copy_cfloat, float2)
DEFINE_COPY(copy_int, int)
DEFINE_COPY(copy_uint, uint)
DEFINE_COPY(copy_long, long)
DEFINE_COPY(copy_ulong, ulong)
DEFINE_COPY(copy_char, char)
DEFINE_COPY(copy_uchar, uchar)
DEFINE_COPY(copy_short, short)
DEFINE_COPY(copy_ushort, ushort)
DEFINE_COPY(copy_half, half)
