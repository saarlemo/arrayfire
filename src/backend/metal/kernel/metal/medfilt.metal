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

struct MedfiltParams {
    ulong dims[4], outputStrides[4], inputStrides[4];
    uint windowLength, windowWidth, padding, oneDimensional;
};

#define DEFINE_MEDFILT(NAME, TYPE)                                         \
kernel void NAME(const device TYPE* input [[buffer(0)]],                   \
                 device TYPE* output [[buffer(1)]],                        \
                 constant MedfiltParams& p [[buffer(2)]],                  \
                 uint gid [[thread_position_in_grid]]) {                   \
    const ulong total=p.dims[0]*p.dims[1]*p.dims[2]*p.dims[3];            \
    if (gid>=total) return;                                                \
    ulong q=gid; const long x=long(q%p.dims[0]); q/=p.dims[0];            \
    const long y=long(q%p.dims[1]); q/=p.dims[1];                         \
    const ulong z=q%p.dims[2],w=q/p.dims[2];                              \
    TYPE values[225]; uint count=0;                                       \
    const uint height=p.oneDimensional ? 1 : p.windowWidth;               \
    for (uint j=0;j<height;++j)                                           \
      for (uint i=0;i<p.windowLength;++i) {                               \
        long ix=x+long(i)-long(p.windowLength/2);                         \
        long iy=y+long(j)-long(height/2);                                 \
        bool outside=ix<0||iy<0||ix>=long(p.dims[0])||iy>=long(p.dims[1]);\
        if (outside&&p.padding==0) values[count++]=TYPE(0);                \
        else {                                                            \
          if (ix<0) ix=-ix; if (iy<0) iy=-iy;                            \
          if (ix>=long(p.dims[0])) ix=2*(long(p.dims[0])-1)-ix;           \
          if (iy>=long(p.dims[1])) iy=2*(long(p.dims[1])-1)-iy;           \
          const ulong ii=ulong(ix)*p.inputStrides[0]+                     \
            ulong(iy)*p.inputStrides[1]+z*p.inputStrides[2]+             \
            w*p.inputStrides[3]; values[count++]=input[ii];              \
        }                                                                 \
      }                                                                   \
    for (uint i=1;i<count;++i) { TYPE key=values[i]; uint j=i;            \
      while (j>0&&key<values[j-1]) { values[j]=values[j-1];--j; }        \
      values[j]=key; }                                                    \
    const uint mid=count/2; TYPE result=values[mid];                       \
    if ((count&1u)==0) result=TYPE((values[mid]+values[mid-1])/2);         \
    const ulong oo=ulong(x)*p.outputStrides[0]+ulong(y)*p.outputStrides[1]+\
      z*p.outputStrides[2]+w*p.outputStrides[3]; output[oo]=result;       \
}

DEFINE_MEDFILT(medfilt_float, float)
DEFINE_MEDFILT(medfilt_int, int)
DEFINE_MEDFILT(medfilt_uint, uint)
DEFINE_MEDFILT(medfilt_char, char)
DEFINE_MEDFILT(medfilt_uchar, uchar)
DEFINE_MEDFILT(medfilt_short, short)
DEFINE_MEDFILT(medfilt_ushort, ushort)
