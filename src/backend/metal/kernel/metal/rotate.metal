/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/
#include <metal_stdlib>
using namespace metal;
struct RotateParams{ulong odims[4],ostrides[4],idims[4],istrides[4];float theta;uint method;};
#define DEFINE_ROTATE(NAME,TYPE)                                           \
kernel void NAME(const device TYPE* input [[buffer(0)]],device TYPE* output [[buffer(1)]],constant RotateParams&p [[buffer(2)]],uint gid [[thread_position_in_grid]]){\
 const ulong total=p.odims[0]*p.odims[1]*p.odims[2]*p.odims[3];if(gid>=total)return;ulong q=gid;const ulong x=q%p.odims[0];q/=p.odims[0];const ulong y=q%p.odims[1];q/=p.odims[1];const ulong z=q%p.odims[2],w=q/p.odims[2];\
 const float c=cos(-p.theta),s=sin(-p.theta);const float nx=.5f*float(p.idims[0]-1),ny=.5f*float(p.idims[1]-1),mx=.5f*float(p.odims[0]-1),my=.5f*float(p.odims[1]-1);\
 const float tx=-(mx*c-my*s-nx),ty=-(mx*s+my*c-ny);const float rc=round(c*1000.0f)/1000.0f,rs=round(s*1000.0f)/1000.0f,rtx=round(tx*1000.0f)/1000.0f,rty=round(ty*1000.0f)/1000.0f;\
 const float fx=float(x)*rc+float(y)*(-rs)+rtx,fy=float(x)*rs+float(y)*rc+rty;const long ix=p.method==4?long(floor(fx)):long(round(fx));const long iy=p.method==4?long(floor(fy)):long(round(fy));\
 const ulong oo=x*p.ostrides[0]+y*p.ostrides[1]+z*p.ostrides[2]+w*p.ostrides[3];\
 if(ix>=0&&iy>=0&&ix<long(p.idims[0])&&iy<long(p.idims[1]))output[oo]=input[ulong(ix)*p.istrides[0]+ulong(iy)*p.istrides[1]+z*p.istrides[2]+w*p.istrides[3]];else output[oo]=TYPE(0);}
DEFINE_ROTATE(rotate_float,float)
DEFINE_ROTATE(rotate_cfloat,float2)
DEFINE_ROTATE(rotate_int,int)
DEFINE_ROTATE(rotate_uint,uint)
DEFINE_ROTATE(rotate_long,long)
DEFINE_ROTATE(rotate_ulong,ulong)
DEFINE_ROTATE(rotate_char,char)
DEFINE_ROTATE(rotate_uchar,uchar)
DEFINE_ROTATE(rotate_short,short)
DEFINE_ROTATE(rotate_ushort,ushort)
