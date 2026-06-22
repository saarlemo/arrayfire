/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/
#include <metal_stdlib>
using namespace metal;
struct MatchTemplateParams { ulong dims[4],outputStrides[4],searchStrides[4],templateDims[4],templateStrides[4]; uint matchType; };
#define DEFINE_MATCH(NAME,TYPE)                                            \
kernel void NAME(const device TYPE* search [[buffer(0)]],                  \
 const device TYPE* templ [[buffer(1)]],device float* output [[buffer(2)]],\
 constant MatchTemplateParams& p [[buffer(3)]],uint gid [[thread_position_in_grid]]) {\
 const ulong total=p.dims[0]*p.dims[1]*p.dims[2]*p.dims[3]; if(gid>=total)return;\
 ulong q=gid;const ulong x=q%p.dims[0];q/=p.dims[0];const ulong y=q%p.dims[1];q/=p.dims[1];const ulong z=q%p.dims[2],w=q/p.dims[2];\
 const ulong count=p.templateDims[0]*p.templateDims[1];float tm=0.0f,wm=0.0f;\
 const bool mean=p.matchType==1||p.matchType==2||p.matchType==4||p.matchType==5;\
 if(mean)for(ulong j=0;j<p.templateDims[1];++j)for(ulong i=0;i<p.templateDims[0];++i){\
  tm+=float(templ[i*p.templateStrides[0]+j*p.templateStrides[1]]);const ulong sx=x+i,sy=y+j;\
  if(sx<p.dims[0]&&sy<p.dims[1])wm+=float(search[sx*p.searchStrides[0]+sy*p.searchStrides[1]+z*p.searchStrides[2]+w*p.searchStrides[3]]);}\
 if(mean){tm/=float(count);wm/=float(count);}float result=0.0f;\
 for(ulong j=0;j<p.templateDims[1];++j)for(ulong i=0;i<p.templateDims[0];++i){\
  const ulong sx=x+i,sy=y+j;const float sv=(sx<p.dims[0]&&sy<p.dims[1])?float(search[sx*p.searchStrides[0]+sy*p.searchStrides[1]+z*p.searchStrides[2]+w*p.searchStrides[3]]):0.0f;\
  const float tv=float(templ[i*p.templateStrides[0]+j*p.templateStrides[1]]);float d;\
  if(p.matchType==0)d=sv-tv;else if(p.matchType==1)d=sv-wm-tv+tm;else if(p.matchType==2)d=sv-(wm/tm)*tv;else if(p.matchType==3)d=sv-tv;else if(p.matchType==4)d=sv-wm-tv+tm;else d=sv-(wm/tm)*tv;\
  result+=(p.matchType<=2)?abs(d):d*d;}\
 output[x*p.outputStrides[0]+y*p.outputStrides[1]+z*p.outputStrides[2]+w*p.outputStrides[3]]=result; }
DEFINE_MATCH(match_float,float)
DEFINE_MATCH(match_int,int)
DEFINE_MATCH(match_uint,uint)
DEFINE_MATCH(match_char,char)
DEFINE_MATCH(match_uchar,uchar)
DEFINE_MATCH(match_short,short)
DEFINE_MATCH(match_ushort,ushort)
