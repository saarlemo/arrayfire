/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/
#include <metal_stdlib>
using namespace metal;
struct LuParams{ulong odims[4],ostrides[4],idims[4],istrides[4];uint lower;};
#define DEFINE_LU(NAME,TYPE,ONE)                                           \
kernel void NAME(const device TYPE*in[[buffer(0)]],device TYPE*out[[buffer(1)]],\
 constant LuParams&p[[buffer(2)]],uint gid[[thread_position_in_grid]]){   \
 const ulong total=p.odims[0]*p.odims[1]*p.odims[2]*p.odims[3];if(gid>=total)return;\
 ulong q=gid;const ulong x=q%p.odims[0];q/=p.odims[0];const ulong y=q%p.odims[1];\
 q/=p.odims[1];const ulong z=q%p.odims[2],w=q/p.odims[2];                 \
 const ulong oo=x*p.ostrides[0]+y*p.ostrides[1]+z*p.ostrides[2]+w*p.ostrides[3];\
 const ulong ii=x*p.istrides[0]+y*p.istrides[1]+z*p.istrides[2]+w*p.istrides[3];\
 out[oo]=p.lower?(x>y?in[ii]:(x==y?ONE:TYPE(0))):(y>=x?in[ii]:TYPE(0));}
DEFINE_LU(lu_float,float,1.0f)
DEFINE_LU(lu_cfloat,float2,float2(1.0f,0.0f))
struct PivotParams{ulong outputCount,pivotCount;};
kernel void convert_pivot(const device int*initial[[buffer(0)]],
 const device int*pivot[[buffer(1)]],device int*out[[buffer(2)]],
 constant PivotParams&p[[buffer(3)]],uint gid[[thread_position_in_grid]]){
 if(gid)return;for(ulong i=0;i<p.outputCount;++i)out[i]=initial[i];
 for(ulong i=0;i<p.pivotCount;++i){const int j=pivot[i]-1;const int v=out[i];
 out[i]=out[j];out[j]=v;}}
