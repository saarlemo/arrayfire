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

struct LuFactorParams {
    ulong rows;
    ulong columns;
    ulong stride0;
    ulong stride1;
    uint step;
};

float luMagnitude(const float value) { return fabs(value); }
float luMagnitude(const float2 value) { return length(value); }

float luMagnitudeSquared(const float value) { return value * value; }
float luMagnitudeSquared(const float2 value) { return dot(value, value); }

float luRealDivide(const float lhs, const float rhs) { return lhs / rhs; }
float2 luRealDivide(const float2 lhs, const float2 rhs) {
    const float denominator = dot(rhs, rhs);
    return float2(lhs.x * rhs.x + lhs.y * rhs.y,
                  lhs.y * rhs.x - lhs.x * rhs.y) /
           denominator;
}

float luMultiply(const float lhs, const float rhs) { return lhs * rhs; }
float2 luMultiply(const float2 lhs, const float2 rhs) {
    return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                  lhs.x * rhs.y + lhs.y * rhs.x);
}

#define DEFINE_LU_FACTOR(NAME, TYPE)                                      \
kernel void NAME##_pivot(device TYPE* input [[buffer(0)]],                \
                         device int* pivot [[buffer(1)]],                \
                         constant LuFactorParams& p [[buffer(2)]],        \
                         uint gid [[thread_position_in_grid]]) {         \
    if (gid != 0 || p.step >= p.rows || p.step >= p.columns) return;      \
    ulong best = p.step;                                                   \
    float bestValue = luMagnitudeSquared(                                 \
        input[p.step * p.stride1 + p.step * p.stride0]);                   \
    for (ulong row = p.step + 1; row < p.rows; ++row) {                    \
        const float value = luMagnitudeSquared(                           \
            input[row * p.stride0 + p.step * p.stride1]);                  \
        if (value > bestValue) {                                           \
            best = row;                                                     \
            bestValue = value;                                              \
        }                                                                    \
    }                                                                        \
    pivot[p.step] = int(best + 1);                                          \
}                                                                            \
kernel void NAME##_swap(device TYPE* input [[buffer(0)]],                 \
                        device int* pivot [[buffer(1)]],                 \
                        constant LuFactorParams& p [[buffer(2)]],         \
                        uint gid [[thread_position_in_grid]]) {          \
    if (gid >= p.columns || p.step >= p.rows || p.step >= p.columns) return; \
    const ulong row = ulong(max(pivot[p.step] - 1, 0));                    \
    if (row == p.step) return;                                              \
    const ulong a = p.step * p.stride0 + ulong(gid) * p.stride1;            \
    const ulong b = row * p.stride0 + ulong(gid) * p.stride1;               \
    const TYPE value = input[a];                                            \
    input[a] = input[b];                                                     \
    input[b] = value;                                                        \
}                                                                            \
kernel void NAME##_update(device TYPE* input [[buffer(0)]],               \
                          device int* pivot [[buffer(1)]],                \
                          constant LuFactorParams& p [[buffer(2)]],       \
                          uint2 gid [[thread_position_in_grid]]) {        \
    if (gid.x <= p.step || gid.x >= p.columns || gid.y <= p.step ||         \
        gid.y >= p.rows) return;                                             \
    const ulong row = ulong(gid.y);                                          \
    const ulong column = ulong(gid.x);                                       \
    const TYPE factor = luRealDivide(                                        \
        input[row * p.stride0 + p.step * p.stride1],                         \
        input[p.step * p.stride0 + p.step * p.stride1]);                     \
    const TYPE product = luMultiply(                                         \
        factor, input[p.step * p.stride0 + column * p.stride1]);             \
    input[row * p.stride0 + column * p.stride1] -= product;                  \
}                                                                            \
kernel void NAME##_factor(device TYPE* input [[buffer(0)]],               \
                          device int* pivot [[buffer(1)]],                \
                          constant LuFactorParams& p [[buffer(2)]],       \
                          uint gid [[thread_position_in_grid]]) {         \
    if (gid <= p.step || gid >= p.rows || p.step >= p.columns) return;      \
    const ulong index = ulong(gid) * p.stride0 + p.step * p.stride1;         \
    input[index] = luRealDivide(                                             \
        input[index], input[p.step * p.stride0 + p.step * p.stride1]);      \
}

DEFINE_LU_FACTOR(lu_factor_float, float)
DEFINE_LU_FACTOR(lu_factor_cfloat, float2)

struct PivotParams{ulong outputCount,pivotCount;};
kernel void convert_pivot(const device int*initial[[buffer(0)]],
 const device int*pivot[[buffer(1)]],device int*out[[buffer(2)]],
 constant PivotParams&p[[buffer(3)]],uint gid[[thread_position_in_grid]]){
 if(gid)return;for(ulong i=0;i<p.outputCount;++i)out[i]=initial[i];
 for(ulong i=0;i<p.pivotCount;++i){const int j=pivot[i]-1;const int v=out[i];
 out[i]=out[j];out[j]=v;}}
