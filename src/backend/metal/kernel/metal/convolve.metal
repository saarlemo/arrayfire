/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement for ArrayFire can be found at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <metal_stdlib>
using namespace metal;

struct ConvolveParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong signalDims[4];
    ulong signalStrides[4];
    ulong filterDims[4];
    ulong filterStrides[4];
    uint rank;
    uint expand;
    uint batchKind;
};

struct SeparableConvolveParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong signalDims[4];
    ulong signalStrides[4];
    ulong filterDims[4];
    ulong filterStrides[4];
    uint convDim;
    uint expand;
};

struct ConvolveNNParams {
    ulong outputDims[4];
    ulong outputStrides[4];
    ulong signalDims[4];
    ulong signalStrides[4];
    ulong filterDims[4];
    ulong filterStrides[4];
    ulong gradientDims[4];
    ulong gradientStrides[4];
    ulong stride[2];
    long padding[2];
    ulong dilation[2];
};

template<typename T>
inline T convMultiply(const T lhs, const T rhs) {
    return lhs * rhs;
}

inline float2 convMultiply(const float2 lhs, const float2 rhs) {
    return float2(lhs.x * rhs.x - lhs.y * rhs.y,
                  lhs.x * rhs.y + lhs.y * rhs.x);
}

template<typename InT, typename AccT>
inline AccT convProduct(const InT signal, const AccT filter) {
    return convMultiply(AccT(signal), AccT(filter));
}

template<typename InT, typename AccT>
inline InT convStore(const AccT value) {
    return InT(value);
}

template<typename InT, typename AccT>
inline void convolveImpl(const device InT* signal,
                         const device AccT* filter, device InT* output,
                         constant ConvolveParams& p, const uint gid) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;

    ulong q = gid;
    ulong c[4];
    for (uint d = 0; d < 4; ++d) {
        c[d] = q % p.outputDims[d];
        q /= p.outputDims[d];
    }

    const bool signalBatched = p.batchKind == 1 || p.batchKind == 3;
    const bool filterBatched = p.batchKind == 2 || p.batchKind == 3;
    ulong signalBase = 0;
    ulong filterBase = 0;
    for (uint d = 3; d >= p.rank && d < 4; --d) {
        if (signalBatched) signalBase += c[d] * p.signalStrides[d];
        if (filterBatched) filterBase += c[d] * p.filterStrides[d];
    }

    const long sx0 = long(c[0]) +
                     (p.expand ? 0 : long(p.filterDims[0] / 2));
    const long sx1 = long(c[1]) +
                     (p.expand ? 0 : long(p.filterDims[1] / 2));
    const long sx2 = long(c[2]) +
                     (p.expand ? 0 : long(p.filterDims[2] / 2));
    AccT sum = AccT(0);
    const ulong f2Count = p.rank > 2 ? p.filterDims[2] : 1;
    const ulong f1Count = p.rank > 1 ? p.filterDims[1] : 1;
    for (ulong f2 = 0; f2 < f2Count; ++f2) {
        const long z = sx2 - long(f2);
        if (p.rank > 2 && (z < 0 || z >= long(p.signalDims[2]))) continue;
        for (ulong f1 = 0; f1 < f1Count; ++f1) {
            const long y = sx1 - long(f1);
            if (p.rank > 1 && (y < 0 || y >= long(p.signalDims[1]))) continue;
            for (ulong f0 = 0; f0 < p.filterDims[0]; ++f0) {
                const long x = sx0 - long(f0);
                if (x < 0 || x >= long(p.signalDims[0])) continue;

                ulong signalIndex = signalBase +
                                    ulong(x) * p.signalStrides[0];
                if (p.rank > 1) signalIndex += ulong(y) * p.signalStrides[1];
                if (p.rank > 2) signalIndex += ulong(z) * p.signalStrides[2];
                const ulong filterIndex =
                    filterBase + f0 * p.filterStrides[0] +
                    f1 * p.filterStrides[1] + f2 * p.filterStrides[2];
                sum += convProduct(signal[signalIndex], filter[filterIndex]);
            }
        }
    }

    ulong outputIndex = 0;
    for (uint d = 0; d < 4; ++d) {
        outputIndex += c[d] * p.outputStrides[d];
    }
    output[outputIndex] = convStore<InT, AccT>(sum);
}

// Experimental regular rank-2 path for large square filters. Each group
// stages the complete signal halo used by an 8x8 output tile once. The
// validity plane preserves the scalar kernel's OOB-tap skip semantics,
// including when a filter coefficient is NaN or infinite.
template <uint FilterSize>
inline void convolveTiled2DImpl(
    const device float* signal, const device float* filter,
    device float* output, constant ConvolveParams& p,
    threadgroup float* signalTile, threadgroup uchar* validTile,
    threadgroup float* filterTile, const uint2 localID,
    const uint2 groupID) {
    constexpr uint tileOutputWidth  = 8;
    constexpr uint tileOutputHeight = 8;
    constexpr uint tileWidth = tileOutputWidth + FilterSize - 1;
    constexpr uint tileHeight = tileOutputHeight + FilterSize - 1;
    constexpr uint tileElements = tileWidth * tileHeight;
    constexpr uint filterElements = FilterSize * FilterSize;

    const long anchorX = p.expand ? 0 : long(FilterSize / 2);
    const long anchorY = p.expand ? 0 : long(FilterSize / 2);
    const long tileOriginX = long(groupID.x * tileOutputWidth) + anchorX -
                             long(FilterSize - 1);
    const long tileOriginY = long(groupID.y * tileOutputHeight) + anchorY -
                             long(FilterSize - 1);
    const uint threadLinear = localID.y * tileOutputWidth + localID.x;
    const uint threadCount = tileOutputWidth * tileOutputHeight;

    for (uint i = threadLinear; i < tileElements; i += threadCount) {
        const uint tx = i % tileWidth;
        const uint ty = i / tileWidth;
        const long sx = tileOriginX + long(tx);
        const long sy = tileOriginY + long(ty);
        const bool inBounds = sx >= 0 && sy >= 0 &&
                              sx < long(p.signalDims[0]) &&
                              sy < long(p.signalDims[1]);
        validTile[i] = inBounds ? 1 : 0;
        signalTile[i] = inBounds
                            ? signal[ulong(sx) * p.signalStrides[0] +
                                     ulong(sy) * p.signalStrides[1]]
                            : 0.0f;
    }

    for (uint i = threadLinear; i < filterElements; i += threadCount) {
        const uint fx = i % FilterSize;
        const uint fy = i / FilterSize;
        filterTile[i] = filter[ulong(fx) * p.filterStrides[0] +
                               ulong(fy) * p.filterStrides[1]];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong ox = ulong(groupID.x * tileOutputWidth + localID.x);
    const ulong oy = ulong(groupID.y * tileOutputHeight + localID.y);
    if (ox >= p.outputDims[0] || oy >= p.outputDims[1]) return;

    float sum = 0.0f;
    for (uint fy = 0; fy < FilterSize; ++fy) {
        const uint tileY = localID.y + FilterSize - 1 - fy;
        for (uint fx = 0; fx < FilterSize; ++fx) {
            const uint tileX = localID.x + FilterSize - 1 - fx;
            const uint tileIndex = tileY * tileWidth + tileX;
            if (validTile[tileIndex]) {
                sum += signalTile[tileIndex] * filterTile[fy * FilterSize + fx];
            }
        }
    }

    output[ox * p.outputStrides[0] + oy * p.outputStrides[1]] = sum;
}

kernel void convolve_float_tiled_7(
    const device float* signal [[buffer(0)]],
    const device float* filter [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ConvolveParams& p [[buffer(3)]],
    uint2 localID [[thread_position_in_threadgroup]],
    uint2 groupID [[threadgroup_position_in_grid]]) {
    threadgroup float signalTile[22 * 22];
    threadgroup uchar validTile[22 * 22];
    threadgroup float filterTile[15 * 15];
    convolveTiled2DImpl<7>(signal, filter, output, p, signalTile, validTile,
                           filterTile, localID, groupID);
}

kernel void convolve_float_tiled_15(
    const device float* signal [[buffer(0)]],
    const device float* filter [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ConvolveParams& p [[buffer(3)]],
    uint2 localID [[thread_position_in_threadgroup]],
    uint2 groupID [[threadgroup_position_in_grid]]) {
    threadgroup float signalTile[22 * 22];
    threadgroup uchar validTile[22 * 22];
    threadgroup float filterTile[15 * 15];
    convolveTiled2DImpl<15>(signal, filter, output, p, signalTile, validTile,
                            filterTile, localID, groupID);
}

template<typename InT, typename AccT>
inline void separableConvolveImpl(const device InT* signal,
                                  const device AccT* filter,
                                  device InT* output,
                                  constant SeparableConvolveParams& p,
                                  const uint gid) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;

    ulong q = gid;
    ulong c[4];
    for (uint d = 0; d < 4; ++d) {
        c[d] = q % p.outputDims[d];
        q /= p.outputDims[d];
    }
    const long center = p.expand ? 0 :
        long(p.filterDims[0] / 2);
    const long x = long(c[0]) + (p.convDim == 0 ? center : 0);
    const long y = long(c[1]) + (p.convDim == 1 ? center : 0);
    AccT sum = AccT(0);
    for (ulong f = 0; f < p.filterDims[0]; ++f) {
        const long sx = p.convDim == 0 ? x - long(f) : x;
        const long sy = p.convDim == 1 ? y - long(f) : y;
        if (sx < 0 || sx >= long(p.signalDims[0]) || sy < 0 ||
            sy >= long(p.signalDims[1])) {
            continue;
        }
        ulong signalIndex = ulong(sx) * p.signalStrides[0] +
                            ulong(sy) * p.signalStrides[1];
        signalIndex += c[2] * p.signalStrides[2] +
                       c[3] * p.signalStrides[3];
        const ulong filterIndex = f * p.filterStrides[0];
        sum += convProduct(signal[signalIndex], filter[filterIndex]);
    }

    ulong outputIndex = c[0] * p.outputStrides[0] +
                        c[1] * p.outputStrides[1] +
                        c[2] * p.outputStrides[2] +
                        c[3] * p.outputStrides[3];
    output[outputIndex] = convStore<InT, AccT>(sum);
}

template<typename InT, typename AccT>
inline void convolveNNImpl(const device InT* signal,
                           const device InT* filter, device InT* output,
                           constant ConvolveNNParams& p, const uint gid) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;

    ulong q = gid;
    ulong c[4];
    for (uint d = 0; d < 4; ++d) {
        c[d] = q % p.outputDims[d];
        q /= p.outputDims[d];
    }

    AccT sum = AccT(0);
    for (ulong ic = 0; ic < p.filterDims[2]; ++ic) {
        for (ulong fy = 0; fy < p.filterDims[1]; ++fy) {
            for (ulong fx = 0; fx < p.filterDims[0]; ++fx) {
                const long x = long(c[0] * p.stride[0]) +
                               long((p.filterDims[0] - 1 - fx) *
                                    p.dilation[0]) -
                               p.padding[0];
                const long y = long(c[1] * p.stride[1]) +
                               long((p.filterDims[1] - 1 - fy) *
                                    p.dilation[1]) -
                               p.padding[1];
                if (x < 0 || x >= long(p.signalDims[0]) || y < 0 ||
                    y >= long(p.signalDims[1])) {
                    continue;
                }

                const ulong signalIndex =
                    ulong(x) * p.signalStrides[0] +
                    ulong(y) * p.signalStrides[1] +
                    ic * p.signalStrides[2] + c[3] * p.signalStrides[3];
                const ulong filterIndex =
                    fx * p.filterStrides[0] + fy * p.filterStrides[1] +
                    ic * p.filterStrides[2] + c[2] * p.filterStrides[3];
                sum += convProduct(signal[signalIndex], filter[filterIndex]);
            }
        }
    }

    const ulong outputIndex = c[0] * p.outputStrides[0] +
                               c[1] * p.outputStrides[1] +
                               c[2] * p.outputStrides[2] +
                               c[3] * p.outputStrides[3];
    output[outputIndex] = convStore<InT, AccT>(sum);
}

template<typename T>
inline void convolveNNDataGradientImpl(
    const device T* incomingGradient, const device T* filter,
    device T* output, constant ConvolveNNParams& p, const uint gid) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;

    ulong q = gid;
    ulong c[4];
    for (uint d = 0; d < 4; ++d) {
        c[d] = q % p.outputDims[d];
        q /= p.outputDims[d];
    }

    float sum = 0.0f;
    for (ulong oc = 0; oc < p.gradientDims[2]; ++oc) {
        for (ulong fy = 0; fy < p.filterDims[1]; ++fy) {
            for (ulong fx = 0; fx < p.filterDims[0]; ++fx) {
                const long xNumerator =
                    long(c[0]) + p.padding[0] -
                    long((p.filterDims[0] - 1 - fx) * p.dilation[0]);
                const long yNumerator =
                    long(c[1]) + p.padding[1] -
                    long((p.filterDims[1] - 1 - fy) * p.dilation[1]);
                if (xNumerator < 0 || yNumerator < 0 ||
                    xNumerator % long(p.stride[0]) != 0 ||
                    yNumerator % long(p.stride[1]) != 0) {
                    continue;
                }
                const long ox = xNumerator / long(p.stride[0]);
                const long oy = yNumerator / long(p.stride[1]);
                if (ox >= long(p.signalDims[0]) ||
                    oy >= long(p.signalDims[1])) {
                    continue;
                }

                const ulong gradientIndex =
                    ulong(ox) * p.gradientStrides[0] +
                    ulong(oy) * p.gradientStrides[1] +
                    oc * p.gradientStrides[2] + c[3] * p.gradientStrides[3];
                const ulong filterIndex =
                    fx * p.filterStrides[0] + fy * p.filterStrides[1] +
                    c[2] * p.filterStrides[2] + oc * p.filterStrides[3];
                sum += float(incomingGradient[gradientIndex]) *
                       float(filter[filterIndex]);
            }
        }
    }

    const ulong outputIndex = c[0] * p.outputStrides[0] +
                               c[1] * p.outputStrides[1] +
                               c[2] * p.outputStrides[2] +
                               c[3] * p.outputStrides[3];
    output[outputIndex] = T(sum);
}

template<typename T>
inline void convolveNNFilterGradientImpl(
    const device T* signal, const device T* incomingGradient,
    device T* output, constant ConvolveNNParams& p, const uint gid) {
    const ulong total = p.outputDims[0] * p.outputDims[1] *
                        p.outputDims[2] * p.outputDims[3];
    if (gid >= total) return;

    ulong q = gid;
    ulong c[4];
    for (uint d = 0; d < 4; ++d) {
        c[d] = q % p.outputDims[d];
        q /= p.outputDims[d];
    }

    float sum = 0.0f;
    for (ulong batch = 0; batch < p.signalDims[3]; ++batch) {
        for (ulong oy = 0; oy < p.gradientDims[1]; ++oy) {
            for (ulong ox = 0; ox < p.gradientDims[0]; ++ox) {
                const long x = long(ox * p.stride[0]) +
                               long((p.filterDims[0] - 1 - c[0]) *
                                    p.dilation[0]) -
                               p.padding[0];
                const long y = long(oy * p.stride[1]) +
                               long((p.filterDims[1] - 1 - c[1]) *
                                    p.dilation[1]) -
                               p.padding[1];
                if (x < 0 || x >= long(p.signalDims[0]) || y < 0 ||
                    y >= long(p.signalDims[1])) {
                    continue;
                }

                const ulong signalIndex =
                    ulong(x) * p.signalStrides[0] +
                    ulong(y) * p.signalStrides[1] +
                    c[2] * p.signalStrides[2] + batch * p.signalStrides[3];
                const ulong gradientIndex =
                    ox * p.gradientStrides[0] + oy * p.gradientStrides[1] +
                    c[3] * p.gradientStrides[2] +
                    batch * p.gradientStrides[3];
                sum += float(signal[signalIndex]) *
                       float(incomingGradient[gradientIndex]);
            }
        }
    }

    const ulong outputIndex = c[0] * p.outputStrides[0] +
                               c[1] * p.outputStrides[1] +
                               c[2] * p.outputStrides[2] +
                               c[3] * p.outputStrides[3];
    output[outputIndex] = T(sum);
}

#define DEFINE_CONVOLVE_KERNEL(NAME, IN_T, ACC_T)                           \
kernel void NAME(const device IN_T* signal [[buffer(0)]],                  \
                 const device ACC_T* filter [[buffer(1)]],                 \
                 device IN_T* output [[buffer(2)]],                        \
                 constant ConvolveParams& p [[buffer(3)]],                 \
                 uint gid [[thread_position_in_grid]]) {                   \
    convolveImpl<IN_T, ACC_T>(signal, filter, output, p, gid);              \
}                                                                            \
kernel void separable_##NAME(const device IN_T* signal [[buffer(0)]],       \
                             const device ACC_T* filter [[buffer(1)]],      \
                             device IN_T* output [[buffer(2)]],              \
                             constant SeparableConvolveParams& p            \
                                 [[buffer(3)]],                              \
                             uint gid [[thread_position_in_grid]]) {        \
    separableConvolveImpl<IN_T, ACC_T>(signal, filter, output, p, gid);     \
}

#define DEFINE_CONVOLVE_NN_KERNEL(NAME, TYPE)                                \
kernel void NAME(const device TYPE* signal [[buffer(0)]],                   \
                 const device TYPE* filter [[buffer(1)]],                   \
                 device TYPE* output [[buffer(2)]],                          \
                 constant ConvolveNNParams& p [[buffer(3)]],                 \
                 uint gid [[thread_position_in_grid]]) {                    \
    convolveNNImpl<TYPE, float>(signal, filter, output, p, gid);             \
}                                                                            \
kernel void NAME##_data_gradient(                                             \
    const device TYPE* incomingGradient [[buffer(0)]],                       \
    const device TYPE* filter [[buffer(1)]], device TYPE* output [[buffer(2)]],\
    constant ConvolveNNParams& p [[buffer(3)]],                               \
    uint gid [[thread_position_in_grid]]) {                                   \
    convolveNNDataGradientImpl(incomingGradient, filter, output, p, gid);    \
}                                                                            \
kernel void NAME##_filter_gradient(                                           \
    const device TYPE* signal [[buffer(0)]],                                  \
    const device TYPE* incomingGradient [[buffer(1)]],                        \
    device TYPE* output [[buffer(2)]],                                        \
    constant ConvolveNNParams& p [[buffer(3)]],                               \
    uint gid [[thread_position_in_grid]]) {                                   \
    convolveNNFilterGradientImpl(signal, incomingGradient, output, p, gid);  \
}

DEFINE_CONVOLVE_NN_KERNEL(convolve_nn_float, float)
DEFINE_CONVOLVE_NN_KERNEL(convolve_nn_half, half)

#undef DEFINE_CONVOLVE_NN_KERNEL

DEFINE_CONVOLVE_KERNEL(convolve_float, float, float)
DEFINE_CONVOLVE_KERNEL(convolve_cfloat, float2, float2)
DEFINE_CONVOLVE_KERNEL(convolve_int_float, int, float)
DEFINE_CONVOLVE_KERNEL(convolve_uint_float, uint, float)
DEFINE_CONVOLVE_KERNEL(convolve_long_float, long, float)
DEFINE_CONVOLVE_KERNEL(convolve_ulong_float, ulong, float)
DEFINE_CONVOLVE_KERNEL(convolve_char_float, char, float)
DEFINE_CONVOLVE_KERNEL(convolve_uchar_float, uchar, float)
DEFINE_CONVOLVE_KERNEL(convolve_short_float, short, float)
DEFINE_CONVOLVE_KERNEL(convolve_ushort_float, ushort, float)

#undef DEFINE_CONVOLVE_KERNEL
