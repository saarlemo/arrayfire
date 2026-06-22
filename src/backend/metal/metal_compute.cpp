/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Metal.hpp>

#include <err_metal.hpp>
#include <metal_compute.hpp>
#include <metal_compute_anisotropic_diffusion.hpp>
#include <metal_compute_approx.hpp>
#include <metal_compute_array.hpp>
#include <metal_compute_assign.hpp>
#include <metal_compute_bilateral.hpp>
#include <metal_compute_canny.hpp>
#include <metal_compute_convolve.hpp>
#include <metal_compute_dot.hpp>
#include <metal_compute_example_function.hpp>
#include <metal_compute_fast.hpp>
#include <metal_compute_fftconvolve.hpp>
#include <metal_compute_flood_fill.hpp>
#include <metal_compute_harris.hpp>
#include <metal_compute_histogram.hpp>
#include <metal_compute_iir.hpp>
#include <metal_compute_index.hpp>
#include <metal_compute_ireduce.hpp>
#include <metal_compute_lu.hpp>
#include <metal_compute_match_template.hpp>
#include <metal_compute_mean.hpp>
#include <metal_compute_meanshift.hpp>
#include <metal_compute_medfilt.hpp>
#include <metal_compute_morph.hpp>
#include <metal_compute_nearest_neighbour.hpp>
#include <metal_compute_orb.hpp>
#include <metal_compute_random.hpp>
#include <metal_compute_reduce.hpp>
#include <metal_compute_regions.hpp>
#include <metal_compute_rotate.hpp>
#include <metal_compute_scan.hpp>
#include <metal_compute_scan_by_key.hpp>
#include <metal_compute_sift.hpp>
#include <metal_compute_sort.hpp>
#include <metal_compute_sort_by_key.hpp>
#include <metal_compute_sparse.hpp>
#include <metal_compute_sparse_arith.hpp>
#include <metal_compute_susan.hpp>
#include <metal_compute_transform.hpp>
#include <metal_kernel_headers/Array.hpp>
#include <metal_kernel_headers/anisotropic_diffusion.hpp>
#include <metal_kernel_headers/approx.hpp>
#include <metal_kernel_headers/assign.hpp>
#include <metal_kernel_headers/bilateral.hpp>
#include <metal_kernel_headers/canny.hpp>
#include <metal_kernel_headers/convolve.hpp>
#include <metal_kernel_headers/copy.hpp>
#include <metal_kernel_headers/diagonal.hpp>
#include <metal_kernel_headers/diff.hpp>
#include <metal_kernel_headers/dot.hpp>
#include <metal_kernel_headers/exampleFunction.hpp>
#include <metal_kernel_headers/fast.hpp>
#include <metal_kernel_headers/fftconvolve.hpp>
#include <metal_kernel_headers/flood_fill.hpp>
#include <metal_kernel_headers/gradient.hpp>
#include <metal_kernel_headers/harris.hpp>
#include <metal_kernel_headers/histogram.hpp>
#include <metal_kernel_headers/hsv_rgb.hpp>
#include <metal_kernel_headers/identity.hpp>
#include <metal_kernel_headers/iir.hpp>
#include <metal_kernel_headers/index.hpp>
#include <metal_kernel_headers/interp.hpp>
#include <metal_kernel_headers/iota.hpp>
#include <metal_kernel_headers/ireduce.hpp>
#include <metal_kernel_headers/join.hpp>
#include <metal_kernel_headers/lookup.hpp>
#include <metal_kernel_headers/lu.hpp>
#include <metal_kernel_headers/match_template.hpp>
#include <metal_kernel_headers/mean.hpp>
#include <metal_kernel_headers/meanshift.hpp>
#include <metal_kernel_headers/medfilt.hpp>
#include <metal_kernel_headers/moments.hpp>
#include <metal_kernel_headers/morph.hpp>
#include <metal_kernel_headers/nearest_neighbour.hpp>
#include <metal_kernel_headers/orb.hpp>
#include <metal_kernel_headers/pad_borders.hpp>
#include <metal_kernel_headers/random_engine.hpp>
#include <metal_kernel_headers/range.hpp>
#include <metal_kernel_headers/reduce.hpp>
#include <metal_kernel_headers/regions.hpp>
#include <metal_kernel_headers/reorder.hpp>
#include <metal_kernel_headers/resize.hpp>
#include <metal_kernel_headers/rotate.hpp>
#include <metal_kernel_headers/scan.hpp>
#include <metal_kernel_headers/scan_by_key.hpp>
#include <metal_kernel_headers/select.hpp>
#include <metal_kernel_headers/shift.hpp>
#include <metal_kernel_headers/sift.hpp>
#include <metal_kernel_headers/sobel.hpp>
#include <metal_kernel_headers/sort.hpp>
#include <metal_kernel_headers/sort_by_key.hpp>
#include <metal_kernel_headers/sort_helper.hpp>
#include <metal_kernel_headers/sparse.hpp>
#include <metal_kernel_headers/sparse_arith.hpp>
#include <metal_kernel_headers/susan.hpp>
#include <metal_kernel_headers/tile.hpp>
#include <metal_kernel_headers/transform.hpp>
#include <metal_kernel_headers/transpose.hpp>
#include <metal_kernel_headers/triangle.hpp>
#include <metal_kernel_headers/unwrap.hpp>
#include <metal_kernel_headers/wrap.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace arrayfire {
namespace metal {
namespace kernel {
namespace {

const std::string computeSourceStorage = [] {
    std::string source;
    const auto append = [&source](const common::Source& kernelSource) {
        source.append(kernelSource.ptr, kernelSource.length - 1);
        source.push_back('\n');
    };

    append(copy_metal_src);
    append(anisotropic_diffusion_metal_src);
    append(Array_metal_src);
    append(approx_metal_src);
    append(interp_metal_src);
    append(assign_metal_src);
    append(bilateral_metal_src);
    append(canny_metal_src);
    append(convolve_metal_src);
    append(exampleFunction_metal_src);
    append(range_metal_src);
    append(random_engine_metal_src);
    append(reduce_metal_src);
    append(iota_metal_src);
    append(ireduce_metal_src);
    append(identity_metal_src);
    append(iir_metal_src);
    append(index_metal_src);
    append(tile_metal_src);
    append(shift_metal_src);
    append(sift_metal_src);
    append(reorder_metal_src);
    append(regions_metal_src);
    append(rotate_metal_src);
    append(resize_metal_src);
    append(scan_metal_src);
    append(scan_by_key_metal_src);
    append(select_metal_src);
    append(join_metal_src);
    append(lookup_metal_src);
    append(lu_metal_src);
    append(match_template_metal_src);
    append(moments_metal_src);
    append(morph_metal_src);
    append(meanshift_metal_src);
    append(mean_metal_src);
    append(medfilt_metal_src);
    append(nearest_neighbour_metal_src);
    append(orb_metal_src);
    append(pad_borders_metal_src);
    append(diagonal_metal_src);
    append(diff_metal_src);
    append(dot_metal_src);
    append(flood_fill_metal_src);
    append(fast_metal_src);
    append(fftconvolve_metal_src);
    append(triangle_metal_src);
    append(gradient_metal_src);
    append(hsv_rgb_metal_src);
    append(histogram_metal_src);
    append(harris_metal_src);
    append(sobel_metal_src);
    append(sort_helper_metal_src);
    append(sort_metal_src);
    append(sort_by_key_metal_src);
    append(sparse_metal_src);
    append(sparse_arith_metal_src);
    append(susan_metal_src);
    append(transpose_metal_src);
    append(transform_metal_src);
    append(unwrap_metal_src);
    append(wrap_metal_src);
    return source;
}();

const char* computeSource = computeSourceStorage.c_str();

struct CopyParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
};

struct RangeParams {
    uint64_t dims[4];
    uint64_t strides[4];
    uint32_t sequenceDimension;
};

const char* copyFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "copy_float";
        case c32: return "copy_cfloat";
        case s32: return "copy_int";
        case u32: return "copy_uint";
        case s64: return "copy_long";
        case u64: return "copy_ulong";
        case s8: return "copy_char";
        case u8:
        case b8: return "copy_uchar";
        case s16: return "copy_short";
        case u16: return "copy_ushort";
        case f16: return "copy_half";
        default: return nullptr;
    }
}

struct ResizeParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint32_t method;
};

struct PadParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    int64_t lower[4];
    uint32_t borderType;
};

struct HsvRgbParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
};

struct MomentsParams {
    uint64_t inputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t moment;
};

struct HistogramParams {
    uint64_t inputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t bins;
    float minValue;
    float maxValue;
    uint32_t linear;
};

struct MorphParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t maskDims[4];
    uint64_t maskStrides[4];
    uint32_t dilation;
    uint32_t volume;
};

struct NearestNeighbourParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t queryDims[4];
    uint64_t queryStrides[4];
    uint64_t trainDims[4];
    uint64_t trainStrides[4];
    uint32_t distanceDimension;
    uint32_t distanceType;
};

struct IirParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t coefficientStrides[4];
    uint64_t feedbackDims[4];
    uint64_t feedbackStrides[4];
    uint32_t feedbackBatched;
};

struct DotParams {
    uint64_t length;
    uint64_t lhsStride;
    uint64_t rhsStride;
    uint32_t conjugateLhs;
    uint32_t conjugateRhs;
};

struct BilateralParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    float spatialSigma;
    float chromaticSigma;
};

struct MeanshiftParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    float spatialSigma;
    float chromaticSigma;
    uint32_t iterations;
    uint32_t color;
};

struct MedfiltParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t windowLength;
    uint32_t windowWidth;
    uint32_t padding;
    uint32_t oneDimensional;
};
struct MatchTemplateParams {
    uint64_t dims[4], outputStrides[4], searchStrides[4], templateDims[4],
        templateStrides[4];
    uint32_t matchType;
};
struct RotateParams {
    uint64_t odims[4], ostrides[4], idims[4], istrides[4];
    float theta;
    uint32_t method;
};
struct LuParams {
    uint64_t odims[4], ostrides[4], idims[4], istrides[4];
    uint32_t lower;
};
struct PivotParams {
    uint64_t outputCount, pivotCount;
};

const char* hsvRgbFunctionName(const af_dtype type, const bool hsvToRgb) {
    return type == f32 ? (hsvToRgb ? "hsv2rgb_float" : "rgb2hsv_float")
                       : nullptr;
}

const char* momentsFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "moments_float";
        case s32: return "moments_int";
        case u32: return "moments_uint";
        case s8: return "moments_char";
        case u8:
        case b8: return "moments_uchar";
        case s16: return "moments_short";
        case u16: return "moments_ushort";
        default: return nullptr;
    }
}

const char* histogramFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "histogram_float";
        case s32: return "histogram_int";
        case u32: return "histogram_uint";
        case s64: return "histogram_long";
        case u64: return "histogram_ulong";
        case s8: return "histogram_char";
        case u8:
        case b8: return "histogram_uchar";
        case s16: return "histogram_short";
        case u16: return "histogram_ushort";
        case f16: return "histogram_half";
        default: return nullptr;
    }
}

const char* morphFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "morph_float";
        case s32: return "morph_int";
        case u32: return "morph_uint";
        case s8: return "morph_char";
        case u8:
        case b8: return "morph_uchar";
        case s16: return "morph_short";
        case u16: return "morph_ushort";
        default: return nullptr;
    }
}

const char* nearestNeighbourFunctionName(const af_dtype inputType,
                                         const af_dtype outputType,
                                         const af_match_type distanceType) {
    if (distanceType == AF_SHD) {
        if (outputType != u32) return nullptr;
        switch (inputType) {
            case u8: return "nearest_hamming_uchar";
            case u16: return "nearest_hamming_ushort";
            case u32: return "nearest_hamming_uint";
            case u64: return "nearest_hamming_ulong";
            default: return nullptr;
        }
    }
    if (distanceType != AF_SAD && distanceType != AF_SSD) return nullptr;
    if (inputType == f32 && outputType == f32) return "nearest_float";
    if (inputType == s32 && outputType == s32) return "nearest_int";
    if (inputType == u32 && outputType == u32) return "nearest_uint";
    if (inputType == s64 && outputType == s64) return "nearest_long";
    if (inputType == u64 && outputType == u64) return "nearest_ulong";
    if (inputType == s8 && outputType == s32) return "nearest_char_int";
    if (inputType == u8 && outputType == u32) return "nearest_uchar_uint";
    if (inputType == s16 && outputType == s32) return "nearest_short_int";
    if (inputType == u16 && outputType == u32) return "nearest_ushort_uint";
    return nullptr;
}

const char* iirFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "iir_float";
        case c32: return "iir_cfloat";
        default: return nullptr;
    }
}

const char* dotFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "dot_float";
        case c32: return "dot_cfloat";
        default: return nullptr;
    }
}

const char* bilateralFunctionName(const af_dtype inputType,
                                  const af_dtype outputType) {
    if (outputType != f32) return nullptr;
    switch (inputType) {
        case f32: return "bilateral_float";
        case s32: return "bilateral_int";
        case u32: return "bilateral_uint";
        case s8: return "bilateral_char";
        case u8:
        case b8: return "bilateral_uchar";
        case s16: return "bilateral_short";
        case u16: return "bilateral_ushort";
        default: return nullptr;
    }
}

const char* meanshiftFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "meanshift_float";
        case s32: return "meanshift_int";
        case u32: return "meanshift_uint";
        case s64: return "meanshift_long";
        case u64: return "meanshift_ulong";
        case s8: return "meanshift_char";
        case u8:
        case b8: return "meanshift_uchar";
        case s16: return "meanshift_short";
        case u16: return "meanshift_ushort";
        default: return nullptr;
    }
}

const char* medfiltFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "medfilt_float";
        case s32: return "medfilt_int";
        case u32: return "medfilt_uint";
        case s8: return "medfilt_char";
        case u8:
        case b8: return "medfilt_uchar";
        case s16: return "medfilt_short";
        case u16: return "medfilt_ushort";
        default: return nullptr;
    }
}
const char* matchTemplateFunctionName(const af_dtype in, const af_dtype out) {
    if (out != f32) return nullptr;
    switch (in) {
        case f32: return "match_float";
        case s32: return "match_int";
        case u32: return "match_uint";
        case s8: return "match_char";
        case u8:
        case b8: return "match_uchar";
        case s16: return "match_short";
        case u16: return "match_ushort";
        default: return nullptr;
    }
}
const char* rotateFunctionName(const af_dtype t) {
    switch (t) {
        case f32: return "rotate_float";
        case c32: return "rotate_cfloat";
        case s32: return "rotate_int";
        case u32: return "rotate_uint";
        case s64: return "rotate_long";
        case u64: return "rotate_ulong";
        case s8: return "rotate_char";
        case u8:
        case b8: return "rotate_uchar";
        case s16: return "rotate_short";
        case u16: return "rotate_ushort";
        default: return nullptr;
    }
}
const char* luFunctionName(const af_dtype t) {
    return t == f32 ? "lu_float" : t == c32 ? "lu_cfloat" : nullptr;
}

const char* floodFillFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "flood_fill_float";
        case u32: return "flood_fill_uint";
        case u16: return "flood_fill_ushort";
        case u8: return "flood_fill_uchar";
        default: return nullptr;
    }
}

const char* indexFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "index_float";
        case c32: return "index_cfloat";
        case s32: return "index_int";
        case u32: return "index_uint";
        case s64: return "index_long";
        case u64: return "index_ulong";
        case s8: return "index_char";
        case u8:
        case b8: return "index_uchar";
        case s16: return "index_short";
        case u16: return "index_ushort";
        case f16: return "index_half";
        default: return nullptr;
    }
}

const char* assignFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "assign_float";
        case c32: return "assign_cfloat";
        case s32: return "assign_int";
        case u32: return "assign_uint";
        case s64: return "assign_long";
        case u64: return "assign_ulong";
        case s8: return "assign_char";
        case u8:
        case b8: return "assign_uchar";
        case s16: return "assign_short";
        case u16: return "assign_ushort";
        case f16: return "assign_half";
        default: return nullptr;
    }
}

const char* transformFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "transform_float";
        case c32: return "transform_cfloat";
        case s32: return "transform_int";
        case u32: return "transform_uint";
        case s64: return "transform_long";
        case u64: return "transform_ulong";
        case s8: return "transform_char";
        case u8:
        case b8: return "transform_uchar";
        case s16: return "transform_short";
        case u16: return "transform_ushort";
        default: return nullptr;
    }
}

const char* sortFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "sort_float";
        case s32: return "sort_int";
        case u32: return "sort_uint";
        case s64: return "sort_long";
        case u64: return "sort_ulong";
        case s8: return "sort_char";
        case u8:
        case b8: return "sort_uchar";
        case s16: return "sort_short";
        case u16: return "sort_ushort";
        default: return nullptr;
    }
}

const char* sortByKeyFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "sort_by_key_float";
        case s32: return "sort_by_key_int";
        case u32: return "sort_by_key_uint";
        case s64: return "sort_by_key_long";
        case u64: return "sort_by_key_ulong";
        case s8: return "sort_by_key_char";
        case u8:
        case b8: return "sort_by_key_uchar";
        case s16: return "sort_by_key_short";
        case u16: return "sort_by_key_ushort";
        default: return nullptr;
    }
}

const char* scanFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "scan_add_float";
        case s32: return "scan_add_int";
        case u32: return "scan_add_uint";
        case s64: return "scan_add_long";
        case u64: return "scan_add_ulong";
        case s8: return "scan_add_char";
        case u8:
        case b8: return "scan_add_uchar";
        case s16: return "scan_add_short";
        case u16: return "scan_add_ushort";
        default: return nullptr;
    }
}

const char* scanByKeyFunctionName(const af_dtype keyType,
                                  const af_dtype valueType) {
    if (keyType != s32) return nullptr;
    switch (valueType) {
        case s32: return "scan_by_key_add_int_int";
        case f32: return "scan_by_key_add_int_float";
        default: return nullptr;
    }
}

struct ScanParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t dimension;
};

struct ScanByKeyParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t keyStrides[4];
    uint64_t inputStrides[4];
    uint32_t dimension;
};

struct MeanParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint32_t dimension;
};

struct ApproxParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t xDims[4];
    uint64_t xStrides[4];
    uint64_t yStrides[4];
    uint32_t xDimension;
    uint32_t yDimension;
    uint32_t method;
    float xBegin;
    float xStep;
    float yBegin;
    float yStep;
    float offGrid;
};

struct ReduceParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint32_t dimension;
    uint32_t changeNan;
    float nanValue;
};

struct IReduceParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint32_t dimension;
};

struct ConvolveParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t signalDims[4];
    uint64_t signalStrides[4];
    uint64_t filterDims[4];
    uint64_t filterStrides[4];
    uint32_t rank;
    uint32_t expand;
};

struct FastParams {
    uint64_t rows;
    uint64_t columns;
    float threshold;
    uint32_t arcLength;
    uint32_t nonmax;
    uint32_t maxFeatures;
    uint32_t edge;
};

struct HarrisParams {
    uint64_t elements;
    uint32_t rows;
    uint32_t columns;
    uint32_t border;
    float k;
};

struct SusanParams {
    uint32_t rows;
    uint32_t columns;
    uint32_t radius;
    uint32_t border;
    float differenceThreshold;
    float geometricThreshold;
};

struct RandomParams {
    uint64_t elements;
    uint64_t seed;
    uint64_t counter;
    uint32_t type;
};

struct MersenneInitParams {
    uint64_t seed;
};

struct SparseParams {
    uint32_t rows;
    uint32_t columns;
    uint32_t outputStride;
    uint32_t nonzeros;
};

struct SparseArithParams {
    uint32_t nonzeros;
    uint32_t rows;
    uint32_t rhsStride;
    uint32_t csr;
    uint32_t reverse;
    uint32_t operation;
};

struct FFTConvolveParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t signalDims[4];
    uint64_t signalStrides[4];
    uint64_t filterDims[4];
    uint64_t filterStrides[4];
    uint64_t offset;
    uint32_t kind;
};

struct OrbParams {
    uint32_t features;
    uint32_t rows;
    uint32_t columns;
    uint32_t patchSize;
};

struct SiftParams {
    uint64_t elements;
};

struct ArrayAddParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t leftDims[4];
    uint64_t leftStrides[4];
    uint64_t rightDims[4];
    uint64_t rightStrides[4];
};

struct ExampleFunctionParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t leftStrides[4];
    uint64_t rightStrides[4];
};

struct SortByKeyParams {
    uint64_t dims[4];
    uint64_t keyStrides[4];
    uint64_t valueStrides[4];
    uint32_t ascending;
};

struct SortParams {
    uint64_t dims[4];
    uint64_t strides[4];
    uint32_t ascending;
};

struct TransformParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint32_t method;
    uint32_t inverse;
    uint32_t perspective;
};

struct AssignParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t rhsDims[4];
    uint64_t rhsStrides[4];
    int64_t offsets[4];
};

struct IndexParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    int64_t offsets[4];
    int64_t steps[4];
};

struct FloodFillParams {
    uint64_t imageDims[4];
    uint64_t imageStrides[4];
    uint64_t seedDims[4];
    uint64_t seedXStrides[4];
    uint64_t seedYStrides[4];
};

struct CannyParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t firstStrides[4];
    uint64_t secondStrides[4];
    uint64_t thirdStrides[4];
};

struct RegionsParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t connectivity;
};

struct DiffusionParams {
    uint64_t dims[4];
    uint64_t strides[4];
    float dt;
    float mct;
    uint32_t flux;
    uint32_t curvature;
};

struct IotaParams {
    uint64_t dims[4];
    uint64_t strides[4];
    uint64_t sourceDims[4];
};

struct IdentityParams {
    uint64_t dims[4];
    uint64_t strides[4];
};

struct TileParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
};

struct ShiftParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint64_t shifts[4];
};

struct ReorderParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint64_t reorderDims[4];
};

struct SelectParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t aDims[4];
    uint64_t aStrides[4];
    uint64_t bDims[4];
    uint64_t bStrides[4];
    uint64_t conditionDims[4];
    uint64_t conditionStrides[4];
    uint32_t flip;
};

struct JoinParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t outputOffset[4];
};

struct LookupParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint32_t dimension;
    uint32_t indexType;
};

struct DiagonalParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    int32_t diagonal;
};

struct DiffParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t dimension;
};

struct TriangleParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t upper;
    uint32_t unitDiagonal;
};

struct TransposeParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t conjugate;
};

struct TransposeInplaceParams {
    uint64_t dims[4];
    uint64_t strides[4];
    uint32_t conjugate;
};

struct UnwrapParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t windowX;
    uint64_t windowY;
    uint64_t strideX;
    uint64_t strideY;
    int64_t paddingX;
    int64_t paddingY;
    uint64_t dilationX;
    uint64_t dilationY;
    uint32_t columnDimension;
};

using WrapParams = UnwrapParams;

struct GradientParams {
    uint64_t dims[4];
    uint64_t inputStrides[4];
    uint64_t gradient0Strides[4];
    uint64_t gradient1Strides[4];
};

struct SobelParams {
    uint64_t dims[4];
    uint64_t inputStrides[4];
    uint64_t derivative0Strides[4];
    uint64_t derivative1Strides[4];
};

std::string description(NS::Error* error, const char* fallback) {
    if (!error || !error->localizedDescription()) { return fallback; }
    const char* value = error->localizedDescription()->utf8String();
    return value ? value : fallback;
}

const char* rangeFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "range_float";
        case s32: return "range_int";
        case u32: return "range_uint";
        case s64: return "range_long";
        case u64: return "range_ulong";
        case s8: return "range_char";
        case u8:
        case b8: return "range_uchar";
        case s16: return "range_short";
        case u16: return "range_ushort";
        case f16: return "range_half";
        default: return nullptr;
    }
}

const char* resizeFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "resize_float";
        case c32: return "resize_cfloat";
        case s32: return "resize_int";
        case u32: return "resize_uint";
        case s64: return "resize_long";
        case u64: return "resize_ulong";
        case s8: return "resize_char";
        case u8:
        case b8: return "resize_uchar";
        case s16: return "resize_short";
        case u16: return "resize_ushort";
        default: return nullptr;
    }
}

const char* padFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "pad_float";
        case c32: return "pad_cfloat";
        case s32: return "pad_int";
        case u32: return "pad_uint";
        case s64: return "pad_long";
        case u64: return "pad_ulong";
        case s8: return "pad_char";
        case u8:
        case b8: return "pad_uchar";
        case s16: return "pad_short";
        case u16: return "pad_ushort";
        default: return nullptr;
    }
}

const char* iotaFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "iota_float";
        case s32: return "iota_int";
        case u32: return "iota_uint";
        case s64: return "iota_long";
        case u64: return "iota_ulong";
        case s8: return "iota_char";
        case u8:
        case b8: return "iota_uchar";
        case s16: return "iota_short";
        case u16: return "iota_ushort";
        case f16: return "iota_half";
        default: return nullptr;
    }
}

const char* identityFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "identity_float";
        case c32: return "identity_cfloat";
        case s32: return "identity_int";
        case u32: return "identity_uint";
        case s64: return "identity_long";
        case u64: return "identity_ulong";
        case s8: return "identity_char";
        case u8:
        case b8: return "identity_uchar";
        case s16: return "identity_short";
        case u16: return "identity_ushort";
        case f16: return "identity_half";
        default: return nullptr;
    }
}

const char* tileFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "tile_float";
        case c32: return "tile_cfloat";
        case s32: return "tile_int";
        case u32: return "tile_uint";
        case s64: return "tile_long";
        case u64: return "tile_ulong";
        case s8: return "tile_char";
        case u8:
        case b8: return "tile_uchar";
        case s16: return "tile_short";
        case u16: return "tile_ushort";
        case f16: return "tile_half";
        default: return nullptr;
    }
}

const char* shiftFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "shift_float";
        case c32: return "shift_cfloat";
        case s32: return "shift_int";
        case u32: return "shift_uint";
        case s64: return "shift_long";
        case u64: return "shift_ulong";
        case s8: return "shift_char";
        case u8:
        case b8: return "shift_uchar";
        case s16: return "shift_short";
        case u16: return "shift_ushort";
        case f16: return "shift_half";
        default: return nullptr;
    }
}

const char* reorderFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "reorder_float";
        case c32: return "reorder_cfloat";
        case s32: return "reorder_int";
        case u32: return "reorder_uint";
        case s64: return "reorder_long";
        case u64: return "reorder_ulong";
        case s8: return "reorder_char";
        case u8:
        case b8: return "reorder_uchar";
        case s16: return "reorder_short";
        case u16: return "reorder_ushort";
        case f16: return "reorder_half";
        default: return nullptr;
    }
}

const char* selectFunctionName(const af_dtype type, const bool scalar) {
    const char* prefix = scalar ? "select_scalar_" : "select_";
    const char* suffix = nullptr;
    switch (type) {
        case f32: suffix = "float"; break;
        case c32: suffix = "cfloat"; break;
        case s32: suffix = "int"; break;
        case u32: suffix = "uint"; break;
        case s64: suffix = "long"; break;
        case u64: suffix = "ulong"; break;
        case s8: suffix = "char"; break;
        case u8:
        case b8: suffix = "uchar"; break;
        case s16: suffix = "short"; break;
        case u16: suffix = "ushort"; break;
        case f16: suffix = "half"; break;
        default: return nullptr;
    }
    static thread_local std::string name;
    name = prefix;
    name += suffix;
    return name.c_str();
}

const char* joinFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "join_float";
        case c32: return "join_cfloat";
        case s32: return "join_int";
        case u32: return "join_uint";
        case s64: return "join_long";
        case u64: return "join_ulong";
        case s8: return "join_char";
        case u8:
        case b8: return "join_uchar";
        case s16: return "join_short";
        case u16: return "join_ushort";
        case f16: return "join_half";
        default: return nullptr;
    }
}

const char* lookupFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "lookup_float";
        case c32: return "lookup_cfloat";
        case s32: return "lookup_int";
        case u32: return "lookup_uint";
        case s64: return "lookup_long";
        case u64: return "lookup_ulong";
        case s8: return "lookup_char";
        case u8:
        case b8: return "lookup_uchar";
        case s16: return "lookup_short";
        case u16: return "lookup_ushort";
        case f16: return "lookup_half";
        default: return nullptr;
    }
}

int lookupIndexType(const af_dtype type) {
    switch (type) {
        case f32: return 0;
        case s32: return 1;
        case u32: return 2;
        case s64: return 3;
        case u64: return 4;
        case s8: return 5;
        case u8:
        case b8: return 6;
        case s16: return 7;
        case u16: return 8;
        case f16: return 9;
        default: return -1;
    }
}

const char* diagonalFunctionName(const af_dtype type, const bool create) {
    switch (type) {
        case f32: return create ? "diag_create_float" : "diag_extract_float";
        case c32: return create ? "diag_create_cfloat" : "diag_extract_cfloat";
        case s32: return create ? "diag_create_int" : "diag_extract_int";
        case u32: return create ? "diag_create_uint" : "diag_extract_uint";
        case s64: return create ? "diag_create_long" : "diag_extract_long";
        case u64: return create ? "diag_create_ulong" : "diag_extract_ulong";
        case s8: return create ? "diag_create_char" : "diag_extract_char";
        case u8:
        case b8: return create ? "diag_create_uchar" : "diag_extract_uchar";
        case s16: return create ? "diag_create_short" : "diag_extract_short";
        case u16: return create ? "diag_create_ushort" : "diag_extract_ushort";
        case f16: return create ? "diag_create_half" : "diag_extract_half";
        default: return nullptr;
    }
}

const char* diffFunctionName(const af_dtype type, const bool secondOrder) {
    switch (type) {
        case f32: return secondOrder ? "diff2_float" : "diff1_float";
        case c32: return secondOrder ? "diff2_cfloat" : "diff1_cfloat";
        case s32: return secondOrder ? "diff2_int" : "diff1_int";
        case u32: return secondOrder ? "diff2_uint" : "diff1_uint";
        case s64: return secondOrder ? "diff2_long" : "diff1_long";
        case u64: return secondOrder ? "diff2_ulong" : "diff1_ulong";
        case s8: return secondOrder ? "diff2_char" : "diff1_char";
        case u8:
        case b8: return secondOrder ? "diff2_uchar" : "diff1_uchar";
        case s16: return secondOrder ? "diff2_short" : "diff1_short";
        case u16: return secondOrder ? "diff2_ushort" : "diff1_ushort";
        case f16: return secondOrder ? "diff2_half" : "diff1_half";
        default: return nullptr;
    }
}

const char* triangleFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "triangle_float";
        case c32: return "triangle_cfloat";
        case s32: return "triangle_int";
        case u32: return "triangle_uint";
        case s64: return "triangle_long";
        case u64: return "triangle_ulong";
        case s8: return "triangle_char";
        case u8:
        case b8: return "triangle_uchar";
        case s16: return "triangle_short";
        case u16: return "triangle_ushort";
        case f16: return "triangle_half";
        default: return nullptr;
    }
}

const char* transposeFunctionName(const af_dtype type, const bool inplace) {
    const char* prefix = inplace ? "transpose_inplace_" : "transpose_";
    const char* suffix = nullptr;
    switch (type) {
        case f32: suffix = "float"; break;
        case c32: suffix = "cfloat"; break;
        case s32: suffix = "int"; break;
        case u32: suffix = "uint"; break;
        case s64: suffix = "long"; break;
        case u64: suffix = "ulong"; break;
        case s8: suffix = "char"; break;
        case u8:
        case b8: suffix = "uchar"; break;
        case s16: suffix = "short"; break;
        case u16: suffix = "ushort"; break;
        case f16: suffix = "half"; break;
        default: return nullptr;
    }
    static thread_local std::string name;
    name = prefix;
    name += suffix;
    return name.c_str();
}

const char* unwrapFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "unwrap_float";
        case c32: return "unwrap_cfloat";
        case s32: return "unwrap_int";
        case u32: return "unwrap_uint";
        case s64: return "unwrap_long";
        case u64: return "unwrap_ulong";
        case s8: return "unwrap_char";
        case u8:
        case b8: return "unwrap_uchar";
        case s16: return "unwrap_short";
        case u16: return "unwrap_ushort";
        case f16: return "unwrap_half";
        default: return nullptr;
    }
}

const char* wrapFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "wrap_float";
        case c32: return "wrap_cfloat";
        case s32: return "wrap_int";
        case u32: return "wrap_uint";
        case s64: return "wrap_long";
        case u64: return "wrap_ulong";
        case s8: return "wrap_char";
        case u8:
        case b8: return "wrap_uchar";
        case s16: return "wrap_short";
        case u16: return "wrap_ushort";
        case f16: return "wrap_half";
        default: return nullptr;
    }
}

const char* gradientFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "gradient_float";
        case c32: return "gradient_cfloat";
        default: return nullptr;
    }
}

const char* sobelFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "sobel_float";
        case s32: return "sobel_int";
        case u32: return "sobel_uint";
        case s8:
        case b8: return "sobel_char";
        case u8: return "sobel_uchar";
        case s16: return "sobel_short";
        case u16: return "sobel_ushort";
        default: return nullptr;
    }
}

class MetalRuntime {
   public:
    MetalRuntime() : device(NS::TransferPtr(MTL::CreateSystemDefaultDevice())) {
        if (!device) {
            AF_ERROR("No Metal device is available", AF_ERR_RUNTIME);
        }

        commandQueue = NS::TransferPtr(device->newCommandQueue());
        if (!commandQueue) {
            AF_ERROR("Could not create a Metal command queue", AF_ERR_RUNTIME);
        }

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        NS::Error* error          = nullptr;
        library                   = NS::TransferPtr(device->newLibrary(
            NS::String::string(computeSource, NS::UTF8StringEncoding), nullptr,
            &error));
        if (!library) {
            const std::string message =
                description(error, "Metal compile failed");
            pool->release();
            AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
        }
        pool->release();
    }

    MTL::ComputePipelineState* pipeline(const char* name) {
        std::lock_guard<std::mutex> lock(mutex);
        const std::string key(name);
        const auto found = pipelines.find(key);
        if (found != pipelines.end()) { return found->second.get(); }

        NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
        auto function             = NS::TransferPtr(library->newFunction(
            NS::String::string(name, NS::UTF8StringEncoding)));
        if (!function) {
            pool->release();
            AF_ERROR("Could not load the Metal range kernel", AF_ERR_RUNTIME);
        }

        NS::Error* error = nullptr;
        auto state       = NS::TransferPtr(
            device->newComputePipelineState(function.get(), &error));
        if (!state) {
            const std::string message =
                description(error, "Metal pipeline creation failed");
            pool->release();
            AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
        }
        pool->release();

        auto inserted = pipelines.emplace(key, std::move(state));
        return inserted.first->second.get();
    }

    MTL::Device* getDevice() const noexcept { return device.get(); }
    MTL::CommandQueue* getCommandQueue() const noexcept {
        return commandQueue.get();
    }

   private:
    NS::SharedPtr<MTL::Device> device;
    NS::SharedPtr<MTL::CommandQueue> commandQueue;
    NS::SharedPtr<MTL::Library> library;
    std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>
        pipelines;
    std::mutex mutex;
};

MetalRuntime& metalRuntime() {
    static MetalRuntime runtime;
    return runtime;
}

void launchSingleInputKernel(void* output, const size_t outputBytes,
                             const af::dim4& outputDims, const void* input,
                             const size_t inputBytes, const void* params,
                             const size_t paramsBytes, const char* functionName,
                             const char* operationName) {
    if (outputBytes == 0) return;
    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(params, paramsBytes, 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string fallback =
            std::string("Metal ") + operationName + " dispatch failed";
        const std::string message =
            description(commandBuffer->error(), fallback.c_str());
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchTwoInputKernel(void* output, const size_t outputBytes,
                          const void* first, const size_t firstBytes,
                          const void* second, const size_t secondBytes,
                          const void* params, const size_t paramsBytes,
                          const size_t dispatchTotal, const char* functionName,
                          const char* operationName) {
    if (outputBytes == 0 || dispatchTotal == 0) return;
    MetalRuntime& runtime = metalRuntime();
    auto firstBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        first, firstBytes, MTL::ResourceStorageModeShared));
    auto secondBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        second, secondBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!firstBuffer || !secondBuffer || !outputBuffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(firstBuffer.get(), 0, 0);
    encoder->setBuffer(secondBuffer.get(), 0, 1);
    encoder->setBuffer(outputBuffer.get(), 0, 2);
    encoder->setBytes(params, paramsBytes, 3);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(dispatchTotal, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string fallback =
            std::string("Metal ") + operationName + " dispatch failed";
        const std::string message =
            description(commandBuffer->error(), fallback.c_str());
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchThreeInputKernel(void* output, const size_t outputBytes,
                            const void* first, const size_t firstBytes,
                            const void* second, const size_t secondBytes,
                            const void* third, const size_t thirdBytes,
                            const void* params, const size_t paramsBytes,
                            const size_t dispatchTotal,
                            const char* functionName,
                            const char* operationName) {
    if (outputBytes == 0 || dispatchTotal == 0) return;
    MetalRuntime& runtime = metalRuntime();
    auto firstBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        first, firstBytes, MTL::ResourceStorageModeShared));
    auto secondBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        second, secondBytes, MTL::ResourceStorageModeShared));
    auto thirdBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        third, thirdBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!firstBuffer || !secondBuffer || !thirdBuffer || !outputBuffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(firstBuffer.get(), 0, 0);
    encoder->setBuffer(secondBuffer.get(), 0, 1);
    encoder->setBuffer(thirdBuffer.get(), 0, 2);
    encoder->setBuffer(outputBuffer.get(), 0, 3);
    encoder->setBytes(params, paramsBytes, 4);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(dispatchTotal, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string fallback =
            std::string("Metal ") + operationName + " dispatch failed";
        const std::string message =
            description(commandBuffer->error(), fallback.c_str());
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchFloodFillKernel(void* output, const size_t outputBytes,
                           const void* image, const size_t imageBytes,
                           const void* seedX, const size_t seedXBytes,
                           const void* seedY, const size_t seedYBytes,
                           const void* params, const size_t paramsBytes,
                           const void* newValue, const void* lower,
                           const void* upper, const size_t valueBytes,
                           const char* functionName) {
    if (outputBytes == 0) return;
    MetalRuntime& runtime = metalRuntime();
    auto imageBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        image, imageBytes, MTL::ResourceStorageModeShared));
    auto seedXBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        seedX, seedXBytes, MTL::ResourceStorageModeShared));
    auto seedYBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        seedY, seedYBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!imageBuffer || !seedXBuffer || !seedYBuffer || !outputBuffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);

    auto* pipeline = runtime.pipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(imageBuffer.get(), 0, 0);
    encoder->setBuffer(seedXBuffer.get(), 0, 1);
    encoder->setBuffer(seedYBuffer.get(), 0, 2);
    encoder->setBuffer(outputBuffer.get(), 0, 3);
    encoder->setBytes(params, paramsBytes, 4);
    encoder->setBytes(newValue, valueBytes, 5);
    encoder->setBytes(lower, valueBytes, 6);
    encoder->setBytes(upper, valueBytes, 7);
    encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal flood fill dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

}  // namespace

bool supportsMetalRange(const af_dtype type) noexcept {
    return rangeFunctionName(type) != nullptr;
}

bool supportsMetalCopy(const af_dtype type) noexcept {
    return copyFunctionName(type) != nullptr;
}

bool supportsMetalResize(const af_dtype type) noexcept {
    return resizeFunctionName(type) != nullptr;
}

bool supportsMetalPadBorders(const af_dtype type) noexcept {
    return padFunctionName(type) != nullptr;
}

bool supportsMetalIota(const af_dtype type) noexcept {
    return iotaFunctionName(type) != nullptr;
}

bool supportsMetalIdentity(const af_dtype type) noexcept {
    return identityFunctionName(type) != nullptr;
}

bool supportsMetalTile(const af_dtype type) noexcept {
    return tileFunctionName(type) != nullptr;
}

bool supportsMetalShift(const af_dtype type) noexcept {
    return shiftFunctionName(type) != nullptr;
}

bool supportsMetalReorder(const af_dtype type) noexcept {
    return reorderFunctionName(type) != nullptr;
}

bool supportsMetalSelect(const af_dtype type) noexcept {
    return selectFunctionName(type, false) != nullptr;
}

bool supportsMetalJoin(const af_dtype type) noexcept {
    return joinFunctionName(type) != nullptr;
}

bool supportsMetalLookup(const af_dtype inputType,
                         const af_dtype indexType) noexcept {
    return lookupFunctionName(inputType) != nullptr &&
           lookupIndexType(indexType) >= 0;
}

bool supportsMetalDiagonal(const af_dtype type) noexcept {
    return diagonalFunctionName(type, true) != nullptr;
}

bool supportsMetalDiff(const af_dtype type) noexcept {
    return diffFunctionName(type, false) != nullptr;
}

bool supportsMetalTriangle(const af_dtype type) noexcept {
    return triangleFunctionName(type) != nullptr;
}

bool supportsMetalTranspose(const af_dtype type) noexcept {
    return transposeFunctionName(type, false) != nullptr;
}

bool supportsMetalUnwrap(const af_dtype type) noexcept {
    return unwrapFunctionName(type) != nullptr;
}

bool supportsMetalWrap(const af_dtype type) noexcept {
    return wrapFunctionName(type) != nullptr;
}

bool supportsMetalGradient(const af_dtype type) noexcept {
    return gradientFunctionName(type) != nullptr;
}

bool supportsMetalSobel(const af_dtype inputType) noexcept {
    return sobelFunctionName(inputType) != nullptr;
}

bool supportsMetalHsvRgb(const af_dtype type) noexcept {
    return hsvRgbFunctionName(type, true) != nullptr;
}

bool supportsMetalMoments(const af_dtype type) noexcept {
    return momentsFunctionName(type) != nullptr;
}

void launchMetalRange(void* output, const size_t bytes, const af::dim4& dims,
                      const af::dim4& strides, const unsigned sequenceDimension,
                      const af_dtype type) {
    if (bytes == 0) { return; }

    RangeParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]    = static_cast<uint64_t>(dims[i]);
        params.strides[i] = static_cast<uint64_t>(strides[i]);
    }
    params.sequenceDimension = sequenceDimension;

    MetalRuntime& runtime = metalRuntime();
    auto buffer           = NS::TransferPtr(
        runtime.getDevice()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    if (!buffer) {
        AF_ERROR("Could not allocate a Metal range buffer", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(rangeFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(buffer.get(), 0, 0);
    encoder->setBytes(&params, sizeof(params), 1);

    const size_t total = static_cast<size_t>(dims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal range dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, buffer->contents(), bytes);
}

void launchMetalCopy(void* output, const size_t outputBytes,
                     const af::dim4& dims, const af::dim4& outputStrides,
                     const void* input, const size_t inputBytes,
                     const af::dim4& inputStrides, const af_dtype type) {
    CopyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    launchSingleInputKernel(output, outputBytes, dims, input, inputBytes,
                            &params, sizeof(params), copyFunctionName(type),
                            "copy");
}

void launchMetalResize(void* output, const size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, const void* input,
                       const size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides,
                       const af_interp_type method, const af_dtype type) {
    ResizeParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.method = static_cast<uint32_t>(method);
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params), resizeFunctionName(type),
                            "resize");
}

void launchMetalPadBorders(void* output, const size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, const void* input,
                           const size_t inputBytes, const af::dim4& inputDims,
                           const af::dim4& inputStrides,
                           const af::dim4& lowerPadding,
                           const af_border_type borderType,
                           const af_dtype type) {
    PadParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        params.lower[i]         = static_cast<int64_t>(lowerPadding[i]);
    }
    params.borderType = static_cast<uint32_t>(borderType);
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params), padFunctionName(type),
                            "pad borders");
}

void launchMetalHsvRgb(void* output, const size_t outputBytes,
                       const af::dim4& outputStrides, const void* input,
                       const size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, const bool hsvToRgb,
                       const af_dtype type) {
    HsvRgbParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(inputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    af::dim4 dispatchDims = inputDims;
    dispatchDims[2]       = 1;
    launchSingleInputKernel(output, outputBytes, dispatchDims, input,
                            inputBytes, &params, sizeof(params),
                            hsvRgbFunctionName(type, hsvToRgb), "HSV/RGB");
}

void launchMetalMoments(void* output, const size_t outputBytes,
                        const af::dim4& outputStrides, const void* input,
                        const size_t inputBytes, const af::dim4& inputDims,
                        const af::dim4& inputStrides,
                        const af_moment_type moment, const af_dtype type) {
    MomentsParams params{};
    for (int i = 0; i < 4; ++i) {
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.moment = static_cast<uint32_t>(moment);
    const af::dim4 dispatchDims(1, 1, inputDims[2], inputDims[3]);
    launchSingleInputKernel(output, outputBytes, dispatchDims, input,
                            inputBytes, &params, sizeof(params),
                            momentsFunctionName(type), "moments");
}

bool supportsMetalHistogram(const af_dtype type) noexcept {
    return histogramFunctionName(type) != nullptr;
}

void launchMetalHistogram(void* output, const size_t outputBytes,
                          const af::dim4& outputStrides, const void* input,
                          const size_t inputBytes, const af::dim4& inputDims,
                          const af::dim4& inputStrides, const unsigned bins,
                          const double minValue, const double maxValue,
                          const bool linear, const af_dtype type) {
    HistogramParams params{};
    for (int i = 0; i < 4; ++i) {
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.bins     = bins;
    params.minValue = static_cast<float>(minValue);
    params.maxValue = static_cast<float>(maxValue);
    params.linear   = linear;
    const af::dim4 dispatchDims(1, 1, inputDims[2], inputDims[3]);
    launchSingleInputKernel(output, outputBytes, dispatchDims, input,
                            inputBytes, &params, sizeof(params),
                            histogramFunctionName(type), "histogram");
}

bool supportsMetalMorph(const af_dtype type) noexcept {
    return morphFunctionName(type) != nullptr;
}

void launchMetalMorph(void* output, const size_t outputBytes,
                      const af::dim4& outputDims, const af::dim4& outputStrides,
                      const void* input, const size_t inputBytes,
                      const af::dim4& inputDims, const af::dim4& inputStrides,
                      const void* mask, const size_t maskBytes,
                      const af::dim4& maskDims, const af::dim4& maskStrides,
                      const bool dilation, const bool volume,
                      const af_dtype type) {
    MorphParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        params.maskDims[i]      = static_cast<uint64_t>(maskDims[i]);
        params.maskStrides[i]   = static_cast<uint64_t>(maskStrides[i]);
    }
    params.dilation = dilation;
    params.volume   = volume;

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto maskBuffer       = NS::TransferPtr(runtime.getDevice()->newBuffer(
        mask, maskBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !maskBuffer || !outputBuffer)
        AF_ERROR("Could not allocate Metal morph buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal morph encoder", AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline(morphFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(maskBuffer.get(), 0, 1);
    encoder->setBuffer(outputBuffer.get(), 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal morph dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

bool supportsMetalNearestNeighbour(const af_dtype inputType,
                                   const af_dtype outputType,
                                   const af_match_type distanceType) noexcept {
    return nearestNeighbourFunctionName(inputType, outputType, distanceType) !=
           nullptr;
}

void launchMetalNearestNeighbour(
    void* output, const size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, const void* query, const size_t queryBytes,
    const af::dim4& queryDims, const af::dim4& queryStrides, const void* train,
    const size_t trainBytes, const af::dim4& trainDims,
    const af::dim4& trainStrides, const unsigned distanceDimension,
    const af_match_type distanceType, const af_dtype inputType,
    const af_dtype outputType) {
    NearestNeighbourParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.queryDims[i]     = static_cast<uint64_t>(queryDims[i]);
        params.queryStrides[i]  = static_cast<uint64_t>(queryStrides[i]);
        params.trainDims[i]     = static_cast<uint64_t>(trainDims[i]);
        params.trainStrides[i]  = static_cast<uint64_t>(trainStrides[i]);
    }
    params.distanceDimension = distanceDimension;
    params.distanceType      = static_cast<uint32_t>(distanceType);

    MetalRuntime& runtime = metalRuntime();
    auto queryBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        query, queryBytes, MTL::ResourceStorageModeShared));
    auto trainBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        train, trainBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!queryBuffer || !trainBuffer || !outputBuffer)
        AF_ERROR("Could not allocate Metal nearest-neighbour buffers",
                 AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal nearest-neighbour encoder",
                 AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline(
        nearestNeighbourFunctionName(inputType, outputType, distanceType));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(queryBuffer.get(), 0, 0);
    encoder->setBuffer(trainBuffer.get(), 0, 1);
    encoder->setBuffer(outputBuffer.get(), 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal nearest-neighbour dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

bool supportsMetalIir(const af_dtype type) noexcept {
    return iirFunctionName(type) != nullptr;
}

void launchMetalIir(void* output, const size_t outputBytes,
                    const af::dim4& outputDims, const af::dim4& outputStrides,
                    const void* coefficients, const size_t coefficientBytes,
                    const af::dim4& coefficientStrides, const void* feedback,
                    const size_t feedbackBytes, const af::dim4& feedbackDims,
                    const af::dim4& feedbackStrides, const bool feedbackBatched,
                    const af_dtype type) {
    IirParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.coefficientStrides[i] =
            static_cast<uint64_t>(coefficientStrides[i]);
        params.feedbackDims[i]    = static_cast<uint64_t>(feedbackDims[i]);
        params.feedbackStrides[i] = static_cast<uint64_t>(feedbackStrides[i]);
    }
    params.feedbackBatched = feedbackBatched;
    const size_t series =
        static_cast<size_t>(outputDims[1] * outputDims[2] * outputDims[3]);
    launchTwoInputKernel(output, outputBytes, coefficients, coefficientBytes,
                         feedback, feedbackBytes, &params, sizeof(params),
                         series, iirFunctionName(type), "IIR");
}

bool supportsMetalDot(const af_dtype type) noexcept {
    return dotFunctionName(type) != nullptr;
}

void launchMetalDot(void* output, const size_t outputBytes, const void* lhs,
                    const size_t lhsBytes, const af::dim4& lhsDims,
                    const af::dim4& lhsStrides, const void* rhs,
                    const size_t rhsBytes, const af::dim4& rhsStrides,
                    const af_mat_prop lhsOption, const af_mat_prop rhsOption,
                    const af_dtype type) {
    DotParams params{};
    params.length       = static_cast<uint64_t>(lhsDims[0]);
    params.lhsStride    = static_cast<uint64_t>(lhsStrides[0]);
    params.rhsStride    = static_cast<uint64_t>(rhsStrides[0]);
    params.conjugateLhs = lhsOption == AF_MAT_CONJ;
    params.conjugateRhs = rhsOption == AF_MAT_CONJ;
    launchTwoInputKernel(output, outputBytes, lhs, lhsBytes, rhs, rhsBytes,
                         &params, sizeof(params), 1, dotFunctionName(type),
                         "dot");
}

bool supportsMetalBilateral(const af_dtype inputType,
                            const af_dtype outputType) noexcept {
    return bilateralFunctionName(inputType, outputType) != nullptr;
}

void launchMetalBilateral(void* output, const size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, const void* input,
                          const size_t inputBytes, const af::dim4& inputStrides,
                          const float spatialSigma, const float chromaticSigma,
                          const af_dtype inputType, const af_dtype outputType) {
    BilateralParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.spatialSigma   = spatialSigma;
    params.chromaticSigma = chromaticSigma;
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params),
                            bilateralFunctionName(inputType, outputType),
                            "bilateral");
}

bool supportsMetalMeanshift(const af_dtype type) noexcept {
    return meanshiftFunctionName(type) != nullptr;
}

void launchMetalMeanshift(void* output, const size_t outputBytes,
                          const af::dim4& dims, const af::dim4& outputStrides,
                          const void* input, const size_t inputBytes,
                          const af::dim4& inputStrides,
                          const float spatialSigma, const float chromaticSigma,
                          const unsigned iterations, const bool color,
                          const af_dtype type) {
    MeanshiftParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.spatialSigma   = spatialSigma;
    params.chromaticSigma = chromaticSigma;
    params.iterations     = iterations;
    params.color          = color;
    af::dim4 dispatchDims = dims;
    if (color) dispatchDims[2] = 1;
    launchSingleInputKernel(output, outputBytes, dispatchDims, input,
                            inputBytes, &params, sizeof(params),
                            meanshiftFunctionName(type), "meanshift");
}

bool supportsMetalMedfilt(const af_dtype type, const dim_t windowLength,
                          const dim_t windowWidth) noexcept {
    return medfiltFunctionName(type) != nullptr && windowLength > 0 &&
           windowWidth > 0 && windowLength * windowWidth <= 225;
}

void launchMetalMedfilt(void* output, const size_t outputBytes,
                        const af::dim4& dims, const af::dim4& outputStrides,
                        const void* input, const size_t inputBytes,
                        const af::dim4& inputStrides, const dim_t windowLength,
                        const dim_t windowWidth, const af_border_type padding,
                        const bool oneDimensional, const af_dtype type) {
    MedfiltParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.windowLength   = static_cast<uint32_t>(windowLength);
    params.windowWidth    = static_cast<uint32_t>(windowWidth);
    params.padding        = static_cast<uint32_t>(padding);
    params.oneDimensional = oneDimensional;
    launchSingleInputKernel(output, outputBytes, dims, input, inputBytes,
                            &params, sizeof(params), medfiltFunctionName(type),
                            "median filter");
}
bool supportsMetalMatchTemplate(const af_dtype inputType,
                                const af_dtype outputType) noexcept {
    return matchTemplateFunctionName(inputType, outputType) != nullptr;
}
void launchMetalMatchTemplate(
    void* output, const size_t outputBytes, const af::dim4& dims,
    const af::dim4& outputStrides, const void* search, const size_t searchBytes,
    const af::dim4& searchStrides, const void* templ,
    const size_t templateBytes, const af::dim4& templateDims,
    const af::dim4& templateStrides, const af_match_type matchType,
    const af_dtype inputType, const af_dtype outputType) {
    MatchTemplateParams p{};
    for (int i = 0; i < 4; ++i) {
        p.dims[i]            = uint64_t(dims[i]);
        p.outputStrides[i]   = uint64_t(outputStrides[i]);
        p.searchStrides[i]   = uint64_t(searchStrides[i]);
        p.templateDims[i]    = uint64_t(templateDims[i]);
        p.templateStrides[i] = uint64_t(templateStrides[i]);
    }
    p.matchType = uint32_t(matchType);
    launchTwoInputKernel(output, outputBytes, search, searchBytes, templ,
                         templateBytes, &p, sizeof(p), size_t(dims.elements()),
                         matchTemplateFunctionName(inputType, outputType),
                         "match template");
}
bool supportsMetalRotate(const af_dtype t, const af_interp_type m) noexcept {
    return rotateFunctionName(t) &&
           (m == AF_INTERP_NEAREST || m == AF_INTERP_LOWER);
}
void launchMetalRotate(void* o, size_t ob, const af::dim4& od,
                       const af::dim4& os, const void* i, size_t ib,
                       const af::dim4& id, const af::dim4& is, float theta,
                       af_interp_type method, af_dtype type) {
    RotateParams p{};
    for (int k = 0; k < 4; ++k) {
        p.odims[k]    = uint64_t(od[k]);
        p.ostrides[k] = uint64_t(os[k]);
        p.idims[k]    = uint64_t(id[k]);
        p.istrides[k] = uint64_t(is[k]);
    }
    p.theta  = theta;
    p.method = uint32_t(method);
    launchSingleInputKernel(o, ob, od, i, ib, &p, sizeof(p),
                            rotateFunctionName(type), "rotate");
}
bool supportsMetalLu(const af_dtype t) noexcept {
    return luFunctionName(t) != nullptr;
}
void launchMetalLuPart(void* o, size_t ob, const af::dim4& od,
                       const af::dim4& os, const void* i, size_t ib,
                       const af::dim4& id, const af::dim4& is, bool lower,
                       af_dtype type) {
    LuParams p{};
    for (int k = 0; k < 4; ++k) {
        p.odims[k]    = uint64_t(od[k]);
        p.ostrides[k] = uint64_t(os[k]);
        p.idims[k]    = uint64_t(id[k]);
        p.istrides[k] = uint64_t(is[k]);
    }
    p.lower = lower;
    launchSingleInputKernel(o, ob, od, i, ib, &p, sizeof(p),
                            luFunctionName(type), "LU split");
}
void launchMetalConvertPivot(void* o, size_t ob, const af::dim4& od,
                             const void* pivot, size_t pb, const af::dim4& pd) {
    PivotParams p{uint64_t(od.elements()), uint64_t(pd.elements())};
    launchTwoInputKernel(o, ob, o, ob, pivot, pb, &p, sizeof(p), 1,
                         "convert_pivot", "pivot conversion");
}

bool supportsMetalFloodFill(const af_dtype type) noexcept {
    return floodFillFunctionName(type) != nullptr;
}

void launchMetalFloodFill(
    void* output, const size_t outputBytes, const af::dim4& imageDims,
    const af::dim4& imageStrides, const void* image, const size_t imageBytes,
    const void* seedX, const size_t seedXBytes, const af::dim4& seedDims,
    const af::dim4& seedXStrides, const void* seedY, const size_t seedYBytes,
    const af::dim4& seedYStrides, const void* newValue, const void* lower,
    const void* upper, const size_t valueBytes, const af_dtype type) {
    FloodFillParams params{};
    for (int i = 0; i < 4; ++i) {
        params.imageDims[i]    = static_cast<uint64_t>(imageDims[i]);
        params.imageStrides[i] = static_cast<uint64_t>(imageStrides[i]);
        params.seedDims[i]     = static_cast<uint64_t>(seedDims[i]);
        params.seedXStrides[i] = static_cast<uint64_t>(seedXStrides[i]);
        params.seedYStrides[i] = static_cast<uint64_t>(seedYStrides[i]);
    }
    launchFloodFillKernel(output, outputBytes, image, imageBytes, seedX,
                          seedXBytes, seedY, seedYBytes, &params,
                          sizeof(params), newValue, lower, upper, valueBytes,
                          floodFillFunctionName(type));
}

void launchMetalCannyNonmax(
    void* output, const size_t outputBytes, const af::dim4& dims,
    const af::dim4& outputStrides, const void* magnitude,
    const size_t magnitudeBytes, const af::dim4& magnitudeStrides,
    const void* derivativeX, const size_t derivativeXBytes,
    const af::dim4& derivativeXStrides, const void* derivativeY,
    const size_t derivativeYBytes, const af::dim4& derivativeYStrides) {
    CannyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.firstStrides[i]  = static_cast<uint64_t>(magnitudeStrides[i]);
        params.secondStrides[i] = static_cast<uint64_t>(derivativeXStrides[i]);
        params.thirdStrides[i]  = static_cast<uint64_t>(derivativeYStrides[i]);
    }
    launchThreeInputKernel(
        output, outputBytes, magnitude, magnitudeBytes, derivativeX,
        derivativeXBytes, derivativeY, derivativeYBytes, &params,
        sizeof(params), static_cast<size_t>(dims.elements()),
        "canny_nonmax_float", "Canny non-maximum suppression");
}

void launchMetalCannyHysteresis(void* output, const size_t outputBytes,
                                const af::dim4& dims,
                                const af::dim4& outputStrides,
                                const void* strong, const size_t strongBytes,
                                const af::dim4& strongStrides, const void* weak,
                                const size_t weakBytes,
                                const af::dim4& weakStrides) {
    CannyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.firstStrides[i]  = static_cast<uint64_t>(strongStrides[i]);
        params.secondStrides[i] = static_cast<uint64_t>(weakStrides[i]);
    }

    MetalRuntime& runtime = metalRuntime();
    auto strongBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        strong, strongBytes, MTL::ResourceStorageModeShared));
    auto weakBuffer       = NS::TransferPtr(runtime.getDevice()->newBuffer(
        weak, weakBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    auto changedBuffer    = NS::TransferPtr(runtime.getDevice()->newBuffer(
        sizeof(uint32_t), MTL::ResourceStorageModeShared));
    if (!strongBuffer || !weakBuffer || !outputBuffer || !changedBuffer)
        AF_ERROR("Could not allocate Metal Canny buffers", AF_ERR_NO_MEM);

    const size_t total = static_cast<size_t>(dims.elements());
    auto dispatch      = [&](const char* functionName, auto bindBuffers) {
        auto commandBuffer =
            NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
        auto encoder =
            commandBuffer
                     ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                     : nullptr;
        if (!commandBuffer || !encoder)
            AF_ERROR("Could not create a Metal command encoder",
                          AF_ERR_RUNTIME);
        auto* pipeline = runtime.pipeline(functionName);
        encoder->setComputePipelineState(pipeline);
        bindBuffers(encoder.get());
        const auto width = std::min<NS::UInteger>(
            256, pipeline->maxTotalThreadsPerThreadgroup());
        encoder->dispatchThreads(MTL::Size(total, 1, 1),
                                      MTL::Size(width, 1, 1));
        encoder->endEncoding();
        commandBuffer->commit();
        commandBuffer->waitUntilCompleted();
        if (commandBuffer->status() == MTL::CommandBufferStatusError) {
            const std::string message = description(
                commandBuffer->error(), "Metal Canny hysteresis failed");
            AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
        }
    };

    dispatch("canny_hysteresis_init_char", [&](MTL::ComputeCommandEncoder* e) {
        e->setBuffer(strongBuffer.get(), 0, 0);
        e->setBuffer(outputBuffer.get(), 0, 1);
        e->setBytes(&params, sizeof(params), 2);
    });

    auto* changed = static_cast<uint32_t*>(changedBuffer->contents());
    do {
        *changed = 0;
        dispatch("canny_hysteresis_step_char",
                 [&](MTL::ComputeCommandEncoder* e) {
                     e->setBuffer(weakBuffer.get(), 0, 0);
                     e->setBuffer(outputBuffer.get(), 0, 1);
                     e->setBuffer(changedBuffer.get(), 0, 2);
                     e->setBytes(&params, sizeof(params), 3);
                 });
    } while (*changed != 0);

    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalRegions(void* output, const size_t outputBytes,
                        const af::dim4& dims, const af::dim4& outputStrides,
                        const void* input, const size_t inputBytes,
                        const af::dim4& inputStrides,
                        const af_connectivity connectivity) {
    RegionsParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.connectivity = static_cast<uint32_t>(connectivity);
    launchSingleInputKernel(output, outputBytes, dims, input, inputBytes,
                            &params, sizeof(params), "regions_float",
                            "connected components");
}

void launchMetalAnisotropicDiffusion(void* inout, const size_t bytes,
                                     const af::dim4& dims,
                                     const af::dim4& strides, const float dt,
                                     const float mct,
                                     const af_flux_function flux,
                                     const bool curvature) {
    DiffusionParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]    = static_cast<uint64_t>(dims[i]);
        params.strides[i] = static_cast<uint64_t>(strides[i]);
    }
    params.dt        = dt;
    params.mct       = mct;
    params.flux      = static_cast<uint32_t>(flux);
    params.curvature = curvature;
    launchSingleInputKernel(inout, bytes, dims, inout, bytes, &params,
                            sizeof(params), "anisotropic_diffusion_float",
                            "anisotropic diffusion");
}

bool supportsMetalIndex(const af_dtype type) noexcept {
    return indexFunctionName(type) != nullptr;
}

void launchMetalIndex(void* output, const size_t outputBytes,
                      const af::dim4& outputDims, const af::dim4& outputStrides,
                      const void* input, const size_t inputBytes,
                      const af::dim4& inputDims, const af::dim4& inputStrides,
                      const af::dim4& offsets,
                      const std::vector<af_seq>& sequences,
                      const af_dtype type) {
    IndexParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        params.offsets[i]       = static_cast<int64_t>(offsets[i]);
        params.steps[i]         = static_cast<int64_t>(sequences[i].step);
    }
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params), indexFunctionName(type),
                            "index");
}

bool supportsMetalAssign(const af_dtype type) noexcept {
    return assignFunctionName(type) != nullptr;
}

void launchMetalAssign(void* output, const size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, const void* rhs,
                       const size_t rhsBytes, const af::dim4& rhsDims,
                       const af::dim4& rhsStrides, const af::dim4& offsets,
                       const af_dtype type) {
    AssignParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.rhsDims[i]       = static_cast<uint64_t>(rhsDims[i]);
        params.rhsStrides[i]    = static_cast<uint64_t>(rhsStrides[i]);
        params.offsets[i]       = static_cast<int64_t>(offsets[i]);
    }
    launchTwoInputKernel(output, outputBytes, output, outputBytes, rhs,
                         rhsBytes, &params, sizeof(params),
                         static_cast<size_t>(outputDims.elements()),
                         assignFunctionName(type), "assign");
}

bool supportsMetalTransform(const af_dtype type, const af_interp_type method,
                            const af::dim4& inputDims,
                            const af::dim4& transformDims) noexcept {
    return transformFunctionName(type) != nullptr &&
           (method == AF_INTERP_NEAREST || method == AF_INTERP_LOWER) &&
           inputDims[2] == 1 && inputDims[3] == 1 && transformDims[2] == 1 &&
           transformDims[3] == 1;
}

void launchMetalTransform(void* output, const size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, const void* input,
                          const size_t inputBytes, const af::dim4& inputDims,
                          const af::dim4& inputStrides, const void* transform,
                          const size_t transformBytes,
                          const af_interp_type method, const bool inverse,
                          const bool perspective, const af_dtype type) {
    TransformParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.method      = static_cast<uint32_t>(method);
    params.inverse     = inverse;
    params.perspective = perspective;
    launchTwoInputKernel(output, outputBytes, input, inputBytes, transform,
                         transformBytes, &params, sizeof(params),
                         static_cast<size_t>(outputDims.elements()),
                         transformFunctionName(type), "transform");
}

bool supportsMetalSort(const af_dtype type) noexcept {
    return sortFunctionName(type) != nullptr;
}

void launchMetalSort(void* inout, const size_t bytes, const af::dim4& dims,
                     const af::dim4& strides, const bool ascending,
                     const af_dtype type) {
    SortParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]    = static_cast<uint64_t>(dims[i]);
        params.strides[i] = static_cast<uint64_t>(strides[i]);
    }
    params.ascending = ascending;
    launchSingleInputKernel(inout, bytes, dims, inout, bytes, &params,
                            sizeof(params), sortFunctionName(type), "sort");
}

bool supportsMetalSortByKey(const af_dtype type) noexcept {
    return sortByKeyFunctionName(type) != nullptr;
}

void launchMetalSortByKey(void* keys, const size_t keyBytes,
                          const af::dim4& keyDims, const af::dim4& keyStrides,
                          void* values, const size_t valueBytes,
                          const af::dim4& valueStrides, const bool ascending,
                          const af_dtype type) {
    SortByKeyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]         = static_cast<uint64_t>(keyDims[i]);
        params.keyStrides[i]   = static_cast<uint64_t>(keyStrides[i]);
        params.valueStrides[i] = static_cast<uint64_t>(valueStrides[i]);
    }
    params.ascending = ascending;

    MetalRuntime& runtime = metalRuntime();
    auto inputKeys        = NS::TransferPtr(runtime.getDevice()->newBuffer(
        keys, keyBytes, MTL::ResourceStorageModeShared));
    auto inputValues      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        values, valueBytes, MTL::ResourceStorageModeShared));
    auto outputKeys       = NS::TransferPtr(runtime.getDevice()->newBuffer(
        keyBytes, MTL::ResourceStorageModeShared));
    auto outputValues     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        valueBytes, MTL::ResourceStorageModeShared));
    if (!inputKeys || !inputValues || !outputKeys || !outputValues)
        AF_ERROR("Could not allocate Metal sort-by-key buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline(sortByKeyFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputKeys.get(), 0, 0);
    encoder->setBuffer(inputValues.get(), 0, 1);
    encoder->setBuffer(outputKeys.get(), 0, 2);
    encoder->setBuffer(outputValues.get(), 0, 3);
    encoder->setBytes(&params, sizeof(params), 4);
    const size_t vectors =
        static_cast<size_t>(keyDims[1] * keyDims[2] * keyDims[3]);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(vectors, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal sort-by-key dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(keys, outputKeys->contents(), keyBytes);
    std::memcpy(values, outputValues->contents(), valueBytes);
}

bool supportsMetalScan(const af_dtype type) noexcept {
    return scanFunctionName(type) != nullptr;
}

void launchMetalScan(void* output, const size_t outputBytes,
                     const af::dim4& dims, const af::dim4& outputStrides,
                     const void* input, const size_t inputBytes,
                     const af::dim4& inputStrides, const int dimension,
                     const af_dtype type) {
    ScanParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.dimension = static_cast<uint32_t>(dimension);
    launchSingleInputKernel(output, outputBytes, dims, input, inputBytes,
                            &params, sizeof(params), scanFunctionName(type),
                            "inclusive scan");
}

bool supportsMetalScanByKey(const af_dtype keyType,
                            const af_dtype valueType) noexcept {
    return scanByKeyFunctionName(keyType, valueType) != nullptr;
}

void launchMetalScanByKey(void* output, const size_t outputBytes,
                          const af::dim4& dims, const af::dim4& outputStrides,
                          const void* keys, const size_t keyBytes,
                          const af::dim4& keyStrides, const void* input,
                          const size_t inputBytes, const af::dim4& inputStrides,
                          const int dimension, const af_dtype keyType,
                          const af_dtype valueType) {
    ScanByKeyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.keyStrides[i]    = static_cast<uint64_t>(keyStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.dimension = static_cast<uint32_t>(dimension);
    launchTwoInputKernel(
        output, outputBytes, keys, keyBytes, input, inputBytes, &params,
        sizeof(params), static_cast<size_t>(dims.elements()),
        scanByKeyFunctionName(keyType, valueType), "inclusive scan-by-key");
}

void launchMetalMeanFloat(void* output, const size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, const void* input,
                          const size_t inputBytes, const af::dim4& inputDims,
                          const af::dim4& inputStrides, const int dimension) {
    MeanParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.dimension = static_cast<uint32_t>(dimension);
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params), "mean_dim_float", "mean");
}

void launchMetalApprox1Float(Param<float> output, CParam<float> input,
                             CParam<float> positions, const int dimension,
                             const float begin, const float step,
                             const float offGrid, const af_interp_type method) {
    ApproxParams p{};
    size_t inputElements = 1, positionElements = 1;
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(output.dims(i));
        p.outputStrides[i] = uint64_t(output.strides(i));
        p.inputDims[i]     = uint64_t(input.dims(i));
        p.inputStrides[i]  = uint64_t(input.strides(i));
        p.xDims[i]         = uint64_t(positions.dims(i));
        p.xStrides[i]      = uint64_t(positions.strides(i));
        inputElements += size_t(input.dims(i) - 1) * size_t(input.strides(i));
        positionElements +=
            size_t(positions.dims(i) - 1) * size_t(positions.strides(i));
    }
    p.xDimension = uint32_t(dimension);
    p.method     = method == AF_INTERP_LOWER;
    p.xBegin     = begin;
    p.xStep      = step;
    p.offGrid    = offGrid;
    launchTwoInputKernel(
        output.get(), size_t(output.dims().elements()) * sizeof(float),
        input.get(), inputElements * sizeof(float), positions.get(),
        positionElements * sizeof(float), &p, sizeof(p),
        size_t(output.dims().elements()),
        method == AF_INTERP_LINEAR ? "interp1_linear_float" : "approx1_float",
        "approx1");
}

void launchMetalApprox2Float(Param<float> output, CParam<float> input,
                             CParam<float> x, const int xDimension,
                             const float xBegin, const float xStep,
                             CParam<float> y, const int yDimension,
                             const float yBegin, const float yStep,
                             const float offGrid, const af_interp_type method) {
    ApproxParams p{};
    size_t inputElements = 1, xElements = 1, yElements = 1;
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(output.dims(i));
        p.outputStrides[i] = uint64_t(output.strides(i));
        p.inputDims[i]     = uint64_t(input.dims(i));
        p.inputStrides[i]  = uint64_t(input.strides(i));
        p.xDims[i]         = uint64_t(x.dims(i));
        p.xStrides[i]      = uint64_t(x.strides(i));
        p.yStrides[i]      = uint64_t(y.strides(i));
        inputElements += size_t(input.dims(i) - 1) * size_t(input.strides(i));
        xElements += size_t(x.dims(i) - 1) * size_t(x.strides(i));
        yElements += size_t(y.dims(i) - 1) * size_t(y.strides(i));
    }
    p.xDimension = uint32_t(xDimension);
    p.yDimension = uint32_t(yDimension);
    p.method     = method == AF_INTERP_LOWER;
    p.xBegin     = xBegin;
    p.xStep      = xStep;
    p.yBegin     = yBegin;
    p.yStep      = yStep;
    p.offGrid    = offGrid;
    launchThreeInputKernel(
        output.get(), size_t(output.dims().elements()) * sizeof(float),
        input.get(), inputElements * sizeof(float), x.get(),
        xElements * sizeof(float), y.get(), yElements * sizeof(float), &p,
        sizeof(p), size_t(output.dims().elements()),
        (method == AF_INTERP_LINEAR || method == AF_INTERP_BILINEAR)
            ? "interp2_linear_float"
            : "approx2_float",
        "approx2");
}

void reduceAddMetal(Param<float> output, CParam<float> input,
                    const int dimension, const bool changeNan,
                    const double nanValue) {
    ReduceParams p{};
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(output.dims(i));
        p.outputStrides[i] = uint64_t(output.strides(i));
        p.inputDims[i]     = uint64_t(input.dims(i));
        p.inputStrides[i]  = uint64_t(input.strides(i));
        inputElements += size_t(input.dims(i) - 1) * size_t(input.strides(i));
    }
    p.dimension = uint32_t(dimension);
    p.changeNan = changeNan;
    p.nanValue  = float(nanValue);
    launchSingleInputKernel(
        output.get(), size_t(output.dims().elements()) * sizeof(float),
        output.dims(), input.get(), inputElements * sizeof(float), &p,
        sizeof(p), "reduce_add_float", "sum reduction");
}

void ireduceMetal(Param<float> output, Param<uint> locations,
                  CParam<float> input, const int dimension, const bool isMax) {
    IReduceParams p{};
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(output.dims(i));
        p.outputStrides[i] = uint64_t(output.strides(i));
        p.inputDims[i]     = uint64_t(input.dims(i));
        p.inputStrides[i]  = uint64_t(input.strides(i));
        inputElements += size_t(input.dims(i) - 1) * size_t(input.strides(i));
    }
    p.dimension                 = uint32_t(dimension);
    const size_t outputElements = size_t(output.dims().elements());
    MetalRuntime& runtime       = metalRuntime();
    auto inputBuffer    = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input.get(), inputElements * sizeof(float),
        MTL::ResourceStorageModeShared));
    auto outputBuffer   = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputElements * sizeof(float), MTL::ResourceStorageModeShared));
    auto locationBuffer = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputElements * sizeof(uint), MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer || !locationBuffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline =
        runtime.pipeline(isMax ? "ireduce_max_float" : "ireduce_min_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBuffer(locationBuffer.get(), 0, 2);
    encoder->setBytes(&p, sizeof(p), 3);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(outputElements, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal indexed reduction failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output.get(), outputBuffer->contents(),
                outputElements * sizeof(float));
    std::memcpy(locations.get(), locationBuffer->contents(),
                outputElements * sizeof(uint));
}

void convolveMetal(Param<float> output, CParam<float> signal,
                   CParam<float> filter, const int rank, const bool expand) {
    ConvolveParams p{};
    size_t signalElements = 1, filterElements = 1;
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(output.dims(i));
        p.outputStrides[i] = uint64_t(output.strides(i));
        p.signalDims[i]    = uint64_t(signal.dims(i));
        p.signalStrides[i] = uint64_t(signal.strides(i));
        p.filterDims[i]    = uint64_t(filter.dims(i));
        p.filterStrides[i] = uint64_t(filter.strides(i));
        signalElements +=
            size_t(signal.dims(i) - 1) * size_t(signal.strides(i));
        filterElements +=
            size_t(filter.dims(i) - 1) * size_t(filter.strides(i));
    }
    p.rank   = uint32_t(rank);
    p.expand = expand;
    launchTwoInputKernel(
        output.get(), size_t(output.dims().elements()) * sizeof(float),
        signal.get(), signalElements * sizeof(float), filter.get(),
        filterElements * sizeof(float), &p, sizeof(p),
        size_t(output.dims().elements()), "convolve_float", "convolution");
}

void fastLocateMetal(CParam<float> input, Param<float> scoreImage,
                     Param<float> x, Param<float> y, Param<float> scores,
                     unsigned* count, const float threshold,
                     const unsigned arcLength, const bool nonmax,
                     const unsigned maxFeatures, const unsigned edge) {
    FastParams p{uint64_t(input.dims(0)),
                 uint64_t(input.dims(1)),
                 threshold,
                 arcLength,
                 nonmax,
                 maxFeatures,
                 edge};
    const size_t inputBytes   = size_t(input.dims().elements()) * sizeof(float);
    const size_t featureBytes = size_t(maxFeatures) * sizeof(float);
    const size_t scoreImageBytes =
        nonmax ? size_t(scoreImage.dims().elements()) * sizeof(float)
               : sizeof(float);
    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input.get(), inputBytes, MTL::ResourceStorageModeShared));
    auto scoreImageBuffer =
        nonmax ? NS::TransferPtr(runtime.getDevice()->newBuffer(
                     scoreImage.get(), scoreImageBytes,
                     MTL::ResourceStorageModeShared))
               : NS::TransferPtr(runtime.getDevice()->newBuffer(
                     scoreImageBytes, MTL::ResourceStorageModeShared));
    auto xBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        featureBytes, MTL::ResourceStorageModeShared));
    auto yBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        featureBytes, MTL::ResourceStorageModeShared));
    auto scoreBuffer = NS::TransferPtr(runtime.getDevice()->newBuffer(
        featureBytes, MTL::ResourceStorageModeShared));
    auto countBuffer = NS::TransferPtr(runtime.getDevice()->newBuffer(
        sizeof(uint32_t), MTL::ResourceStorageModeShared));
    if (!inputBuffer || !scoreImageBuffer || !xBuffer || !yBuffer ||
        !scoreBuffer || !countBuffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    *static_cast<uint32_t*>(countBuffer->contents()) = 0;
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline("fast_locate_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(scoreImageBuffer.get(), 0, 1);
    encoder->setBuffer(xBuffer.get(), 0, 2);
    encoder->setBuffer(yBuffer.get(), 0, 3);
    encoder->setBuffer(scoreBuffer.get(), 0, 4);
    encoder->setBuffer(countBuffer.get(), 0, 5);
    encoder->setBytes(&p, sizeof(p), 6);
    encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal FAST dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    *count                 = *static_cast<uint32_t*>(countBuffer->contents());
    const size_t populated = std::min<size_t>(*count, maxFeatures);
    std::memcpy(x.get(), xBuffer->contents(), populated * sizeof(float));
    std::memcpy(y.get(), yBuffer->contents(), populated * sizeof(float));
    std::memcpy(scores.get(), scoreBuffer->contents(),
                populated * sizeof(float));
    if (nonmax)
        std::memcpy(scoreImage.get(), scoreImageBuffer->contents(),
                    scoreImageBytes);
}

void harrisSecondOrderMetal(Param<float> ixx, Param<float> ixy,
                            Param<float> iyy, CParam<float> ix,
                            CParam<float> iy) {
    const size_t elements = size_t(ix.dims().elements());
    const size_t bytes    = elements * sizeof(float);
    HarrisParams p{elements, 0, 0, 0, 0};
    MetalRuntime& runtime = metalRuntime();
    auto ixBuffer         = NS::TransferPtr(runtime.getDevice()->newBuffer(
        ix.get(), bytes, MTL::ResourceStorageModeShared));
    auto iyBuffer         = NS::TransferPtr(runtime.getDevice()->newBuffer(
        iy.get(), bytes, MTL::ResourceStorageModeShared));
    auto ixxBuffer        = NS::TransferPtr(
        runtime.getDevice()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    auto ixyBuffer = NS::TransferPtr(
        runtime.getDevice()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    auto iyyBuffer = NS::TransferPtr(
        runtime.getDevice()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    if (!ixBuffer || !iyBuffer || !ixxBuffer || !ixyBuffer || !iyyBuffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline("harris_second_order_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(ixBuffer.get(), 0, 0);
    encoder->setBuffer(iyBuffer.get(), 0, 1);
    encoder->setBuffer(ixxBuffer.get(), 0, 2);
    encoder->setBuffer(ixyBuffer.get(), 0, 3);
    encoder->setBuffer(iyyBuffer.get(), 0, 4);
    encoder->setBytes(&p, sizeof(p), 5);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(elements, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal Harris derivative dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(ixx.get(), ixxBuffer->contents(), bytes);
    std::memcpy(ixy.get(), ixyBuffer->contents(), bytes);
    std::memcpy(iyy.get(), iyyBuffer->contents(), bytes);
}

void harrisResponseMetal(Param<float> output, const unsigned rows,
                         const unsigned columns, CParam<float> ixx,
                         CParam<float> ixy, CParam<float> iyy, const float k,
                         const unsigned border) {
    const size_t elements = size_t(output.dims().elements());
    HarrisParams p{elements, rows, columns, border, k};
    const size_t bytes = elements * sizeof(float);
    launchThreeInputKernel(output.get(), bytes, ixx.get(), bytes, ixy.get(),
                           bytes, iyy.get(), bytes, &p, sizeof(p), elements,
                           "harris_response_float", "Harris response");
}

void susanResponseMetal(Param<float> output, CParam<float> input,
                        const unsigned rows, const unsigned columns,
                        const unsigned radius, const float differenceThreshold,
                        const float geometricThreshold, const unsigned border) {
    SusanParams p{
        rows, columns, radius, border, differenceThreshold, geometricThreshold};
    const size_t bytes = size_t(input.dims().elements()) * sizeof(float);
    launchSingleInputKernel(output.get(), bytes, output.dims(), input.get(),
                            bytes, &p, sizeof(p), "susan_response_float",
                            "SUSAN response");
}

void randomUniformMetal(Param<float> output, const unsigned long long seed,
                        const unsigned long long counter,
                        const af_random_engine_type type) {
    const size_t elements = size_t(output.dims().elements());
    const size_t bytes    = elements * sizeof(float);
    RandomParams p{elements, seed, counter, uint32_t(type)};
    MetalRuntime& runtime = metalRuntime();
    auto outputBuffer     = NS::TransferPtr(
        runtime.getDevice()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    if (!outputBuffer)
        AF_ERROR("Could not allocate a Metal buffer", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline("random_uniform_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(outputBuffer.get(), 0, 0);
    encoder->setBytes(&p, sizeof(p), 1);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(elements, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal random dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output.get(), outputBuffer->contents(), bytes);
}

void randomMersenneInitMetal(Param<uint> state, CParam<uint> table,
                             const unsigned long long seed) {
    MersenneInitParams p{seed};
    launchSingleInputKernel(
        state.get(), size_t(state.dims().elements()) * sizeof(uint),
        state.dims(), table.get(),
        size_t(table.dims().elements()) * sizeof(uint), &p, sizeof(p),
        "random_mersenne_init", "Mersenne initialization");
}

void sparseToDenseMetal(Param<float> output, CParam<float> values,
                        CParam<int> rows, CParam<int> columns, const bool csr) {
    const size_t outputElements = size_t(output.dims().elements());
    const size_t nonzeros       = size_t(values.dims().elements());
    SparseParams p{uint32_t(output.dims(0)), uint32_t(output.dims(1)),
                   uint32_t(output.strides(1)), uint32_t(nonzeros)};
    MetalRuntime& runtime = metalRuntime();
    auto valuesBuffer     = NS::TransferPtr(
        runtime.getDevice()->newBuffer(values.get(), nonzeros * sizeof(float),
                                           MTL::ResourceStorageModeShared));
    auto rowsBuffer    = NS::TransferPtr(runtime.getDevice()->newBuffer(
        rows.get(), size_t(rows.dims().elements()) * sizeof(int),
        MTL::ResourceStorageModeShared));
    auto columnsBuffer = NS::TransferPtr(runtime.getDevice()->newBuffer(
        columns.get(), size_t(columns.dims().elements()) * sizeof(int),
        MTL::ResourceStorageModeShared));
    auto outputBuffer  = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputElements * sizeof(float), MTL::ResourceStorageModeShared));
    if (!valuesBuffer || !rowsBuffer || !columnsBuffer || !outputBuffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* zeroPipeline = runtime.pipeline("sparse_zero_float");
    encoder->setComputePipelineState(zeroPipeline);
    encoder->setBuffer(outputBuffer.get(), 0, 0);
    encoder->setBytes(&p, sizeof(p), 1);
    auto width = std::min<NS::UInteger>(
        256, zeroPipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(outputElements, 1, 1),
                             MTL::Size(width, 1, 1));
    auto* scatterPipeline = runtime.pipeline(csr ? "sparse_csr_to_dense_float"
                                                 : "sparse_coo_to_dense_float");
    encoder->setComputePipelineState(scatterPipeline);
    encoder->setBuffer(valuesBuffer.get(), 0, 0);
    encoder->setBuffer(rowsBuffer.get(), 0, 1);
    encoder->setBuffer(columnsBuffer.get(), 0, 2);
    encoder->setBuffer(outputBuffer.get(), 0, 3);
    encoder->setBytes(&p, sizeof(p), 4);
    const size_t scatterElements = csr ? size_t(output.dims(0)) : nonzeros;
    width                        = std::min<NS::UInteger>(
        256, scatterPipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(scatterElements, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal sparse conversion failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output.get(), outputBuffer->contents(),
                outputElements * sizeof(float));
}

void sparseArithMetal(Param<float> values, CParam<int> rows,
                      CParam<int> columns, CParam<float> rhs, const bool csr,
                      const bool reverse, const unsigned operation) {
    const size_t nonzeros    = size_t(values.dims().elements());
    const size_t valueBytes  = nonzeros * sizeof(float);
    const size_t rhsElements = size_t(rhs.dims().elements());
    SparseArithParams p{uint32_t(nonzeros),       uint32_t(rhs.dims(0)),
                        uint32_t(rhs.strides(1)), uint32_t(csr),
                        uint32_t(reverse),        operation};
    MetalRuntime& runtime = metalRuntime();
    auto valuesBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        values.get(), valueBytes, MTL::ResourceStorageModeShared));
    auto rowsBuffer       = NS::TransferPtr(runtime.getDevice()->newBuffer(
        rows.get(), size_t(rows.dims().elements()) * sizeof(int),
        MTL::ResourceStorageModeShared));
    auto columnsBuffer    = NS::TransferPtr(runtime.getDevice()->newBuffer(
        columns.get(), size_t(columns.dims().elements()) * sizeof(int),
        MTL::ResourceStorageModeShared));
    auto rhsBuffer        = NS::TransferPtr(
        runtime.getDevice()->newBuffer(rhs.get(), rhsElements * sizeof(float),
                                              MTL::ResourceStorageModeShared));
    if (!valuesBuffer || !rowsBuffer || !columnsBuffer || !rhsBuffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline("sparse_arith_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(valuesBuffer.get(), 0, 0);
    encoder->setBuffer(rowsBuffer.get(), 0, 1);
    encoder->setBuffer(columnsBuffer.get(), 0, 2);
    encoder->setBuffer(rhsBuffer.get(), 0, 3);
    encoder->setBytes(&p, sizeof(p), 4);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(nonzeros, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal sparse arithmetic failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(values.get(), valuesBuffer->contents(), valueBytes);
}

void fftConvolveMultiplyMetal(Param<float> packed, const af::dim4& signalDims,
                              const af::dim4& signalStrides,
                              const af::dim4& filterDims,
                              const af::dim4& filterStrides,
                              const AF_BATCH_KIND kind, const dim_t offset) {
    FFTConvolveParams p{};
    const af::dim4& outputDims = kind == AF_BATCH_RHS ? filterDims : signalDims;
    const af::dim4& outputStrides =
        kind == AF_BATCH_RHS ? filterStrides : signalStrides;
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(outputDims[i]);
        p.outputStrides[i] = uint64_t(outputStrides[i]);
        p.signalDims[i]    = uint64_t(signalDims[i]);
        p.signalStrides[i] = uint64_t(signalStrides[i]);
        p.filterDims[i]    = uint64_t(filterDims[i]);
        p.filterStrides[i] = uint64_t(filterStrides[i]);
    }
    p.offset           = uint64_t(offset);
    p.kind             = uint32_t(kind);
    const size_t bytes = size_t(packed.dims().elements()) * sizeof(float);
    const size_t complexElements = size_t(outputDims.elements()) / 2;
    MetalRuntime& runtime        = metalRuntime();
    auto buffer = NS::TransferPtr(runtime.getDevice()->newBuffer(
        packed.get(), bytes, MTL::ResourceStorageModeShared));
    if (!buffer) AF_ERROR("Could not allocate a Metal buffer", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = runtime.pipeline("fftconvolve_multiply_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(buffer.get(), 0, 0);
    encoder->setBytes(&p, sizeof(p), 1);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(complexElements, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal FFT convolution multiply failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(packed.get(), buffer->contents(), bytes);
}

void orbCentroidMetal(const float* x, const float* y, float* orientation,
                      const unsigned features, CParam<float> image,
                      const unsigned patchSize) {
    OrbParams p{features, uint32_t(image.dims(0)), uint32_t(image.dims(1)),
                patchSize};
    const size_t featureBytes = size_t(features) * sizeof(float);
    const size_t imageBytes   = size_t(image.dims().elements()) * sizeof(float);
    launchThreeInputKernel(orientation, featureBytes, x, featureBytes, y,
                           featureBytes, image.get(), imageBytes, &p, sizeof(p),
                           features, "orb_centroid_float",
                           "ORB centroid orientation");
}

void siftSubtractMetal(Array<float>& output, const Array<float>& first,
                       const Array<float>& second) {
    const size_t elements = size_t(output.elements());
    const size_t bytes    = elements * sizeof(float);
    SiftParams p{elements};
    launchTwoInputKernel(output.get(), bytes, first.get(), bytes, second.get(),
                         bytes, &p, sizeof(p), elements, "sift_subtract_float",
                         "SIFT pyramid subtraction");
}

void arrayAddMetal(Param<float> output, CParam<float> left,
                   CParam<float> right) {
    ArrayAddParams p{};
    size_t leftElements = 1, rightElements = 1;
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(output.dims(i));
        p.outputStrides[i] = uint64_t(output.strides(i));
        p.leftDims[i]      = uint64_t(left.dims(i));
        p.leftStrides[i]   = uint64_t(left.strides(i));
        p.rightDims[i]     = uint64_t(right.dims(i));
        p.rightStrides[i]  = uint64_t(right.strides(i));
        leftElements += size_t(left.dims(i) - 1) * size_t(left.strides(i));
        rightElements += size_t(right.dims(i) - 1) * size_t(right.strides(i));
    }
    launchTwoInputKernel(
        output.get(), size_t(output.dims().elements()) * sizeof(float),
        left.get(), leftElements * sizeof(float), right.get(),
        rightElements * sizeof(float), &p, sizeof(p),
        size_t(output.dims().elements()), "array_add_float", "array add");
}

void exampleFunctionMetal(Param<float> output, CParam<float> left,
                          CParam<float> right) {
    ExampleFunctionParams p{};
    size_t leftElements = 1, rightElements = 1;
    for (int i = 0; i < 4; ++i) {
        p.dims[i]          = uint64_t(output.dims(i));
        p.outputStrides[i] = uint64_t(output.strides(i));
        p.leftStrides[i]   = uint64_t(left.strides(i));
        p.rightStrides[i]  = uint64_t(right.strides(i));
        leftElements += size_t(left.dims(i) - 1) * size_t(left.strides(i));
        rightElements += size_t(right.dims(i) - 1) * size_t(right.strides(i));
    }
    const size_t elements = size_t(output.dims().elements());
    launchTwoInputKernel(output.get(), elements * sizeof(float), left.get(),
                         leftElements * sizeof(float), right.get(),
                         rightElements * sizeof(float), &p, sizeof(p), elements,
                         "example_function_float", "example function");
}

void launchMetalIota(void* output, const size_t bytes, const af::dim4& dims,
                     const af::dim4& strides, const af::dim4& sourceDims,
                     const af_dtype type) {
    if (bytes == 0) { return; }

    IotaParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]       = static_cast<uint64_t>(dims[i]);
        params.strides[i]    = static_cast<uint64_t>(strides[i]);
        params.sourceDims[i] = static_cast<uint64_t>(sourceDims[i]);
    }

    MetalRuntime& runtime = metalRuntime();
    auto buffer           = NS::TransferPtr(
        runtime.getDevice()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    if (!buffer) {
        AF_ERROR("Could not allocate a Metal iota buffer", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(iotaFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(buffer.get(), 0, 0);
    encoder->setBytes(&params, sizeof(params), 1);
    const size_t total = static_cast<size_t>(dims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal iota dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, buffer->contents(), bytes);
}

void launchMetalIdentity(void* output, const size_t bytes, const af::dim4& dims,
                         const af::dim4& strides, const af_dtype type) {
    if (bytes == 0) { return; }

    IdentityParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]    = static_cast<uint64_t>(dims[i]);
        params.strides[i] = static_cast<uint64_t>(strides[i]);
    }

    MetalRuntime& runtime = metalRuntime();
    auto buffer           = NS::TransferPtr(
        runtime.getDevice()->newBuffer(bytes, MTL::ResourceStorageModeShared));
    if (!buffer) {
        AF_ERROR("Could not allocate a Metal identity buffer", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(identityFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(buffer.get(), 0, 0);
    encoder->setBytes(&params, sizeof(params), 1);
    const size_t total = static_cast<size_t>(dims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal identity dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, buffer->contents(), bytes);
}

void launchMetalTile(void* output, const size_t outputBytes,
                     const af::dim4& outputDims, const af::dim4& outputStrides,
                     const void* input, const size_t inputBytes,
                     const af::dim4& inputDims, const af::dim4& inputStrides,
                     const af_dtype type) {
    if (outputBytes == 0) { return; }

    TileParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal tile buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(tileFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal tile dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalShift(void* output, const size_t outputBytes,
                      const af::dim4& outputDims, const af::dim4& outputStrides,
                      const void* input, const size_t inputBytes,
                      const af::dim4& inputStrides, const af::dim4& shifts,
                      const af_dtype type) {
    if (outputBytes == 0) { return; }

    ShiftParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        const int64_t dim       = static_cast<int64_t>(outputDims[i]);
        const int64_t shift     = static_cast<int64_t>(shifts[i]);
        params.shifts[i]        = static_cast<uint64_t>(
            -(shift % dim) + dim * static_cast<int64_t>(shift > 0));
    }

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal shift buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(shiftFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal shift dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalReorder(void* output, const size_t outputBytes,
                        const af::dim4& outputDims,
                        const af::dim4& outputStrides, const void* input,
                        const size_t inputBytes, const af::dim4& inputStrides,
                        const af::dim4& reorderDims, const af_dtype type) {
    if (outputBytes == 0) { return; }

    ReorderParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        params.reorderDims[i]   = static_cast<uint64_t>(reorderDims[i]);
    }

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal reorder buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(reorderFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal reorder dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalSelect(
    void* output, const size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, const void* condition,
    const size_t conditionBytes, const af::dim4& conditionDims,
    const af::dim4& conditionStrides, const void* a, const size_t aBytes,
    const af::dim4& aDims, const af::dim4& aStrides, const void* b,
    const size_t bBytes, const af::dim4& bDims, const af::dim4& bStrides,
    const void* scalar, const bool flip, const af_dtype type) {
    if (outputBytes == 0) { return; }

    SelectParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]       = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i]    = static_cast<uint64_t>(outputStrides[i]);
        params.aDims[i]            = static_cast<uint64_t>(aDims[i]);
        params.aStrides[i]         = static_cast<uint64_t>(aStrides[i]);
        params.bDims[i]            = static_cast<uint64_t>(bDims[i]);
        params.bStrides[i]         = static_cast<uint64_t>(bStrides[i]);
        params.conditionDims[i]    = static_cast<uint64_t>(conditionDims[i]);
        params.conditionStrides[i] = static_cast<uint64_t>(conditionStrides[i]);
    }
    params.flip = flip;

    MetalRuntime& runtime = metalRuntime();
    auto conditionBuffer  = NS::TransferPtr(runtime.getDevice()->newBuffer(
        condition, conditionBytes, MTL::ResourceStorageModeShared));
    auto aBuffer          = NS::TransferPtr(runtime.getDevice()->newBuffer(
        a, aBytes, MTL::ResourceStorageModeShared));
    NS::SharedPtr<MTL::Buffer> bBuffer;
    if (b) {
        bBuffer = NS::TransferPtr(runtime.getDevice()->newBuffer(
            b, bBytes, MTL::ResourceStorageModeShared));
    }
    auto outputBuffer = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!conditionBuffer || !aBuffer || (b && !bBuffer) || !outputBuffer) {
        AF_ERROR("Could not allocate Metal select buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(selectFunctionName(type, scalar != nullptr));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(conditionBuffer.get(), 0, 0);
    encoder->setBuffer(aBuffer.get(), 0, 1);
    if (scalar) {
        encoder->setBuffer(outputBuffer.get(), 0, 2);
        encoder->setBytes(
            scalar, outputBytes / static_cast<size_t>(outputDims.elements()),
            3);
    } else {
        encoder->setBuffer(bBuffer.get(), 0, 2);
        encoder->setBuffer(outputBuffer.get(), 0, 3);
    }
    encoder->setBytes(&params, sizeof(params), 4);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal select dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalJoinAppend(void* output, const size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, const void* input,
                           const size_t inputBytes, const af::dim4& inputDims,
                           const af::dim4& inputStrides,
                           const af::dim4& outputOffset,
                           const bool preserveOutput, const af_dtype type) {
    if (inputBytes == 0) { return; }

    JoinParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        params.outputOffset[i]  = static_cast<uint64_t>(outputOffset[i]);
    }

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer =
        preserveOutput
            ? NS::TransferPtr(runtime.getDevice()->newBuffer(
                  output, outputBytes, MTL::ResourceStorageModeShared))
            : NS::TransferPtr(runtime.getDevice()->newBuffer(
                  outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal join buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(joinFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    const size_t total = static_cast<size_t>(inputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal join dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalLookup(void* output, const size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, const void* input,
                       const size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, const void* indices,
                       const size_t indexBytes, const unsigned dimension,
                       const af_dtype inputType, const af_dtype indexType) {
    if (outputBytes == 0) { return; }

    LookupParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.dimension = dimension;
    params.indexType = static_cast<uint32_t>(lookupIndexType(indexType));

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto indexBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        indices, indexBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !indexBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal lookup buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(lookupFunctionName(inputType));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(indexBuffer.get(), 0, 1);
    encoder->setBuffer(outputBuffer.get(), 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal lookup dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalDiagonal(void* output, const size_t outputBytes,
                         const af::dim4& outputDims,
                         const af::dim4& outputStrides, const void* input,
                         const size_t inputBytes, const af::dim4& inputDims,
                         const af::dim4& inputStrides, const int diagonal,
                         const af_dtype type, const bool create) {
    if (outputBytes == 0) { return; }

    DiagonalParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.diagonal = diagonal;

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal diagonal buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(diagonalFunctionName(type, create));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal diagonal dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalDiagCreate(void* output, const size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, const void* input,
                           const size_t inputBytes, const af::dim4& inputDims,
                           const af::dim4& inputStrides, const int diagonal,
                           const af_dtype type) {
    launchMetalDiagonal(output, outputBytes, outputDims, outputStrides, input,
                        inputBytes, inputDims, inputStrides, diagonal, type,
                        true);
}

void launchMetalDiagExtract(void* output, const size_t outputBytes,
                            const af::dim4& outputDims,
                            const af::dim4& outputStrides, const void* input,
                            const size_t inputBytes, const af::dim4& inputDims,
                            const af::dim4& inputStrides, const int diagonal,
                            const af_dtype type) {
    launchMetalDiagonal(output, outputBytes, outputDims, outputStrides, input,
                        inputBytes, inputDims, inputStrides, diagonal, type,
                        false);
}

void launchMetalDiff(void* output, const size_t outputBytes,
                     const af::dim4& outputDims, const af::dim4& outputStrides,
                     const void* input, const size_t inputBytes,
                     const af::dim4& inputStrides, const unsigned dimension,
                     const bool secondOrder, const af_dtype type) {
    if (outputBytes == 0) { return; }

    DiffParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.dimension = dimension;

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal diff buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(diffFunctionName(type, secondOrder));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal diff dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalTriangle(void* output, const size_t outputBytes,
                         const af::dim4& outputDims,
                         const af::dim4& outputStrides, const void* input,
                         const size_t inputBytes, const af::dim4& inputStrides,
                         const bool upper, const bool unitDiagonal,
                         const af_dtype type) {
    if (outputBytes == 0) { return; }

    TriangleParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.upper        = upper;
    params.unitDiagonal = unitDiagonal;

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal triangle buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(triangleFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal triangle dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalTranspose(void* output, const size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, const void* input,
                          const size_t inputBytes, const af::dim4& inputStrides,
                          const bool conjugate, const af_dtype type) {
    if (outputBytes == 0) { return; }

    TransposeParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.conjugate = conjugate;

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal transpose buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(transposeFunctionName(type, false));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal transpose dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalTransposeInplace(void* input, const size_t inputBytes,
                                 const af::dim4& dims, const af::dim4& strides,
                                 const bool conjugate, const af_dtype type) {
    if (inputBytes == 0) { return; }

    TransposeInplaceParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]    = static_cast<uint64_t>(dims[i]);
        params.strides[i] = static_cast<uint64_t>(strides[i]);
    }
    params.conjugate = conjugate;

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer) {
        AF_ERROR("Could not allocate a Metal transpose buffer", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(transposeFunctionName(type, true));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBytes(&params, sizeof(params), 1);
    const size_t total = static_cast<size_t>(dims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal inplace transpose dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(input, inputBuffer->contents(), inputBytes);
}

void launchMetalUnwrap(void* output, const size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, const void* input,
                       const size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, const dim_t windowX,
                       const dim_t windowY, const dim_t strideX,
                       const dim_t strideY, const dim_t paddingX,
                       const dim_t paddingY, const dim_t dilationX,
                       const dim_t dilationY, const unsigned columnDimension,
                       const af_dtype type) {
    if (outputBytes == 0) { return; }

    UnwrapParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.windowX         = static_cast<uint64_t>(windowX);
    params.windowY         = static_cast<uint64_t>(windowY);
    params.strideX         = static_cast<uint64_t>(strideX);
    params.strideY         = static_cast<uint64_t>(strideY);
    params.paddingX        = static_cast<int64_t>(paddingX);
    params.paddingY        = static_cast<int64_t>(paddingY);
    params.dilationX       = static_cast<uint64_t>(dilationX);
    params.dilationY       = static_cast<uint64_t>(dilationY);
    params.columnDimension = columnDimension;

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal unwrap buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(unwrapFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal unwrap dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalWrap(void* output, const size_t outputBytes,
                     const af::dim4& outputDims, const af::dim4& outputStrides,
                     const void* input, const size_t inputBytes,
                     const af::dim4& inputDims, const af::dim4& inputStrides,
                     const dim_t windowX, const dim_t windowY,
                     const dim_t strideX, const dim_t strideY,
                     const dim_t paddingX, const dim_t paddingY,
                     const dim_t dilationX, const dim_t dilationY,
                     const unsigned columnDimension, const af_dtype type) {
    if (outputBytes == 0) { return; }

    WrapParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.windowX         = static_cast<uint64_t>(windowX);
    params.windowY         = static_cast<uint64_t>(windowY);
    params.strideX         = static_cast<uint64_t>(strideX);
    params.strideY         = static_cast<uint64_t>(strideY);
    params.paddingX        = static_cast<int64_t>(paddingX);
    params.paddingY        = static_cast<int64_t>(paddingY);
    params.dilationX       = static_cast<uint64_t>(dilationX);
    params.dilationY       = static_cast<uint64_t>(dilationY);
    params.columnDimension = columnDimension;

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto outputBuffer     = NS::TransferPtr(runtime.getDevice()->newBuffer(
        output, outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !outputBuffer) {
        AF_ERROR("Could not allocate Metal wrap buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(wrapFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(outputBuffer.get(), 0, 1);
    encoder->setBytes(&params, sizeof(params), 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal wrap dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(output, outputBuffer->contents(), outputBytes);
}

void launchMetalGradient(void* gradient0, const af::dim4& gradient0Strides,
                         void* gradient1, const af::dim4& gradient1Strides,
                         const size_t outputBytes, const void* input,
                         const size_t inputBytes, const af::dim4& inputDims,
                         const af::dim4& inputStrides, const af_dtype type) {
    if (outputBytes == 0) { return; }

    GradientParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]             = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]     = static_cast<uint64_t>(inputStrides[i]);
        params.gradient0Strides[i] = static_cast<uint64_t>(gradient0Strides[i]);
        params.gradient1Strides[i] = static_cast<uint64_t>(gradient1Strides[i]);
    }

    MetalRuntime& runtime = metalRuntime();
    auto inputBuffer      = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto gradient0Buffer  = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    auto gradient1Buffer  = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !gradient0Buffer || !gradient1Buffer) {
        AF_ERROR("Could not allocate Metal gradient buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(gradientFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(gradient0Buffer.get(), 0, 1);
    encoder->setBuffer(gradient1Buffer.get(), 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    const size_t total = static_cast<size_t>(inputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal gradient dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(gradient0, gradient0Buffer->contents(), outputBytes);
    std::memcpy(gradient1, gradient1Buffer->contents(), outputBytes);
}

void launchMetalSobel(void* derivative0, const af::dim4& derivative0Strides,
                      void* derivative1, const af::dim4& derivative1Strides,
                      const size_t outputBytes, const void* input,
                      const size_t inputBytes, const af::dim4& inputDims,
                      const af::dim4& inputStrides, const af_dtype inputType) {
    if (outputBytes == 0) { return; }

    SobelParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]         = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i] = static_cast<uint64_t>(inputStrides[i]);
        params.derivative0Strides[i] =
            static_cast<uint64_t>(derivative0Strides[i]);
        params.derivative1Strides[i] =
            static_cast<uint64_t>(derivative1Strides[i]);
    }

    MetalRuntime& runtime  = metalRuntime();
    auto inputBuffer       = NS::TransferPtr(runtime.getDevice()->newBuffer(
        input, inputBytes, MTL::ResourceStorageModeShared));
    auto derivative0Buffer = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    auto derivative1Buffer = NS::TransferPtr(runtime.getDevice()->newBuffer(
        outputBytes, MTL::ResourceStorageModeShared));
    if (!inputBuffer || !derivative0Buffer || !derivative1Buffer) {
        AF_ERROR("Could not allocate Metal Sobel buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(runtime.getCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        runtime.pipeline(sobelFunctionName(inputType));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputBuffer.get(), 0, 0);
    encoder->setBuffer(derivative0Buffer.get(), 0, 1);
    encoder->setBuffer(derivative1Buffer.get(), 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    const size_t total = static_cast<size_t>(inputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            description(commandBuffer->error(), "Metal Sobel dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    std::memcpy(derivative0, derivative0Buffer->contents(), outputBytes);
    std::memcpy(derivative1, derivative1Buffer->contents(), outputBytes);
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
