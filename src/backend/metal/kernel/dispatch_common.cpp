/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel/dispatch_common.hpp>

#include <string>

namespace arrayfire {
namespace metal {
namespace kernel {

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

const char* gemmFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "gemm_float";
        case c32: return "gemm_cfloat";
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

const char* luFactorFunctionName(const af_dtype t) {
    return t == f32 ? "lu_factor_float"
                    : t == c32 ? "lu_factor_cfloat"
                               : nullptr;
}

const char* qrFunctionName(const af_dtype t) {
    return t == f32 ? "qr_float" : t == c32 ? "qr_cfloat" : nullptr;
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

const char* sortByKeyFunctionName(const af_dtype keyType,
                                  const af_dtype valueType) {
    const auto suffix = [](const af_dtype type, const bool allowComplex) {
        switch (type) {
            case f32: return "float";
            case c32: return allowComplex ? "cfloat" : nullptr;
            case s32: return "int";
            case u32: return "uint";
            case s64: return "long";
            case u64: return "ulong";
            case s8:
            case b8: return "char";
            case u8: return "uchar";
            case s16: return "short";
            case u16: return "ushort";
            default: return static_cast<const char*>(nullptr);
        }
    };
    const char* key = suffix(keyType, false);
    const char* value = suffix(valueType, true);
    if (!key || !value) return nullptr;
    static thread_local std::string name;
    name = "sort_by_key_";
    name += key;
    name += "_";
    name += value;
    return name.c_str();
}

const char* sortIndexFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "sort_index_float";
        case s32: return "sort_index_int";
        case u32: return "sort_index_uint";
        case s64: return "sort_index_long";
        case u64: return "sort_index_ulong";
        case s8: return "sort_index_char";
        case u8:
        case b8: return "sort_index_uchar";
        case s16: return "sort_index_short";
        case u16: return "sort_index_ushort";
        default: return nullptr;
    }
}

const char* scanFunctionName(const af_dtype inputType,
                             const af_dtype outputType) {
    if (inputType == f32 && outputType == f32) return "scan_float_float";
    if (inputType == c32 && outputType == c32) return "scan_cfloat_cfloat";
    if (inputType == s32 && outputType == s32) return "scan_int_int";
    if (inputType == u32 && outputType == u32) return "scan_uint_uint";
    if (inputType == s64 && outputType == s64) return "scan_long_long";
    if (inputType == u64 && outputType == u64) return "scan_ulong_ulong";
    if (inputType == s8 && outputType == s32) return "scan_char_int";
    if ((inputType == u8 || inputType == b8) && outputType == u32)
        return "scan_uchar_uint";
    if (inputType == b8 && outputType == s32) return "scan_uchar_int";
    if (inputType == s16 && outputType == s32) return "scan_short_int";
    if (inputType == u16 && outputType == u32) return "scan_ushort_uint";
    return nullptr;
}

const char* scanByKeyFunctionName(const af_dtype keyType,
                                  const af_dtype valueType) {
    const char* key = nullptr;
    switch (keyType) {
        case s32:
        case u32: key = "uint"; break;
        case s64:
        case u64: key = "ulong"; break;
        default: return nullptr;
    }
    const char* value = nullptr;
    switch (valueType) {
        case f32: value = "float"; break;
        case c32: value = "cfloat"; break;
        case s32: value = "int"; break;
        case u32: value = "uint"; break;
        case s64: value = "long"; break;
        case u64: value = "ulong"; break;
        default: return nullptr;
    }
    static thread_local std::string name;
    name = "scan_by_key_";
    name += key;
    name += "_";
    name += value;
    return name.c_str();
}

const char* meanFunctionName(const af_dtype inputType,
                             const af_dtype outputType) {
    if (inputType == f32 && outputType == f32) return "mean_float_float";
    if (inputType == c32 && outputType == c32) return "mean_cfloat_cfloat";
    if (inputType == s32 && outputType == f32) return "mean_int_float";
    if (inputType == u32 && outputType == f32) return "mean_uint_float";
    if ((inputType == s8 || inputType == b8) && outputType == f32)
        return "mean_char_float";
    if (inputType == u8 && outputType == f32) return "mean_uchar_float";
    if (inputType == s16 && outputType == f32) return "mean_short_float";
    if (inputType == u16 && outputType == f32) return "mean_ushort_float";
    if (inputType == f16 && outputType == f16) return "mean_half_half";
    if (inputType == f16 && outputType == f32) return "mean_half_float";
    return nullptr;
}

const char* meanWeightedFunctionName(const af_dtype valueType,
                                     const af_dtype weightType) {
    if (weightType != f32) return nullptr;
    switch (valueType) {
        case f32: return "mean_weighted_float";
        case c32: return "mean_weighted_cfloat";
        case f16: return "mean_weighted_half";
        default: return nullptr;
    }
}

const char* reduceFunctionName(const af_dtype inputType,
                               const af_dtype outputType) {
    const auto suffix = [](const af_dtype type) {
        switch (type) {
            case f32: return "float";
            case c32: return "cfloat";
            case s32: return "int";
            case u32: return "uint";
            case s64: return "long";
            case u64: return "ulong";
            case b8: return "bool";
            case s8: return "char";
            case u8: return "uchar";
            case s16: return "short";
            case u16: return "ushort";
            case f16: return "half";
            default: return static_cast<const char*>(nullptr);
        }
    };

    const bool same = inputType == outputType;
    const bool promotedInteger =
        ((inputType == b8 || inputType == s8 || inputType == s16) &&
         outputType == s32) ||
        ((inputType == u8 || inputType == u16) && outputType == u32);
    const bool promotedFloat =
        outputType == f32 &&
        (inputType == s32 || inputType == u32 || inputType == b8 ||
         inputType == s8 || inputType == u8 || inputType == s16 ||
         inputType == u16 || inputType == f16);
    const bool countOutput =
        outputType == u32 && suffix(inputType) != nullptr;
    const bool logicalOutput =
        outputType == b8 && suffix(inputType) != nullptr;
    if (!same && !promotedInteger && !promotedFloat && !countOutput &&
        !logicalOutput)
        return nullptr;

    const char* input = suffix(inputType);
    const char* output = suffix(outputType);
    if (!input || !output) return nullptr;
    static thread_local std::string name;
    name = "reduce_";
    name += input;
    name += "_";
    name += output;
    return name.c_str();
}

const char* reduceByKeyCompactFunctionName(const af_dtype keyType) {
    switch (keyType) {
        case s32: return "reduce_by_key_compact_int";
        case u32: return "reduce_by_key_compact_uint";
        default: return nullptr;
    }
}

const char* reduceByKeyFunctionName(const af_dtype keyType,
                                    const af_dtype inputType,
                                    const af_dtype outputType) {
    const char* reduction = reduceFunctionName(inputType, outputType);
    if (!reduction) return nullptr;
    const char* key = keyType == s32 ? "int" : keyType == u32 ? "uint" : nullptr;
    if (!key) return nullptr;
    static thread_local std::string name;
    name = "reduce_by_key_";
    name += key;
    name += reduction + 6;
    return name.c_str();
}

const char* ireduceFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "ireduce_float";
        case c32: return "ireduce_cfloat";
        case s32: return "ireduce_int";
        case u32: return "ireduce_uint";
        case s64: return "ireduce_long";
        case u64: return "ireduce_ulong";
        case s8: return "ireduce_char";
        case u8:
        case b8: return "ireduce_uchar";
        case s16: return "ireduce_short";
        case u16: return "ireduce_ushort";
        case f16: return "ireduce_half";
        default: return nullptr;
    }
}

const char* ireduceAllFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "ireduce_all_float";
        case c32: return "ireduce_all_cfloat";
        case s32: return "ireduce_all_int";
        case u32: return "ireduce_all_uint";
        case s64: return "ireduce_all_long";
        case u64: return "ireduce_all_ulong";
        case s8: return "ireduce_all_char";
        case u8:
        case b8: return "ireduce_all_uchar";
        case s16: return "ireduce_all_short";
        case u16: return "ireduce_all_ushort";
        case f16: return "ireduce_all_half";
        default: return nullptr;
    }
}

const char* rreduceFunctionName(const af_dtype type) {
    const char* name = ireduceFunctionName(type);
    if (!name) return nullptr;
    static thread_local std::string rname;
    rname = "rreduce_";
    rname += name + 8;
    return rname.c_str();
}

const char* fastFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "fast_locate_float";
        case f16: return "fast_locate_half";
        case s32: return "fast_locate_int";
        case u32: return "fast_locate_uint";
        case s8: return "fast_locate_char";
        case u8:
        case b8: return "fast_locate_uchar";
        case s16: return "fast_locate_short";
        case u16: return "fast_locate_ushort";
        default: return nullptr;
    }
}

const char* exampleFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "example_function_float";
        case c32: return "example_function_cfloat";
        case s32: return "example_function_int";
        case u32: return "example_function_uint";
        case s8:
        case b8: return "example_function_char";
        case u8: return "example_function_uchar";
        default: return nullptr;
    }
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

const char* susanResponseFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "susan_response_float";
        case s32: return "susan_response_int";
        case u32: return "susan_response_uint";
        case s8: return "susan_response_char";
        case u8:
        case b8: return "susan_response_uchar";
        case s16: return "susan_response_short";
        case u16: return "susan_response_ushort";
        default: return nullptr;
    }
}

const char* susanNonMaxFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "susan_nonmax_float";
        case s32: return "susan_nonmax_int";
        case u32: return "susan_nonmax_uint";
        case s8: return "susan_nonmax_char";
        case u8:
        case b8: return "susan_nonmax_uchar";
        case s16: return "susan_nonmax_short";
        case u16: return "susan_nonmax_ushort";
        default: return nullptr;
    }
}

const char* svdInitFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "svd_init_float";
        case c32: return "svd_init_cfloat";
        default: return nullptr;
    }
}

const char* svdStageFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "svd_stage_float";
        case c32: return "svd_stage_cfloat";
        default: return nullptr;
    }
}

const char* svdSortFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "svd_sort_float";
        case c32: return "svd_sort_cfloat";
        default: return nullptr;
    }
}

const char* svdFinalizeFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "svd_finalize_float";
        case c32: return "svd_finalize_cfloat";
        default: return nullptr;
    }
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
