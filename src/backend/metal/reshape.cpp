/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <copy.hpp>

#include <arith.hpp>
#include <assign.hpp>
#include <common/cast.hpp>
#include <common/half.hpp>
#include <index.hpp>
#include <err_metal.hpp>

#include <type_traits>
#include <vector>

namespace arrayfire {
namespace metal {
namespace {

template<typename T>
constexpr bool supportsMetalReshapeType =
    !std::is_same<T, double>::value && !std::is_same<T, cdouble>::value;

std::vector<af_index_t> reshapeIndices(const dim4 &dims) {
    std::vector<af_index_t> indices(4);
    for (int i = 0; i < 4; ++i) {
        indices[i].idx.seq =
            af_seq{0.0, static_cast<double>(dims[i]) - 1.0, 1.0};
        indices[i].isSeq   = true;
        indices[i].isBatch = false;
    }
    return indices;
}

}  // namespace

template<typename T>
void multiply_inplace(Array<T> &in, double val) {
    if constexpr (supportsMetalReshapeType<T>) {
        auto factor = createValueArray<T>(in.dims(), scalar<T>(val));
        in = arithOp<T, af_mul_t>(in, factor, in.dims());
    } else {
        AF_ERROR("Metal reshape scaling does not support double precision",
                 AF_ERR_NOT_SUPPORTED);
    }
}

template<typename inType, typename outType>
Array<outType> reshape(const Array<inType> &in, const dim4 &outDims,
                       outType defaultValue, double scale) {
    if constexpr (supportsMetalReshapeType<inType> &&
                  supportsMetalReshapeType<outType>) {
        Array<outType> transformed = common::cast<outType>(in);
        if (scale != 1.0) {
            auto factor = createValueArray<outType>(
                transformed.dims(), scalar<outType>(scale));
            transformed = arithOp<outType, af_mul_t>(
                transformed, factor, transformed.dims());
        }
        if (transformed.dims() == outDims) { return transformed; }

        Array<outType> out = createValueArray(outDims, defaultValue);
        dim4 copyDims;
        for (int i = 0; i < 4; ++i)
            copyDims[i] = std::min(in.dims()[i], outDims[i]);
        if (copyDims.elements() == 0) { return out; }

        auto indices = reshapeIndices(copyDims);
        auto source  = index<outType>(transformed, indices.data());
        assign<outType>(out, indices.data(), source);
        return out;
    } else {
        AF_ERROR("Metal reshape does not support double precision",
                 AF_ERR_NOT_SUPPORTED);
    }
}

#define INSTANTIATE(T) \
    template void multiply_inplace<T>(Array<T> & in, double norm);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)

#define INSTANTIATE_PAD_ARRAY(SRC_T)                                          \
    template Array<float> reshape<SRC_T, float>(const Array<SRC_T> &,         \
                                                const dim4 &, float, double); \
    template Array<double> reshape<SRC_T, double>(                            \
        const Array<SRC_T> &, const dim4 &, double, double);                  \
    template Array<cfloat> reshape<SRC_T, cfloat>(                            \
        const Array<SRC_T> &, const dim4 &, cfloat, double);                  \
    template Array<cdouble> reshape<SRC_T, cdouble>(                          \
        const Array<SRC_T> &, const dim4 &, cdouble, double);                 \
    template Array<int> reshape<SRC_T, int>(const Array<SRC_T> &,             \
                                            const dim4 &, int, double);       \
    template Array<uint> reshape<SRC_T, uint>(const Array<SRC_T> &,           \
                                              const dim4 &, uint, double);    \
    template Array<intl> reshape<SRC_T, intl>(const Array<SRC_T> &,           \
                                              const dim4 &, intl, double);    \
    template Array<uintl> reshape<SRC_T, uintl>(const Array<SRC_T> &,         \
                                                const dim4 &, uintl, double); \
    template Array<short> reshape<SRC_T, short>(const Array<SRC_T> &,         \
                                                const dim4 &, short, double); \
    template Array<ushort> reshape<SRC_T, ushort>(                            \
        const Array<SRC_T> &, const dim4 &, ushort, double);                  \
    template Array<schar> reshape<SRC_T, schar>(const Array<SRC_T> &,         \
                                                const dim4 &, schar, double); \
    template Array<uchar> reshape<SRC_T, uchar>(const Array<SRC_T> &,         \
                                                const dim4 &, uchar, double); \
    template Array<char> reshape<SRC_T, char>(const Array<SRC_T> &,           \
                                              const dim4 &, char, double);

INSTANTIATE_PAD_ARRAY(float)
INSTANTIATE_PAD_ARRAY(double)
INSTANTIATE_PAD_ARRAY(int)
INSTANTIATE_PAD_ARRAY(uint)
INSTANTIATE_PAD_ARRAY(intl)
INSTANTIATE_PAD_ARRAY(uintl)
INSTANTIATE_PAD_ARRAY(schar)
INSTANTIATE_PAD_ARRAY(uchar)
INSTANTIATE_PAD_ARRAY(char)
INSTANTIATE_PAD_ARRAY(ushort)
INSTANTIATE_PAD_ARRAY(short)
INSTANTIATE_PAD_ARRAY(arrayfire::common::half)

#define INSTANTIATE_PAD_ARRAY_COMPLEX(SRC_T)                 \
    template Array<cfloat> reshape<SRC_T, cfloat>(           \
        const Array<SRC_T> &, const dim4 &, cfloat, double); \
    template Array<cdouble> reshape<SRC_T, cdouble>(         \
        const Array<SRC_T> &, const dim4 &, cdouble, double);

INSTANTIATE_PAD_ARRAY_COMPLEX(cfloat)
INSTANTIATE_PAD_ARRAY_COMPLEX(cdouble)
}  // namespace metal
}  // namespace arrayfire
