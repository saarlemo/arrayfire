/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <af/traits.hpp>
#include <cstddef>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalSortByKey(af_dtype keyType, af_dtype valueType) noexcept;
bool supportsMetalSortIndex(af_dtype type) noexcept;
void launchMetalSortByKey(BufferParam keys, size_t keyBytes,
                          const af::dim4& keyDims,
                          const af::dim4& keyStrides, BufferParam values,
                          size_t valueBytes, const af::dim4& valueStrides,
                          int dimension, bool ascending, af_dtype keyType,
                          af_dtype valueType);
void launchMetalSortIndex(BufferParam keys, size_t keyBytes,
                          const af::dim4& keyDims,
                          const af::dim4& keyStrides, BufferParam values,
                          const af::dim4& valueStrides, int dimension,
                          bool ascending, af_dtype keyType);

template<typename Tk, typename Tv>
void sortByKeyMetal(Param<Tk> keys, Param<Tv> values, const int dimension,
                    const bool ascending) {
    launchMetalSortByKey(
        keys.bufferParam(),
        static_cast<size_t>(keys.dims().elements()) * sizeof(Tk), keys.dims(),
        keys.strides(), values.bufferParam(),
        static_cast<size_t>(values.dims().elements()) * sizeof(Tv),
        values.strides(), dimension, ascending,
        static_cast<af_dtype>(af::dtype_traits<Tk>::af_type),
        static_cast<af_dtype>(af::dtype_traits<Tv>::af_type));
}

template<typename T>
void sortIndexMetal(Param<T> keys, Param<uint> values,
                    const int dimension, const bool ascending) {
    launchMetalSortIndex(
        keys.bufferParam(),
        static_cast<size_t>(keys.dims().elements()) * sizeof(T), keys.dims(),
        keys.strides(), values.bufferParam(), values.strides(), dimension,
        ascending,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
