/*******************************************************
 * Copyright (c) 2026, ArrayFire
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

bool supportsMetalSortByKey(af_dtype type) noexcept;
void launchMetalSortByKey(void* keys, size_t keyBytes, const af::dim4& keyDims,
                          const af::dim4& keyStrides, void* values,
                          size_t valueBytes, const af::dim4& valueStrides,
                          bool ascending, af_dtype type);

template<typename T>
void sort0ByKeyMetal(Param<T> keys, Param<T> values, const bool ascending) {
    launchMetalSortByKey(
        keys.get(), static_cast<size_t>(keys.dims().elements()) * sizeof(T),
        keys.dims(), keys.strides(), values.get(),
        static_cast<size_t>(values.dims().elements()) * sizeof(T),
        values.strides(), ascending,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
