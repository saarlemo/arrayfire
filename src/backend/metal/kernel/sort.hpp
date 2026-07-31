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

bool supportsMetalSort(af_dtype type) noexcept;
void launchMetalSort(BufferParam inout, size_t bytes, const af::dim4& dims,
                     const af::dim4& strides, int dimension, bool ascending,
                     af_dtype type);

template<typename T>
void sortMetal(Param<T> inout, const int dimension, const bool ascending) {
    launchMetalSort(inout.bufferParam(),
                    static_cast<size_t>(inout.dims().elements()) * sizeof(T),
                    inout.dims(), inout.strides(), dimension, ascending,
                    static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
