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
#include <math.hpp>
#include <af/dim4.hpp>

#include <cstring>

namespace arrayfire {
namespace metal {
namespace kernel {

template<typename T>
void stridedCopy(T* dst, af::dim4 const& ostrides, T const* src,
                 af::dim4 const& dims, af::dim4 const& strides, unsigned dim) {
    if (dim == 0) {
        if (strides[dim] == 1) {
            // FIXME: Check for errors / exceptions
            std::memcpy(dst, src, dims[dim] * sizeof(T));
        } else {
            for (dim_t i = 0; i < dims[dim]; i++) {
                dst[i] = src[strides[dim] * i];
            }
        }
    } else {
        for (dim_t i = dims[dim]; i > 0; i--) {
            stridedCopy<T>(dst, ostrides, src, dims, strides, dim - 1);
            src += strides[dim];
            dst += ostrides[dim];
        }
    }
}

bool supportsMetalCopy(af_dtype type) noexcept;

void launchMetalCopy(BufferParam output, size_t outputBytes,
                     const af::dim4& dims,
                     const af::dim4& outputStrides, dim_t outputOffset,
                     BufferParam input, size_t inputBytes,
                     const af::dim4& inputStrides, dim_t inputOffset,
                     af_dtype type);

template<typename T>
void copyMetal(Param<T> output, CParam<T> input) {
    launchMetalCopy(
        {output.getBuffer(), 0},
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), output.getOffset(), {input.getBuffer(), 0},
        static_cast<size_t>(input.dims().elements()) * sizeof(T),
        input.strides(), input.getOffset(),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
