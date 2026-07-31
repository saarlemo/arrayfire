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

bool supportsMetalTransform(af_dtype type, af_interp_type method) noexcept;

void launchMetalTransform(BufferParam output, size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, dim_t outputOffset,
                          BufferParam input, size_t inputBytes,
                          const af::dim4& inputDims,
                          const af::dim4& inputStrides, dim_t inputOffset,
                          BufferParam transform, size_t transformBytes,
                          const af::dim4& transformDims,
                          const af::dim4& transformStrides,
                          dim_t transformOffset, af_interp_type method,
                          bool inverse, bool perspective, af_dtype type);

template<typename T>
void transformMetal(Param<T> output, CParam<T> input, CParam<float> transform,
                    const af_interp_type method, const bool inverse,
                    const bool perspective) {
    launchMetalTransform(
        {output.getBuffer(), 0},
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), output.getOffset(), {input.getBuffer(), 0},
        static_cast<size_t>(input.dims().elements()) * sizeof(T), input.dims(),
        input.strides(), input.getOffset(), {transform.getBuffer(), 0},
        static_cast<size_t>(transform.dims().elements()) * sizeof(float),
        transform.dims(), transform.strides(), transform.getOffset(), method,
        inverse, perspective,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
