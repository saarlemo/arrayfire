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

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalSobel(af_dtype inputType) noexcept;

void launchMetalSobel(BufferParam derivative0,
                      const af::dim4& derivative0Strides,
                      BufferParam derivative1,
                      const af::dim4& derivative1Strides,
                      size_t outputBytes, BufferParam input, size_t inputBytes,
                      const af::dim4& inputDims, const af::dim4& inputStrides,
                      af_dtype inputType);

template<typename Ti, typename To>
void sobelMetal(Param<To> derivative0, Param<To> derivative1,
                CParam<Ti> input) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalSobel(
        derivative0.bufferParam(), derivative0.strides(),
        derivative1.bufferParam(), derivative1.strides(),
        static_cast<size_t>(derivative0.dims().elements()) * sizeof(To),
        input.bufferParam(), inputElements * sizeof(Ti), input.dims(),
        input.strides(),
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
