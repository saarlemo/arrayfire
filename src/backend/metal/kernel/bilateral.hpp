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

bool supportsMetalBilateral(af_dtype inputType, af_dtype outputType) noexcept;

void launchMetalBilateral(BufferParam output, size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, BufferParam input,
                          size_t inputBytes, const af::dim4& inputStrides,
                          float spatialSigma, float chromaticSigma,
                          af_dtype inputType, af_dtype outputType);

template<typename InT, typename OutT>
void bilateralMetal(Param<OutT> output, CParam<InT> input,
                    const float spatialSigma, const float chromaticSigma) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalBilateral(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(OutT),
        output.dims(), output.strides(), input.bufferParam(),
        inputElements * sizeof(InT), input.strides(), spatialSigma,
        chromaticSigma, static_cast<af_dtype>(af::dtype_traits<InT>::af_type),
        static_cast<af_dtype>(af::dtype_traits<OutT>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
