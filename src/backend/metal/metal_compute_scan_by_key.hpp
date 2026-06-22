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

bool supportsMetalScanByKey(af_dtype keyType, af_dtype valueType) noexcept;
void launchMetalScanByKey(void* output, size_t outputBytes,
                          const af::dim4& dims, const af::dim4& outputStrides,
                          const void* keys, size_t keyBytes,
                          const af::dim4& keyStrides, const void* input,
                          size_t inputBytes, const af::dim4& inputStrides,
                          int dimension, af_dtype keyType, af_dtype valueType);

template<typename Ti, typename Tk, typename To>
void scanByKeyMetal(Param<To> output, CParam<Tk> keys, CParam<Ti> input,
                    const int dimension) {
    size_t keyElements   = 1;
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        keyElements += static_cast<size_t>(keys.dims(i) - 1) *
                       static_cast<size_t>(keys.strides(i));
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalScanByKey(
        output.get(),
        static_cast<size_t>(output.dims().elements()) * sizeof(To),
        output.dims(), output.strides(), keys.get(), keyElements * sizeof(Tk),
        keys.strides(), input.get(), inputElements * sizeof(Ti),
        input.strides(), dimension,
        static_cast<af_dtype>(af::dtype_traits<Tk>::af_type),
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
