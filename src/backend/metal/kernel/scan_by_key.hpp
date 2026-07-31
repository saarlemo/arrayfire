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
#include <kernel/scan.hpp>

#include <cstddef>
#include <cstdint>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalScanByKey(af_dtype keyType, af_dtype valueType) noexcept;
void launchMetalScanByKey(BufferParam output, size_t outputBytes,
                          const af::dim4& dims,
                          const af::dim4& outputStrides, BufferParam keys,
                          size_t keyBytes, const af::dim4& keyStrides,
                          BufferParam input, size_t inputBytes,
                          const af::dim4& inputStrides, int dimension,
                          af_dtype keyType, af_dtype valueType,
                          uint32_t operation, bool inclusive);

template<af_op_t op, typename Ti, typename Tk, typename To>
void scanByKeyMetal(Param<To> output, CParam<Tk> keys, CParam<Ti> input,
                    const int dimension, const bool inclusive) {
    const af_dtype keyType =
        static_cast<af_dtype>(af::dtype_traits<Tk>::af_type);
    const af_dtype valueType =
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type);
    launchMetalScanByKey(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(To),
        output.dims(), output.strides(), keys.bufferParam(), sizeof(Tk),
        keys.strides(), input.bufferParam(), sizeof(Ti), input.strides(),
        dimension, keyType, valueType, scanOperation<op>(), inclusive);
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
