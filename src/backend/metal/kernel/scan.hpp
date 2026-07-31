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
#include <optypes.hpp>

#include <cstddef>
#include <cstdint>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalScan(af_dtype inputType, af_dtype outputType) noexcept;
void launchMetalScan(BufferParam output, size_t outputBytes,
                     const af::dim4& dims,
                     const af::dim4& outputStrides, BufferParam input,
                     size_t inputBytes, const af::dim4& inputStrides,
                     int dimension, af_dtype inputType, af_dtype outputType,
                     uint32_t operation, bool inclusive);

template<af_op_t op>
constexpr uint32_t scanOperation() noexcept {
    if constexpr (op == af_add_t) {
        return 0;
    } else if constexpr (op == af_mul_t) {
        return 1;
    } else if constexpr (op == af_min_t) {
        return 2;
    } else if constexpr (op == af_max_t) {
        return 3;
    } else {
        static_assert(op == af_notzero_t, "Unsupported Metal scan operation");
        return 4;
    }
}

template<af_op_t op, typename Ti, typename To>
void scanMetal(Param<To> output, CParam<Ti> input, const int dimension,
               const bool inclusive) {
    const af_dtype inputType =
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type);
    const af_dtype outputType =
        static_cast<af_dtype>(af::dtype_traits<To>::af_type);
    launchMetalScan(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(To),
        output.dims(), output.strides(), input.bufferParam(), sizeof(Ti),
        input.strides(), dimension, inputType, outputType, scanOperation<op>(),
        inclusive);
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
