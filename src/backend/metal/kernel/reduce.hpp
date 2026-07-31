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
#include <cstdint>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalReduce(af_dtype inputType, af_dtype outputType) noexcept;
bool supportsMetalReduceByKey(af_dtype keyType, af_dtype inputType,
                              af_dtype outputType) noexcept;
void launchMetalReduce(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam input, size_t inputBytes,
    const af::dim4& inputDims, const af::dim4& inputStrides, int dimension,
    uint32_t operation, bool reduceAll, bool changeNan, double nanValue,
    af_dtype inputType, af_dtype outputType);
void launchMetalReduceByKeyCompact(
    BufferParam outputKeys, size_t outputBytes, BufferParam count,
    BufferParam inputKeys, size_t inputBytes, const af::dim4& keyDims,
    const af::dim4& keyStrides, af_dtype keyType);
void launchMetalReduceByKey(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam keys, size_t keyBytes,
    const af::dim4& keyStrides, BufferParam input, size_t inputBytes,
    const af::dim4& inputDims, const af::dim4& inputStrides, int dimension,
    uint32_t operation, int nReduced, bool changeNan, double nanValue,
    af_dtype keyType, af_dtype inputType, af_dtype outputType);

template<af_op_t op>
constexpr uint32_t reduceOperation() noexcept {
    if constexpr (op == af_add_t) {
        return 0;
    } else if constexpr (op == af_mul_t) {
        return 1;
    } else if constexpr (op == af_min_t) {
        return 2;
    } else if constexpr (op == af_max_t) {
        return 3;
    } else if constexpr (op == af_notzero_t) {
        return 4;
    } else if constexpr (op == af_or_t) {
        return 5;
    } else {
        static_assert(op == af_and_t, "Unsupported Metal reduction operation");
        return 6;
    }
}

template<af_op_t op, typename Ti, typename To>
void reduceMetal(Param<To> output, CParam<Ti> input, const int dimension,
                 const bool reduceAll, const bool changeNan,
                 const double nanValue) {
    launchMetalReduce(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(To),
        output.dims(), output.strides(), input.bufferParam(), sizeof(Ti),
        input.dims(), input.strides(), dimension, reduceOperation<op>(),
        reduceAll, changeNan, nanValue,
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type),
        static_cast<af_dtype>(af::dtype_traits<To>::af_type));
}

template<typename Tk>
void reduceByKeyCompactMetal(Param<Tk> outputKeys, Param<int> count,
                             CParam<Tk> inputKeys) {
    launchMetalReduceByKeyCompact(
        outputKeys.bufferParam(),
        static_cast<size_t>(outputKeys.dims().elements()) * sizeof(Tk),
        count.bufferParam(), inputKeys.bufferParam(), sizeof(Tk),
        inputKeys.dims(), inputKeys.strides(),
        static_cast<af_dtype>(af::dtype_traits<Tk>::af_type));
}

template<af_op_t op, typename Ti, typename Tk, typename To>
void reduceByKeyMetal(Param<To> output, CParam<Tk> keys, CParam<Ti> input,
                      const int dimension, const int nReduced,
                      const bool changeNan, const double nanValue) {
    launchMetalReduceByKey(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(To),
        output.dims(), output.strides(), keys.bufferParam(), sizeof(Tk),
        keys.strides(), input.bufferParam(), sizeof(Ti), input.dims(),
        input.strides(), dimension, reduceOperation<op>(), nReduced, changeNan,
        nanValue, static_cast<af_dtype>(af::dtype_traits<Tk>::af_type),
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type),
        static_cast<af_dtype>(af::dtype_traits<To>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
