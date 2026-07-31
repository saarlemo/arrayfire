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

bool supportsMetalSelect(af_dtype type) noexcept;

void launchMetalSelect(BufferParam output, size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, BufferParam condition,
                       size_t conditionBytes, const af::dim4& conditionDims,
                       const af::dim4& conditionStrides, BufferParam a,
                       size_t aBytes, const af::dim4& aDims,
                       const af::dim4& aStrides, BufferParam b, size_t bBytes,
                       const af::dim4& bDims, const af::dim4& bStrides,
                       const void* scalar, bool flip, af_dtype type);

template<typename T>
void selectMetal(Param<T> output, CParam<char> condition, CParam<T> a,
                 CParam<T> b) {
    const auto span = [](const auto& param) {
        size_t elements = 1;
        for (int i = 0; i < 4; ++i) {
            elements += static_cast<size_t>(param.dims(i) - 1) *
                        static_cast<size_t>(param.strides(i));
        }
        return elements;
    };
    launchMetalSelect(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), condition.bufferParam(),
        span(condition) * sizeof(char), condition.dims(), condition.strides(),
        a.bufferParam(), span(a) * sizeof(T), a.dims(), a.strides(),
        b.bufferParam(),
        span(b) * sizeof(T), b.dims(), b.strides(), nullptr, false,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T, bool flip>
void selectScalarMetal(Param<T> output, CParam<char> condition, CParam<T> a,
                       const T scalar) {
    const auto span = [](const auto& param) {
        size_t elements = 1;
        for (int i = 0; i < 4; ++i) {
            elements += static_cast<size_t>(param.dims(i) - 1) *
                        static_cast<size_t>(param.strides(i));
        }
        return elements;
    };
    launchMetalSelect(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T), output.dims(),
        output.strides(), condition.bufferParam(),
        span(condition) * sizeof(char), condition.dims(), condition.strides(),
        a.bufferParam(), span(a) * sizeof(T), a.dims(), a.strides(), {nullptr, 0},
        0,
        af::dim4(1), af::dim4(1), &scalar, flip,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
