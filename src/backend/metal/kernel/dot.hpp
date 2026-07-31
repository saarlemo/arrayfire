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
#include <af/dim4.hpp>
#include <af/traits.hpp>
#include <cstddef>
#include <type_traits>
#include <types.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalDot(af_dtype type) noexcept;
const char* gemmFunctionName(af_dtype type);
void launchMetalDot(BufferParam output, size_t outputBytes, BufferParam lhs,
                    size_t lhsBytes, const af::dim4& lhsDims,
                    const af::dim4& lhsStrides, BufferParam rhs,
                    size_t rhsBytes, const af::dim4& rhsStrides,
                    af_mat_prop lhsOption, af_mat_prop rhsOption,
                    af_dtype type);

void launchMetalGemm(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam lhs, size_t lhsBytes,
    const af::dim4& lhsDims, const af::dim4& lhsStrides, BufferParam rhs,
    size_t rhsBytes, const af::dim4& rhsDims, const af::dim4& rhsStrides,
    af_mat_prop lhsOption, af_mat_prop rhsOption, float alphaReal,
    float alphaImag, float betaReal, float betaImag, af_dtype type);

template<typename T>
void dotMetal(Param<T> output, CParam<T> lhs, CParam<T> rhs,
              const af_mat_prop lhsOption, const af_mat_prop rhsOption) {
    size_t lhsElements = 1, rhsElements = 1;
    for (int i = 0; i < 4; ++i) {
        lhsElements += static_cast<size_t>(lhs.dims(i) - 1) *
                       static_cast<size_t>(lhs.strides(i));
        rhsElements += static_cast<size_t>(rhs.dims(i) - 1) *
                       static_cast<size_t>(rhs.strides(i));
    }
    launchMetalDot(output.bufferParam(), sizeof(T), lhs.bufferParam(),
                   lhsElements * sizeof(T), lhs.dims(), lhs.strides(),
                   rhs.bufferParam(),
                   rhsElements * sizeof(T), rhs.strides(), lhsOption, rhsOption,
                   static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void gemmMetal(Param<T> output, const af_mat_prop lhsOption,
               const af_mat_prop rhsOption, const T* alpha, CParam<T> lhs,
               CParam<T> rhs, const T* beta) {
    float alphaReal = 0.0f, alphaImag = 0.0f;
    float betaReal  = 0.0f, betaImag  = 0.0f;
    if constexpr (std::is_same_v<T, cfloat>) {
        alphaReal = alpha->real();
        alphaImag = alpha->imag();
        betaReal  = beta->real();
        betaImag  = beta->imag();
    } else {
        alphaReal = static_cast<float>(*alpha);
        betaReal  = static_cast<float>(*beta);
    }
    launchMetalGemm(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), lhs.bufferParam(),
        static_cast<size_t>(lhs.dims().elements()) * sizeof(T), lhs.dims(),
        lhs.strides(), rhs.bufferParam(),
        static_cast<size_t>(rhs.dims().elements()) * sizeof(T), rhs.dims(),
        rhs.strides(), lhsOption, rhsOption, alphaReal, alphaImag, betaReal,
        betaImag, static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}
}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
