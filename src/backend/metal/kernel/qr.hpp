/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <af/dim4.hpp>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalQR(af_dtype type) noexcept;
void launchMetalQRFactor(BufferParam input, size_t inputBytes,
                         const af::dim4& dims, const af::dim4& strides,
                         BufferParam tau, size_t tauBytes, af_dtype type);
void launchMetalQRGenerate(BufferParam output, size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, BufferParam packed,
                           size_t packedBytes, const af::dim4& packedDims,
                           const af::dim4& packedStrides, BufferParam tau,
                           size_t tauBytes, af_dtype type);

template<typename T>
void qrFactorMetal(Param<T> input, Param<T> tau) {
    launchMetalQRFactor(
        input.bufferParam(), size_t(input.dims().elements()) * sizeof(T),
        input.dims(), input.strides(), tau.bufferParam(),
        size_t(tau.dims().elements()) * sizeof(T),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void qrGenerateMetal(Param<T> output, CParam<T> packed, CParam<T> tau) {
    launchMetalQRGenerate(
        output.bufferParam(), size_t(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), packed.bufferParam(),
        size_t(packed.dims().elements()) * sizeof(T), packed.dims(),
        packed.strides(), tau.bufferParam(),
        size_t(tau.dims().elements()) * sizeof(T),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
