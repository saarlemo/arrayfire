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
#include <common/internal_enums.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

void fftConvolvePackMetal(Param<float> output, const af::dim4& outputDims,
                          const af::dim4& outputStrides, BufferParam input,
                          const af::dim4& inputDims,
                          const af::dim4& inputStrides, af_dtype type);

void fftConvolvePadMetal(Param<float> output, const af::dim4& outputDims,
                         const af::dim4& outputStrides, BufferParam input,
                         const af::dim4& inputDims,
                         const af::dim4& inputStrides, dim_t offset,
                         af_dtype type);

void fftConvolveReorderMetal(BufferParam output, BufferParam packed,
                             const af::dim4& outputDims,
                             const af::dim4& outputStrides,
                             const af::dim4& inputDims,
                             const af::dim4& inputStrides,
                             const af::dim4& filterDims, dim_t filterOffset,
                             dim_t signalHalfDim0, dim_t fftScale,
                             AF_BATCH_KIND kind, af_dtype type, int rank,
                             bool expand);

template<typename T>
void packData(Param<float> output, const af::dim4& outputDims,
              const af::dim4& outputStrides, CParam<T> input) {
    fftConvolvePackMetal(output, outputDims, outputStrides,
                         input.bufferParam(), input.dims(), input.strides(),
                         static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void padArray(Param<float> output, const af::dim4& outputDims,
              const af::dim4& outputStrides, CParam<T> input, dim_t offset) {
    fftConvolvePadMetal(output, outputDims, outputStrides,
                        input.bufferParam(), input.dims(), input.strides(),
                        offset,
                        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void reorder(Param<T> output, Param<float> packed, CParam<T> filter,
             dim_t signalHalfDim0, dim_t fftScale,
             const af::dim4& signalDims, const af::dim4& signalStrides,
             const af::dim4& filterTmpDims,
             const af::dim4& filterTmpStrides, AF_BATCH_KIND kind, int rank,
             bool expand) {
    fftConvolveReorderMetal(
        output.bufferParam(), packed.bufferParam(), output.dims(), output.strides(),
        kind == AF_BATCH_RHS ? filterTmpDims : signalDims,
        kind == AF_BATCH_RHS ? filterTmpStrides : signalStrides, filter.dims(),
        signalStrides[3] * signalDims[3], signalHalfDim0, fftScale, kind,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type), rank, expand);
}

void fftConvolveMultiplyMetal(Param<float> packed,
                              const af::dim4& signalDims,
                              const af::dim4& signalStrides,
                              const af::dim4& filterDims,
                              const af::dim4& filterStrides, AF_BATCH_KIND kind,
                              dim_t offset);

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
