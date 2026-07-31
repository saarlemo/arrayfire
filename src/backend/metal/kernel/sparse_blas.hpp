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
#include <af/defines.h>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalSparseMatmul(af_dtype type) noexcept;

void launchMetalSparseMatmul(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam values, BufferParam rowIndex,
    BufferParam columnIndex, BufferParam rhs, const af::dim4& rhsStrides,
    const af::dim4& sparseDims, af_mat_prop operation, af_dtype type);

template<typename T>
void sparseMatmulMetal(Param<T> output, CParam<T> values,
                       CParam<int> rowIndex, CParam<int> columnIndex,
                       CParam<T> rhs, const af::dim4 sparseDims,
                       const af_mat_prop operation) {
    launchMetalSparseMatmul(
        output.bufferParam(), output.dims(), output.strides(),
        values.bufferParam(), rowIndex.bufferParam(), columnIndex.bufferParam(),
        rhs.bufferParam(), rhs.strides(), sparseDims, operation,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
