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
#include <common/ArrayInfo.hpp>

#include <af/seq.h>
#include <af/traits.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalAssign(af_dtype type) noexcept;

void launchMetalAssign(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& destinationStrides, dim_t outputOffset, BufferParam rhs,
    size_t rhsBytes, const af::dim4& rhsDims,
    const af::dim4& rhsStrides, dim_t rhsOffset, const af::dim4& offsets,
    const std::array<bool, 4>& isSequence,
    const std::array<BufferParam, 4>& indexBuffers,
    const std::array<dim_t, 4>& indexOffsets,
    const std::array<dim_t, 4>& indexStrides, af_dtype type);

template<typename T>
void assignMetal(Param<T> output, const af::dim4 dataDims, CParam<T> rhs,
                 const std::vector<bool> isSequence,
                 const std::vector<af_seq> sequences,
                 const std::vector<CParam<uint>> indexArrays) {
    const af::dim4 offsets = toOffset(sequences, dataDims);
    const af::dim4 destinationStrides = toStride(sequences, dataDims);
    std::array<bool, 4> sequenceFlags{};
    std::array<BufferParam, 4> indexBuffers{};
    std::array<dim_t, 4> indexOffsets{};
    std::array<dim_t, 4> indexStrides{};
    for (int i = 0; i < 4; ++i) {
        sequenceFlags[i] = isSequence[i];
        if (isSequence[i]) {
            indexBuffers[i] = {rhs.getBuffer(), 0};
        } else {
            indexBuffers[i] = {indexArrays[i].getBuffer(), 0};
            indexOffsets[i] = indexArrays[i].getOffset();
            indexStrides[i] = indexArrays[i].strides(0);
        }
    }
    launchMetalAssign(
        {output.getBuffer(), 0},
        static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), destinationStrides, output.getOffset(),
        {rhs.getBuffer(), 0},
        static_cast<size_t>(rhs.dims().elements()) * sizeof(T), rhs.dims(),
        rhs.strides(), rhs.getOffset(), offsets, sequenceFlags, indexBuffers,
        indexOffsets, indexStrides,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
