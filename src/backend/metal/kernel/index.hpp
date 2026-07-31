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

#include <af/seq.h>
#include <af/traits.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalIndex(af_dtype type) noexcept;

void launchMetalIndex(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, dim_t outputOffset, BufferParam input,
    size_t inputBytes, const af::dim4& inputDims,
    const af::dim4& inputStrides, dim_t inputOffset,
    const af::dim4& offsets, const std::vector<af_seq>& sequences,
    const std::array<bool, 4>& isSequence,
    const std::array<BufferParam, 4>& indexBuffers,
    const std::array<dim_t, 4>& indexOffsets,
    const std::array<dim_t, 4>& indexStrides, af_dtype type);

template<typename T>
void indexMetal(Param<T> output, CParam<T> input, const af::dim4 dataDims,
                const std::vector<bool> isSequence,
                const std::vector<af_seq> sequences,
                const std::vector<CParam<uint>> indexArrays) {
    af::dim4 offsets(0);
    std::array<bool, 4> sequenceFlags{};
    std::array<BufferParam, 4> indexBuffers{};
    std::array<dim_t, 4> indexOffsets{};
    std::array<dim_t, 4> indexStrides{};
    for (int i = 0; i < 4; ++i) {
        if (sequences[i].step != 0 && sequences[i].begin >= 0)
            offsets[i] = static_cast<dim_t>(sequences[i].begin);
        else if (sequences[i].begin <= -1)
            offsets[i] = dataDims[i] + static_cast<dim_t>(sequences[i].begin);
        sequenceFlags[i] = isSequence[i];
        if (isSequence[i]) {
            indexBuffers[i] = {input.getBuffer(), 0};
        } else {
            indexBuffers[i] = {indexArrays[i].getBuffer(), 0};
            indexOffsets[i] = indexArrays[i].getOffset();
            indexStrides[i] = indexArrays[i].strides(0);
        }
    }
    launchMetalIndex(
        {output.getBuffer(), 0},
        static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), output.getOffset(),
        {input.getBuffer(), 0},
        static_cast<size_t>(input.dims().elements()) * sizeof(T), input.dims(),
        input.strides(), input.getOffset(), offsets, sequences, sequenceFlags,
        indexBuffers, indexOffsets, indexStrides,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
