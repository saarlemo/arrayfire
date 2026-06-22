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
#include <af/traits.hpp>

#include <cstddef>
#include <vector>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalIndex(af_dtype type) noexcept;

void launchMetalIndex(void* output, size_t outputBytes,
                      const af::dim4& outputDims, const af::dim4& outputStrides,
                      const void* input, size_t inputBytes,
                      const af::dim4& inputDims, const af::dim4& inputStrides,
                      const af::dim4& offsets,
                      const std::vector<af_seq>& sequences, af_dtype type);

template<typename T>
void indexMetal(Param<T> output, CParam<T> input, const af::dim4 dataDims,
                const std::vector<af_seq> sequences) {
    af::dim4 offsets(0);
    for (int i = 0; i < 4; ++i) {
        if (sequences[i].step != 0 && sequences[i].begin >= 0)
            offsets[i] = static_cast<dim_t>(sequences[i].begin);
        else if (sequences[i].begin <= -1)
            offsets[i] = dataDims[i] + static_cast<dim_t>(sequences[i].begin);
    }
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i)
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    launchMetalIndex(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.dims(), input.strides(), offsets, sequences,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
