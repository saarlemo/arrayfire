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

bool supportsMetalAssign(af_dtype type) noexcept;

void launchMetalAssign(void* output, size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, const void* rhs,
                       size_t rhsBytes, const af::dim4& rhsDims,
                       const af::dim4& rhsStrides, const af::dim4& offsets,
                       af_dtype type);

template<typename T>
void assignMetal(Param<T> output, const af::dim4 dataDims, CParam<T> rhs,
                 const std::vector<af_seq> sequences) {
    af::dim4 offsets(0);
    for (int i = 0; i < 4; ++i) {
        if (sequences[i].step != 0 && sequences[i].begin >= 0)
            offsets[i] = static_cast<dim_t>(sequences[i].begin);
        else if (sequences[i].begin <= -1)
            offsets[i] = dataDims[i] + static_cast<dim_t>(sequences[i].begin);
    }
    size_t rhsElements = 1;
    for (int i = 0; i < 4; ++i)
        rhsElements += static_cast<size_t>(rhs.dims(i) - 1) *
                       static_cast<size_t>(rhs.strides(i));
    launchMetalAssign(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), rhs.get(), rhsElements * sizeof(T),
        rhs.dims(), rhs.strides(), offsets,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
