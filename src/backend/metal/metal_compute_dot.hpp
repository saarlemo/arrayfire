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

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalDot(af_dtype type) noexcept;
void launchMetalDot(void* output, size_t outputBytes, const void* lhs,
                    size_t lhsBytes, const af::dim4& lhsDims,
                    const af::dim4& lhsStrides, const void* rhs,
                    size_t rhsBytes, const af::dim4& rhsStrides,
                    af_mat_prop lhsOption, af_mat_prop rhsOption,
                    af_dtype type);

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
    launchMetalDot(output.get(), sizeof(T), lhs.get(), lhsElements * sizeof(T),
                   lhs.dims(), lhs.strides(), rhs.get(),
                   rhsElements * sizeof(T), rhs.strides(), lhsOption, rhsOption,
                   static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
