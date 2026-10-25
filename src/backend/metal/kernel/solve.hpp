/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <af/traits.hpp>
#include <cstddef>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalSolve(af_dtype) noexcept;

void launchMetalTriangularSolve(BufferParam a, const af::dim4& aDims,
                                const af::dim4& aStrides, BufferParam b,
                                const af::dim4& bDims,
                                const af::dim4& bStrides, bool upper,
                                bool unit, af_dtype type);

void launchMetalLUSolve(BufferParam a, const af::dim4& aDims,
                        const af::dim4& aStrides, BufferParam pivot,
                        const af::dim4& pivotDims,
                        const af::dim4& pivotStrides, BufferParam b,
                        const af::dim4& bDims, const af::dim4& bStrides,
                        af_dtype type);

template<typename T>
void solveTriangularMetal(Param<const T> a, Param<T> b, const bool upper,
                          const bool unit) {
    launchMetalTriangularSolve(
        a.bufferParam(), a.dims(), a.strides(), b.bufferParam(), b.dims(),
        b.strides(), upper, unit,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void solveLUMetal(Param<const T> a, Param<const int> pivot, Param<T> b) {
    launchMetalLUSolve(
        a.bufferParam(), a.dims(), a.strides(), pivot.bufferParam(),
        pivot.dims(), pivot.strides(), b.bufferParam(), b.dims(), b.strides(),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
