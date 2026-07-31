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

void launchMetalGeneralSolve(BufferParam a, const af::dim4& aDims,
                             const af::dim4& aStrides, BufferParam b,
                             const af::dim4& bDims,
                             const af::dim4& bStrides, af_dtype type);

void launchMetalLeastSquares(BufferParam a, const af::dim4& aDims,
                             const af::dim4& aStrides, BufferParam b,
                             const af::dim4& bDims,
                             const af::dim4& bStrides, BufferParam gram,
                             const af::dim4& gramDims,
                             const af::dim4& gramStrides, BufferParam rhs,
                             const af::dim4& rhsDims,
                             const af::dim4& rhsStrides, BufferParam output,
                             const af::dim4& outputDims,
                             const af::dim4& outputStrides, af_dtype type);

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

template<typename T>
void solveGeneralMetal(Param<T> a, Param<T> b) {
    launchMetalGeneralSolve(
        a.bufferParam(), a.dims(), a.strides(), b.bufferParam(), b.dims(),
        b.strides(), static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void solveLeastSquaresMetal(Param<const T> a, Param<const T> b,
                            Param<T> gram, Param<T> rhs, Param<T> output) {
    launchMetalLeastSquares(
        a.bufferParam(), a.dims(), a.strides(), b.bufferParam(), b.dims(),
        b.strides(), gram.bufferParam(), gram.dims(), gram.strides(),
        rhs.bufferParam(), rhs.dims(), rhs.strides(), output.bufferParam(),
        output.dims(), output.strides(),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
