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

#include <cstddef>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalSparseArith(af_dtype type) noexcept;

void launchMetalSparseArithDense(
    BufferParam output, size_t outputBytes, BufferParam values,
    size_t valuesBytes, BufferParam rows, size_t rowsBytes,
    BufferParam columns, size_t columnsBytes, BufferParam rhs,
    size_t rhsBytes, const af::dim4& outputDims, const af::dim4& outputStrides,
    const af::dim4& rhsStrides, bool csr, bool reverse, unsigned operation,
    af_dtype type);

void launchMetalSparseArithValues(
    BufferParam values, size_t valuesBytes, BufferParam rows, size_t rowsBytes,
    BufferParam columns, size_t columnsBytes, BufferParam rhs,
    size_t rhsBytes, const af::dim4& rhsDims, const af::dim4& rhsStrides,
    bool csr, bool reverse, unsigned operation, af_dtype type);

void launchMetalSparseCsrArithCount(
    BufferParam outputRows, size_t outputRowsBytes, BufferParam lhsRows,
    size_t lhsRowsBytes, BufferParam lhsColumns, size_t lhsColumnsBytes,
    BufferParam rhsRows, size_t rhsRowsBytes, BufferParam rhsColumns,
    size_t rhsColumnsBytes, size_t rows, af_dtype type);

void launchMetalSparseCsrArith(
    BufferParam outputValues, size_t outputValuesBytes,
    BufferParam outputColumns, size_t outputColumnsBytes,
    BufferParam outputRows, size_t outputRowsBytes, BufferParam lhsValues,
    size_t lhsValuesBytes, BufferParam lhsRows, size_t lhsRowsBytes,
    BufferParam lhsColumns, size_t lhsColumnsBytes, BufferParam rhsValues,
    size_t rhsValuesBytes, BufferParam rhsRows, size_t rhsRowsBytes,
    BufferParam rhsColumns, size_t rhsColumnsBytes, size_t rows,
    unsigned operation, af_dtype type);

template<typename T>
void sparseArithDenseMetal(Param<T> output, CParam<T> values,
                           CParam<int> rows, CParam<int> columns,
                           CParam<T> rhs, bool csr, bool reverse,
                           unsigned operation) {
    launchMetalSparseArithDense(
        output.bufferParam(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        values.bufferParam(), static_cast<size_t>(values.dims().elements()) * sizeof(T),
        rows.bufferParam(), static_cast<size_t>(rows.dims().elements()) * sizeof(int),
        columns.bufferParam(), static_cast<size_t>(columns.dims().elements()) * sizeof(int),
        rhs.bufferParam(), static_cast<size_t>(rhs.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), rhs.strides(), csr, reverse, operation,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void sparseArithValuesMetal(Param<T> values, CParam<int> rows,
                            CParam<int> columns, CParam<T> rhs, bool csr,
                            bool reverse, unsigned operation) {
    launchMetalSparseArithValues(
        values.bufferParam(), static_cast<size_t>(values.dims().elements()) * sizeof(T),
        rows.bufferParam(), static_cast<size_t>(rows.dims().elements()) * sizeof(int),
        columns.bufferParam(), static_cast<size_t>(columns.dims().elements()) * sizeof(int),
        rhs.bufferParam(), static_cast<size_t>(rhs.dims().elements()) * sizeof(T),
        rhs.dims(), rhs.strides(), csr, reverse, operation,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void sparseCsrArithCountMetal(Param<int> outputRows, CParam<int> lhsRows,
                              CParam<int> lhsColumns, CParam<int> rhsRows,
                              CParam<int> rhsColumns) {
    launchMetalSparseCsrArithCount(
        outputRows.bufferParam(), static_cast<size_t>(outputRows.dims().elements()) * sizeof(int),
        lhsRows.bufferParam(), static_cast<size_t>(lhsRows.dims().elements()) * sizeof(int),
        lhsColumns.bufferParam(), static_cast<size_t>(lhsColumns.dims().elements()) * sizeof(int),
        rhsRows.bufferParam(), static_cast<size_t>(rhsRows.dims().elements()) * sizeof(int),
        rhsColumns.bufferParam(), static_cast<size_t>(rhsColumns.dims().elements()) * sizeof(int),
        static_cast<size_t>(outputRows.dims(0) - 1),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void sparseCsrArithMetal(Param<T> outputValues, Param<int> outputColumns,
                         CParam<int> outputRows, CParam<T> lhsValues,
                         CParam<int> lhsRows, CParam<int> lhsColumns,
                         CParam<T> rhsValues, CParam<int> rhsRows,
                         CParam<int> rhsColumns, unsigned operation) {
    launchMetalSparseCsrArith(
        outputValues.bufferParam(), static_cast<size_t>(outputValues.dims().elements()) * sizeof(T),
        outputColumns.bufferParam(), static_cast<size_t>(outputColumns.dims().elements()) * sizeof(int),
        outputRows.bufferParam(), static_cast<size_t>(outputRows.dims().elements()) * sizeof(int),
        lhsValues.bufferParam(), static_cast<size_t>(lhsValues.dims().elements()) * sizeof(T),
        lhsRows.bufferParam(), static_cast<size_t>(lhsRows.dims().elements()) * sizeof(int),
        lhsColumns.bufferParam(), static_cast<size_t>(lhsColumns.dims().elements()) * sizeof(int),
        rhsValues.bufferParam(), static_cast<size_t>(rhsValues.dims().elements()) * sizeof(T),
        rhsRows.bufferParam(), static_cast<size_t>(rhsRows.dims().elements()) * sizeof(int),
        rhsColumns.bufferParam(), static_cast<size_t>(rhsColumns.dims().elements()) * sizeof(int),
        static_cast<size_t>(outputRows.dims(0) - 1), operation,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
