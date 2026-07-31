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

bool supportsMetalSparse(af_dtype type) noexcept;

void launchMetalDenseToCsr(BufferParam values, size_t valuesBytes,
                           BufferParam rowIdx, size_t rowIdxBytes,
                           BufferParam colIdx, size_t colIdxBytes,
                           BufferParam input, size_t inputBytes,
                           const af::dim4& inputDims,
                           const af::dim4& inputStrides, size_t nonzeros,
                           af_dtype type);

void launchMetalCsrToCoo(BufferParam outputValues, size_t outputValuesBytes,
                         BufferParam outputRows, size_t outputRowsBytes,
                         BufferParam outputColumns, size_t outputColumnsBytes,
                         BufferParam inputValues, size_t inputValuesBytes,
                         BufferParam inputRows, size_t inputRowsBytes,
                         BufferParam inputColumns, size_t inputColumnsBytes,
                         size_t nonzeros, size_t rows, af_dtype type);

void launchMetalCooToCsr(BufferParam outputValues, size_t outputValuesBytes,
                         BufferParam outputRowIdx, size_t outputRowIdxBytes,
                         BufferParam outputColumns, size_t outputColumnsBytes,
                         BufferParam inputValues, size_t inputValuesBytes,
                         BufferParam inputRows, size_t inputRowsBytes,
                         BufferParam inputColumns, size_t inputColumnsBytes,
                         BufferParam cursor, size_t cursorBytes, size_t rows,
                         size_t nonzeros, af_dtype type);

void launchMetalSparseToDense(BufferParam output, size_t outputBytes,
                              BufferParam values, size_t valuesBytes,
                              BufferParam rows, size_t rowsBytes,
                              BufferParam columns, size_t columnsBytes,
                              const af::dim4& outputDims,
                              const af::dim4& outputStrides, bool csr,
                              af_dtype type);

template<typename T>
void sparseDenseToCsrMetal(Param<T> values, Param<int> rowIdx,
                           Param<int> colIdx, CParam<T> input) {
    launchMetalDenseToCsr(
        values.bufferParam(), static_cast<size_t>(values.dims().elements()) * sizeof(T),
        rowIdx.bufferParam(), static_cast<size_t>(rowIdx.dims().elements()) * sizeof(int),
        colIdx.bufferParam(), static_cast<size_t>(colIdx.dims().elements()) * sizeof(int),
        input.bufferParam(), static_cast<size_t>(input.dims().elements()) * sizeof(T),
        input.dims(), input.strides(), static_cast<size_t>(values.dims().elements()),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void sparseCsrToCooMetal(Param<T> outputValues, Param<int> outputRows,
                         Param<int> outputColumns, CParam<T> inputValues,
                         CParam<int> inputRows, CParam<int> inputColumns) {
    launchMetalCsrToCoo(
        outputValues.bufferParam(), static_cast<size_t>(outputValues.dims().elements()) * sizeof(T),
        outputRows.bufferParam(), static_cast<size_t>(outputRows.dims().elements()) * sizeof(int),
        outputColumns.bufferParam(), static_cast<size_t>(outputColumns.dims().elements()) * sizeof(int),
        inputValues.bufferParam(), static_cast<size_t>(inputValues.dims().elements()) * sizeof(T),
        inputRows.bufferParam(), static_cast<size_t>(inputRows.dims().elements()) * sizeof(int),
        inputColumns.bufferParam(), static_cast<size_t>(inputColumns.dims().elements()) * sizeof(int),
        static_cast<size_t>(inputValues.dims().elements()),
        static_cast<size_t>(inputRows.dims(0) - 1),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void sparseCooToCsrMetal(Param<T> outputValues, Param<int> outputRowIdx,
                         Param<int> outputColumns, CParam<T> inputValues,
                         CParam<int> inputRows, CParam<int> inputColumns,
                         Param<int> cursor) {
    launchMetalCooToCsr(
        outputValues.bufferParam(), static_cast<size_t>(outputValues.dims().elements()) * sizeof(T),
        outputRowIdx.bufferParam(), static_cast<size_t>(outputRowIdx.dims().elements()) * sizeof(int),
        outputColumns.bufferParam(), static_cast<size_t>(outputColumns.dims().elements()) * sizeof(int),
        inputValues.bufferParam(), static_cast<size_t>(inputValues.dims().elements()) * sizeof(T),
        inputRows.bufferParam(), static_cast<size_t>(inputRows.dims().elements()) * sizeof(int),
        inputColumns.bufferParam(), static_cast<size_t>(inputColumns.dims().elements()) * sizeof(int),
        cursor.bufferParam(), static_cast<size_t>(cursor.dims().elements()) * sizeof(int),
        static_cast<size_t>(outputRowIdx.dims(0) - 1),
        static_cast<size_t>(inputValues.dims().elements()),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void sparseToDenseMetal(Param<T> output, CParam<T> values,
                        CParam<int> rows, CParam<int> columns, bool csr) {
    launchMetalSparseToDense(
        output.bufferParam(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        values.bufferParam(), static_cast<size_t>(values.dims().elements()) * sizeof(T),
        rows.bufferParam(), static_cast<size_t>(rows.dims().elements()) * sizeof(int),
        columns.bufferParam(), static_cast<size_t>(columns.dims().elements()) * sizeof(int),
        output.dims(), output.strides(), csr,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
