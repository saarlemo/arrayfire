/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <sparse_blas.hpp>

#include <common/err_common.hpp>
#include <kernel/sparse_blas.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <types.hpp>

#include <af/dim4.hpp>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> matmul(const common::SparseArray<T> &lhs, const Array<T> &rhs,
                const af_mat_prop optLhs, const af_mat_prop optRhs) {
    // Sparse matmul currently treats the dense operand as non-transposed,
    // matching the previous MKL and host implementations.
    UNUSED(optRhs);
    if (optLhs != AF_MAT_NONE && optLhs != AF_MAT_TRANS &&
        optLhs != AF_MAT_CTRANS) {
        AF_ERROR("INVALID af_mat_prop", AF_ERR_ARG);
    }

    const af::dim4 &lhsDims = lhs.dims();
    const af::dim4 &rhsDims = rhs.dims();
    const int lhsRowDim     = optLhs == AF_MAT_NONE ? 0 : 1;
    const int outputRows    = lhsDims[lhsRowDim];
    const int outputColumns = rhsDims[1];

    Array<T> out = createValueArray<T>(
        af::dim4(outputRows, outputColumns, 1, 1), scalar<T>(0));

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalSparseMatmul(type)) {
        AF_ERROR("Input type is not supported by the Metal sparse matmul kernel",
                 AF_ERR_NOT_SUPPORTED);
    }

    getQueue().enqueueNative(kernel::sparseMatmulMetal<T>, out,
                             lhs.getValues(), lhs.getRowIdx(),
                             lhs.getColIdx(), rhs, lhsDims, optLhs);
    return out;
}

#define INSTANTIATE_SPARSE(T)                                            \
    template Array<T> matmul<T>(const common::SparseArray<T> &lhs,       \
                                const Array<T> &rhs, af_mat_prop optLhs, \
                                af_mat_prop optRhs);

INSTANTIATE_SPARSE(float)
INSTANTIATE_SPARSE(double)
INSTANTIATE_SPARSE(cfloat)
INSTANTIATE_SPARSE(cdouble)

}  // namespace metal
}  // namespace arrayfire
