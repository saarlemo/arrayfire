/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arith.hpp>
#include <common/SparseArray.hpp>
#include <common/err_common.hpp>
#include <copy.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <sparse.hpp>
#include <sparse_arith.hpp>
#include <af/dim4.hpp>

#include <kernel/sparse_arith.hpp>

using arrayfire::common::createArrayDataSparseArray;
using arrayfire::common::createEmptySparseArray;
using arrayfire::common::SparseArray;

namespace arrayfire {
namespace metal {

namespace {

template<af_op_t op>
unsigned operationCode() {
    return op == af_add_t   ? 0U
           : op == af_sub_t ? 1U
           : op == af_mul_t ? 2U
                            : 3U;
}

template<typename T>
void checkSparseArithType() {
    if (!kernel::supportsMetalSparseArith(
            static_cast<af_dtype>(af::dtype_traits<T>::af_type))) {
        AF_ERROR("Metal sparse arithmetic type is not supported",
                 AF_ERR_NOT_SUPPORTED);
    }
}

}  // namespace

template<typename T, af_op_t op>
Array<T> arithOpD(const SparseArray<T> &lhs, const Array<T> &rhs,
                  const bool reverse) {
    checkSparseArithType<T>();
    Array<T> out  = createEmptyArray<T>(dim4(0));
    Array<T> zero = createValueArray<T>(rhs.dims(), scalar<T>(0));
    switch (op) {
        case af_add_t: out = copyArray<T>(rhs); break;
        case af_sub_t:
            out = reverse ? copyArray<T>(rhs)
                          : arithOp<T, af_sub_t>(zero, rhs, rhs.dims());
            break;
        default: out = copyArray<T>(rhs);
    }

    const bool csr = lhs.getStorage() == AF_STORAGE_CSR;
    if (!csr && lhs.getStorage() != AF_STORAGE_COO) {
        AF_ERROR("Sparse Arithmetic only supported for CSR or COO",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(
        kernel::sparseArithDenseMetal<T>, out, lhs.getValues(), lhs.getRowIdx(),
        lhs.getColIdx(), rhs, csr, reverse, operationCode<op>());
    return out;
}

template<typename T, af_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const Array<T> &rhs,
                       const bool reverse) {
    checkSparseArithType<T>();
    SparseArray<T> out = createArrayDataSparseArray<T>(
        lhs.dims(), lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
        lhs.getStorage(), true);
    const bool csr = out.getStorage() == AF_STORAGE_CSR;
    if (!csr && out.getStorage() != AF_STORAGE_COO) {
        AF_ERROR("Sparse Arithmetic only supported for CSR or COO",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(
        kernel::sparseArithValuesMetal<T>, out.getValues(), out.getRowIdx(),
        out.getColIdx(), rhs, csr, reverse, operationCode<op>());
    return out;
}

template<typename T, af_op_t op>
SparseArray<T> arithOp(const SparseArray<T> &lhs, const SparseArray<T> &rhs) {
    checkSparseArithType<T>();
    const af::storage sfmt = lhs.getStorage();
    if (sfmt != AF_STORAGE_CSR || rhs.getStorage() != AF_STORAGE_CSR) {
        AF_ERROR("Sparse-sparse arithmetic only supported for CSR",
                 AF_ERR_NOT_SUPPORTED);
    }

    const dim4 &dims = lhs.dims();
    const uint M     = dims[0];
    auto rowArr      = createEmptyArray<int>(dim4(M + 1));

    getQueue().enqueueNative(kernel::sparseCsrArithCountMetal<T>, rowArr,
                              lhs.getRowIdx(), lhs.getColIdx(), rhs.getRowIdx(),
                              rhs.getColIdx());
    getQueue().sync();

    const uint nnz = static_cast<uint>(rowArr.getHostPtr()[M]);
    auto out       = createEmptySparseArray<T>(dims, nnz, sfmt);
    copyArray(out.getRowIdx(), rowArr);

    getQueue().enqueueNative(
        kernel::sparseCsrArithMetal<T>, out.getValues(), out.getColIdx(),
        out.getRowIdx(), lhs.getValues(), lhs.getRowIdx(), lhs.getColIdx(),
        rhs.getValues(), rhs.getRowIdx(), rhs.getColIdx(), operationCode<op>());
    return out;
}

#define INSTANTIATE(T)                                                         \
    template Array<T> arithOpD<T, af_add_t>(                                   \
        const SparseArray<T> &, const Array<T> &, const bool);                 \
    template Array<T> arithOpD<T, af_sub_t>(                                   \
        const SparseArray<T> &, const Array<T> &, const bool);                 \
    template Array<T> arithOpD<T, af_mul_t>(                                   \
        const SparseArray<T> &, const Array<T> &, const bool);                 \
    template Array<T> arithOpD<T, af_div_t>(                                   \
        const SparseArray<T> &, const Array<T> &, const bool);                 \
    template SparseArray<T> arithOp<T, af_add_t>(                              \
        const SparseArray<T> &, const Array<T> &, const bool);                 \
    template SparseArray<T> arithOp<T, af_sub_t>(                              \
        const SparseArray<T> &, const Array<T> &, const bool);                 \
    template SparseArray<T> arithOp<T, af_mul_t>(                              \
        const SparseArray<T> &, const Array<T> &, const bool);                 \
    template SparseArray<T> arithOp<T, af_div_t>(                              \
        const SparseArray<T> &, const Array<T> &, const bool);                 \
    template SparseArray<T> arithOp<T, af_add_t>(                              \
        const common::SparseArray<T> &, const common::SparseArray<T> &);       \
    template SparseArray<T> arithOp<T, af_sub_t>(                              \
        const common::SparseArray<T> &, const common::SparseArray<T> &);       \
    template SparseArray<T> arithOp<T, af_mul_t>(                              \
        const common::SparseArray<T> &, const common::SparseArray<T> &);       \
    template SparseArray<T> arithOp<T, af_div_t>(                              \
        const common::SparseArray<T> &, const common::SparseArray<T> &);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

#undef INSTANTIATE

}  // namespace metal
}  // namespace arrayfire
