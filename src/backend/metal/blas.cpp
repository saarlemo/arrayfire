/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <blas.hpp>

#include <Array.hpp>
#include <common/cast.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <copy.hpp>
#include <kernel/dot.hpp>
#include <platform.hpp>
#include <types.hpp>

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/traits.hpp>

#include <type_traits>

using af::dtype_traits;
using arrayfire::common::cast;
using arrayfire::common::half;

namespace arrayfire {
namespace metal {

template<typename Ti, typename To>
void gemm(Array<To> &out, af_mat_prop optLhs, af_mat_prop optRhs,
          const To *alpha, const Array<Ti> &lhs, const Array<Ti> &rhs,
          const To *beta) {
    if constexpr (std::is_same_v<Ti, float> ||
                  std::is_same_v<Ti, cfloat>) {
        getQueue().enqueueNative(kernel::gemmMetal<Ti>, out, optLhs, optRhs,
                                 alpha, lhs, rhs, beta);
    } else {
        AF_ERROR("Input type is not supported by the Metal GEMM kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
}

template<>
void gemm<half>(Array<half> &out, af_mat_prop optLhs, af_mat_prop optRhs,
                const half *alpha, const Array<half> &lhs,
                const Array<half> &rhs, const half *beta) {
    Array<float> outArr    = createValueArray<float>(out.dims(), 0);
    const auto float_alpha = static_cast<float>(*alpha);
    const auto float_beta  = static_cast<float>(*beta);
    gemm<float>(outArr, optLhs, optRhs, &float_alpha, cast<float>(lhs),
                cast<float>(rhs), &float_beta);
    copyArray(out, outArr);
}

template<>
void gemm<schar, float>(Array<float> &out, af_mat_prop optLhs,
                        af_mat_prop optRhs, const float *alpha,
                        const Array<schar> &lhs, const Array<schar> &rhs,
                        const float *beta) {
    TYPE_ERROR(3, af_dtype::s8);
}

template<typename T>
Array<T> dot(const Array<T> &lhs, const Array<T> &rhs, af_mat_prop optLhs,
             af_mat_prop optRhs) {
    Array<T> out        = createEmptyArray<T>(af::dim4(1));
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalDot(type))
        AF_ERROR("Input type is not supported by the Metal dot kernel",
                 AF_ERR_NOT_SUPPORTED);
    getQueue().enqueueNative(kernel::dotMetal<T>, out, lhs, rhs, optLhs,
                             optRhs);
    return out;
}

template<>
Array<half> dot<half>(const Array<half> &lhs, const Array<half> &rhs,
                      af_mat_prop optLhs, af_mat_prop optRhs) {
    Array<float> out = dot(cast<float>(lhs), cast<float>(rhs), optLhs, optRhs);
    return cast<half>(out);
}

#undef BT
#undef REINTEPRET_CAST

#define INSTANTIATE_GEMM(TYPE)                                               \
    template void gemm<TYPE>(Array<TYPE> & out, af_mat_prop optLhs,          \
                             af_mat_prop optRhs, const TYPE *alphas,         \
                             const Array<TYPE> &lhs, const Array<TYPE> &rhs, \
                             const TYPE *beta)

INSTANTIATE_GEMM(float);
INSTANTIATE_GEMM(cfloat);
INSTANTIATE_GEMM(double);
INSTANTIATE_GEMM(cdouble);

#define INSTANTIATE_DOT(TYPE)                                                  \
    template Array<TYPE> dot<TYPE>(const Array<TYPE> &lhs,                     \
                                   const Array<TYPE> &rhs, af_mat_prop optLhs, \
                                   af_mat_prop optRhs)

INSTANTIATE_DOT(float);
INSTANTIATE_DOT(double);
INSTANTIATE_DOT(cfloat);
INSTANTIATE_DOT(cdouble);

}  // namespace metal
}  // namespace arrayfire
