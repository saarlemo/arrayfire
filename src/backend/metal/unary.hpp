/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Array.hpp>
#include <common/jit/UnaryNode.hpp>
#include <optypes.hpp>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {

template<af_op_t op>
const char *unaryName();

#define METAL_UNARY_OP(OP, NAME)             \
    template<>                               \
    inline const char *unaryName<OP>() {     \
        return NAME;                         \
    }

METAL_UNARY_OP(af_sin_t, "af_jit_sin")
METAL_UNARY_OP(af_cos_t, "af_jit_cos")
METAL_UNARY_OP(af_tan_t, "af_jit_tan")
METAL_UNARY_OP(af_asin_t, "af_jit_asin")
METAL_UNARY_OP(af_acos_t, "af_jit_acos")
METAL_UNARY_OP(af_atan_t, "af_jit_atan")
METAL_UNARY_OP(af_sinh_t, "af_jit_sinh")
METAL_UNARY_OP(af_cosh_t, "af_jit_cosh")
METAL_UNARY_OP(af_tanh_t, "af_jit_tanh")
METAL_UNARY_OP(af_asinh_t, "af_jit_asinh")
METAL_UNARY_OP(af_acosh_t, "af_jit_acosh")
METAL_UNARY_OP(af_atanh_t, "af_jit_atanh")
METAL_UNARY_OP(af_exp_t, "af_jit_exp")
METAL_UNARY_OP(af_sigmoid_t, "af_jit_sigmoid")
METAL_UNARY_OP(af_expm1_t, "af_jit_expm1")
METAL_UNARY_OP(af_erf_t, "af_jit_erf")
METAL_UNARY_OP(af_erfc_t, "af_jit_erfc")
METAL_UNARY_OP(af_tgamma_t, "af_jit_tgamma")
METAL_UNARY_OP(af_lgamma_t, "af_jit_lgamma")
METAL_UNARY_OP(af_log_t, "af_jit_log")
METAL_UNARY_OP(af_log1p_t, "af_jit_log1p")
METAL_UNARY_OP(af_log10_t, "af_jit_log10")
METAL_UNARY_OP(af_log2_t, "af_jit_log2")
METAL_UNARY_OP(af_sqrt_t, "af_jit_sqrt")
METAL_UNARY_OP(af_rsqrt_t, "af_jit_rsqrt")
METAL_UNARY_OP(af_cbrt_t, "af_jit_cbrt")
METAL_UNARY_OP(af_trunc_t, "af_jit_trunc")
METAL_UNARY_OP(af_round_t, "af_jit_round")
METAL_UNARY_OP(af_signbit_t, "af_jit_signbit")
METAL_UNARY_OP(af_ceil_t, "af_jit_ceil")
METAL_UNARY_OP(af_floor_t, "af_jit_floor")
METAL_UNARY_OP(af_isinf_t, "af_jit_isinf")
METAL_UNARY_OP(af_isnan_t, "af_jit_isnan")
METAL_UNARY_OP(af_iszero_t, "af_jit_iszero")
METAL_UNARY_OP(af_noop_t, "af_jit_noop")
METAL_UNARY_OP(af_bitnot_t, "af_jit_bitnot")

#undef METAL_UNARY_OP

template<typename T, af_op_t op>
Array<T> unaryOp(const Array<T> &in,
                 af::dim4 outDims = af::dim4(-1, -1, -1, -1)) {
    auto createUnary = [](std::array<common::Node_ptr, 1> &operands) {
        return common::Node_ptr(new common::UnaryNode(
            static_cast<af::dtype>(af::dtype_traits<T>::af_type),
            unaryName<op>(), operands[0], op));
    };

    if (outDims == af::dim4(-1, -1, -1, -1)) outDims = in.dims();
    common::Node_ptr node =
        common::createNaryNode<T, 1>(outDims, createUnary, {&in});
    return createNodeArray<T>(outDims, std::move(node));
}

template<typename T, af_op_t op>
Array<char> checkOp(const Array<T> &in,
                    af::dim4 outDims = af::dim4(-1, -1, -1, -1)) {
    auto createUnary = [](std::array<common::Node_ptr, 1> &operands) {
        return common::Node_ptr(new common::UnaryNode(
            static_cast<af::dtype>(af::dtype_traits<char>::af_type),
            unaryName<op>(), operands[0], op));
    };

    if (outDims == af::dim4(-1, -1, -1, -1)) outDims = in.dims();
    common::Node_ptr node =
        common::createNaryNode<T, 1>(outDims, createUnary, {&in});
    return createNodeArray<char>(outDims, std::move(node));
}

}  // namespace metal
}  // namespace arrayfire
