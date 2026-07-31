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
#include <binary.hpp>
#include <common/jit/BinaryNode.hpp>
#include <common/jit/UnaryNode.hpp>
#include <optypes.hpp>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {

template<typename To, typename Ti>
Array<To> cplx(const Array<Ti> &lhs, const Array<Ti> &rhs,
               const af::dim4 &outDims) {
    return common::createBinaryNode<To, Ti, af_cplx2_t>(lhs, rhs, outDims);
}

template<typename To, typename Ti>
Array<To> complexUnary(const Array<Ti> &in, const char *name, af_op_t op) {
    common::Node_ptr node = std::make_shared<common::UnaryNode>(
        static_cast<af::dtype>(af::dtype_traits<To>::af_type), name,
        in.getNode(), op);
    return createNodeArray<To>(in.dims(), std::move(node));
}

template<typename To, typename Ti>
Array<To> real(const Array<Ti> &in) {
    return complexUnary<To>(in, "af_jit_real", af_real_t);
}

template<typename To, typename Ti>
Array<To> imag(const Array<Ti> &in) {
    return complexUnary<To>(in, "af_jit_imag", af_imag_t);
}

template<typename To, typename Ti>
Array<To> abs(const Array<Ti> &in) {
    return complexUnary<To>(in, "af_jit_abs", af_abs_t);
}

template<typename T>
Array<T> conj(const Array<T> &in) {
    return complexUnary<T>(in, "af_jit_conj", af_conj_t);
}

}  // namespace metal
}  // namespace arrayfire
