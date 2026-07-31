/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <optypes.hpp>

namespace arrayfire {
namespace metal {

template<typename To, typename Ti, af_op_t op>
struct BinOp;

#define METAL_BINARY_OP(OP, NAME)             \
    template<typename To, typename Ti>        \
    struct BinOp<To, Ti, OP> {                \
        const char *name() { return NAME; }   \
    };

METAL_BINARY_OP(af_add_t, "af_jit_add")
METAL_BINARY_OP(af_sub_t, "af_jit_sub")
METAL_BINARY_OP(af_mul_t, "af_jit_mul")
METAL_BINARY_OP(af_div_t, "af_jit_div")
METAL_BINARY_OP(af_eq_t, "af_jit_eq")
METAL_BINARY_OP(af_neq_t, "af_jit_neq")
METAL_BINARY_OP(af_lt_t, "af_jit_lt")
METAL_BINARY_OP(af_le_t, "af_jit_le")
METAL_BINARY_OP(af_gt_t, "af_jit_gt")
METAL_BINARY_OP(af_ge_t, "af_jit_ge")
METAL_BINARY_OP(af_and_t, "af_jit_and")
METAL_BINARY_OP(af_or_t, "af_jit_or")
METAL_BINARY_OP(af_bitand_t, "af_jit_bitand")
METAL_BINARY_OP(af_bitor_t, "af_jit_bitor")
METAL_BINARY_OP(af_bitxor_t, "af_jit_bitxor")
METAL_BINARY_OP(af_bitshiftl_t, "af_jit_bitshiftl")
METAL_BINARY_OP(af_bitshiftr_t, "af_jit_bitshiftr")
METAL_BINARY_OP(af_min_t, "af_jit_min")
METAL_BINARY_OP(af_max_t, "af_jit_max")
METAL_BINARY_OP(af_rem_t, "af_jit_rem")
METAL_BINARY_OP(af_mod_t, "af_jit_mod")
METAL_BINARY_OP(af_pow_t, "af_jit_pow")
METAL_BINARY_OP(af_cplx2_t, "af_jit_cplx")
METAL_BINARY_OP(af_atan2_t, "af_jit_atan2")
METAL_BINARY_OP(af_hypot_t, "af_jit_hypot")

#undef METAL_BINARY_OP

}  // namespace metal
}  // namespace arrayfire
