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
#include <common/half.hpp>
#include <common/jit/UnaryNode.hpp>
#include <err_metal.hpp>
#include <math.hpp>
#include <optypes.hpp>
#include <traits.hpp>
#include <types.hpp>
#include <af/dim4.hpp>

namespace arrayfire {
namespace metal {

template<typename To, typename Ti>
struct CastOp;

#define METAL_CAST_OP(TYPE, NAME)          \
    template<typename Ti>                  \
    struct CastOp<TYPE, Ti> {              \
        const char *name() { return NAME; } \
    };

METAL_CAST_OP(float, "af_jit_cast_float")
METAL_CAST_OP(double, "af_jit_cast_double")
METAL_CAST_OP(int, "af_jit_cast_int")
METAL_CAST_OP(uint, "af_jit_cast_uint")
METAL_CAST_OP(schar, "af_jit_cast_char")
METAL_CAST_OP(uchar, "af_jit_cast_uchar")
METAL_CAST_OP(short, "af_jit_cast_short")
METAL_CAST_OP(ushort, "af_jit_cast_ushort")
METAL_CAST_OP(intl, "af_jit_cast_long")
METAL_CAST_OP(uintl, "af_jit_cast_ulong")
METAL_CAST_OP(common::half, "af_jit_cast_half")
METAL_CAST_OP(cfloat, "af_jit_cast_cfloat")
METAL_CAST_OP(cdouble, "af_jit_cast_cdouble")
METAL_CAST_OP(char, "af_jit_cast_bool")

#undef METAL_CAST_OP

}  // namespace metal
}  // namespace arrayfire
