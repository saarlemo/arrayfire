/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <common/kernel_type.hpp>
#include <common/traits.hpp>
#include <af/traits.hpp>
#include <complex>

namespace arrayfire {
namespace common {
class half;
}

namespace metal {

namespace {
template<typename T>
const char *shortname(bool caps = false) {
    return caps ? "X" : "x";
}

template<typename T>
const char *getFullName() {
    return af::dtype_traits<T>::getName();
}

#define METAL_TYPE_NAME(TYPE, SHORT_LOWER, SHORT_UPPER, FULL) \
    template<>                                                 \
    inline const char *shortname<TYPE>(bool caps) {            \
        return caps ? SHORT_UPPER : SHORT_LOWER;                \
    }                                                           \
    template<>                                                 \
    inline const char *getFullName<TYPE>() {                   \
        return FULL;                                            \
    }

METAL_TYPE_NAME(float, "s", "S", "float")
METAL_TYPE_NAME(double, "d", "D", "double")
METAL_TYPE_NAME(std::complex<float>, "c", "C", "float2")
METAL_TYPE_NAME(std::complex<double>, "z", "Z", "double2")
METAL_TYPE_NAME(int, "i", "I", "int")
METAL_TYPE_NAME(unsigned int, "u", "U", "uint")
METAL_TYPE_NAME(char, "j", "J", "char")
METAL_TYPE_NAME(signed char, "a", "A", "char")
METAL_TYPE_NAME(unsigned char, "v", "V", "uchar")
METAL_TYPE_NAME(long long, "l", "L", "long")
METAL_TYPE_NAME(unsigned long long, "k", "K", "ulong")
METAL_TYPE_NAME(short, "p", "P", "short")
METAL_TYPE_NAME(unsigned short, "q", "Q", "ushort")
METAL_TYPE_NAME(arrayfire::common::half, "h", "H", "half")

#undef METAL_TYPE_NAME

}  // namespace

using cdouble = std::complex<double>;
using cfloat  = std::complex<float>;
using intl    = long long;
using uint    = unsigned int;
using schar   = signed char;
using uchar   = unsigned char;
using uintl   = unsigned long long;
using ushort  = unsigned short;

template<typename T>
using compute_t = typename common::kernel_type<T>::compute;

template<typename T>
using data_t = typename common::kernel_type<T>::data;

}  // namespace metal

namespace common {
template<typename T>
struct kernel_type;

class half;

template<>
struct kernel_type<arrayfire::common::half> {
    using data = arrayfire::common::half;

    // These are the types within a kernel
    using native = float;

    using compute = float;
};
}  // namespace common

}  // namespace arrayfire
