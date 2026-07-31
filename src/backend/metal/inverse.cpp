/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/err_common.hpp>
#include <inverse.hpp>

#if defined(WITH_LINEAR_ALGEBRA)

#include <err_metal.hpp>
#include <handle.hpp>
#include <metal_inverse.hpp>
#include <af/dim4.hpp>

#include <identity.hpp>
#include <platform.hpp>
#include <solve.hpp>

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> inverse(const Array<T> &in) {
    int M = in.dims()[0];
    int N = in.dims()[1];

    if (M != N) {
        Array<T> I = identity<T>(in.dims());
        return solve(in, I);
    }

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (type != f32 && type != c32) {
        AF_ERROR("Metal inverse supports only single-precision values",
                 AF_ERR_NOT_SUPPORTED);
    }

    Array<T> input  = copyArray<T>(in);
    Array<T> output = createEmptyArray<T>(in.dims());
    input.device();
    output.device();
    inverseMatrix(output.getBuffer(), input.getBuffer(), in.dims(), type);
    return output;
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace metal
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> inverse(const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

#define INSTANTIATE(T) template Array<T> inverse<T>(const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)

}  // namespace metal
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA
