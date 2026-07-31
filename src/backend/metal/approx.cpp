/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <approx.hpp>
#include <err_metal.hpp>
#include <kernel/approx.hpp>
#include <platform.hpp>
#include <af/dim4.hpp>

#include <type_traits>

namespace arrayfire {
namespace metal {

template<typename Ty, typename Tp>
void approx1(Array<Ty> &yo, const Array<Ty> &yi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const af_interp_type method, const float offGrid) {
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
        case AF_INTERP_LINEAR:
        case AF_INTERP_LINEAR_COSINE:
        case AF_INTERP_CUBIC:
        case AF_INTERP_CUBIC_SPLINE: break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
    }
    if constexpr (std::is_same_v<Tp, float> &&
                  (std::is_same_v<Ty, float> ||
                   std::is_same_v<Ty, cfloat>)) {
        const af_dtype valueType =
            static_cast<af_dtype>(af::dtype_traits<Ty>::af_type);
        const af_dtype positionType =
            static_cast<af_dtype>(af::dtype_traits<Tp>::af_type);
        if (!kernel::supportsMetalApprox(valueType, positionType)) {
            AF_ERROR("Approximation type or layout is not supported by Metal",
                     AF_ERR_NOT_SUPPORTED);
        }
        getQueue().enqueueNative(kernel::approx1Metal<Ty>, yo, yi, xo, xdim,
                                 xi_beg, xi_step, offGrid, method);
    } else {
        AF_ERROR("Approximation type or layout is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
}

template<typename Ty, typename Tp>
void approx2(Array<Ty> &zo, const Array<Ty> &zi, const Array<Tp> &xo,
             const int xdim, const Tp &xi_beg, const Tp &xi_step,
             const Array<Tp> &yo, const int ydim, const Tp &yi_beg,
             const Tp &yi_step, const af_interp_type method,
             const float offGrid) {
    switch (method) {
        case AF_INTERP_NEAREST:
        case AF_INTERP_LOWER:
        case AF_INTERP_LINEAR:
        case AF_INTERP_BILINEAR:
        case AF_INTERP_LINEAR_COSINE:
        case AF_INTERP_BILINEAR_COSINE:
        case AF_INTERP_CUBIC:
        case AF_INTERP_BICUBIC:
        case AF_INTERP_CUBIC_SPLINE:
        case AF_INTERP_BICUBIC_SPLINE: break;
        default: AF_ERROR("Unsupported interpolation type", AF_ERR_ARG);
    }
    if constexpr (std::is_same_v<Tp, float> &&
                  (std::is_same_v<Ty, float> ||
                   std::is_same_v<Ty, cfloat>)) {
        const af_dtype valueType =
            static_cast<af_dtype>(af::dtype_traits<Ty>::af_type);
        const af_dtype positionType =
            static_cast<af_dtype>(af::dtype_traits<Tp>::af_type);
        if (!kernel::supportsMetalApprox(valueType, positionType)) {
            AF_ERROR("Approximation type or layout is not supported by Metal",
                     AF_ERR_NOT_SUPPORTED);
        }
        getQueue().enqueueNative(kernel::approx2Metal<Ty>, zo, zi, xo, xdim,
                                 xi_beg, xi_step, yo, ydim, yi_beg, yi_step,
                                 offGrid, method);
    } else {
        AF_ERROR("Approximation type or layout is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
}

#define INSTANTIATE(Ty, Tp)                                       \
    template void approx1<Ty, Tp>(                                \
        Array<Ty> & yo, const Array<Ty> &yi, const Array<Tp> &xo, \
        const int xdim, const Tp &xi_beg, const Tp &xi_step,      \
        const af_interp_type method, const float offGrid);        \
    template void approx2<Ty, Tp>(                                \
        Array<Ty> & zo, const Array<Ty> &zi, const Array<Tp> &xo, \
        const int xdim, const Tp &xi_beg, const Tp &xi_step,      \
        const Array<Tp> &yo, const int ydim, const Tp &yi_beg,    \
        const Tp &yi_step, const af_interp_type method, const float offGrid);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(cfloat, float)
INSTANTIATE(cdouble, double)

}  // namespace metal
}  // namespace arrayfire
