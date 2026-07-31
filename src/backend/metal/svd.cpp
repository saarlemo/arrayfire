/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/err_common.hpp>
#include <err_metal.hpp>
#include <svd.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <kernel/svd.hpp>
#include <platform.hpp>
#include <queue.hpp>

namespace arrayfire {
namespace metal {

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in) {
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalSvd(type)) {
        AF_ERROR("Input type is not supported by the Metal SVD kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    kernel::svdInPlaceMetal<T, Tr>(s, u, vt, in);
}

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in) {
    Array<T> in_copy = copyArray<T>(in);
    svdInPlace(s, u, vt, in_copy);
}

}  // namespace metal
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

template<typename T, typename Tr>
void svd(Array<Tr> &s, Array<T> &u, Array<T> &vt, const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

template<typename T, typename Tr>
void svdInPlace(Array<Tr> &s, Array<T> &u, Array<T> &vt, Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

}  // namespace metal
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

#define INSTANTIATE_SVD(T, Tr)                                           \
    template void svd<T, Tr>(Array<Tr> & s, Array<T> & u, Array<T> & vt, \
                             const Array<T> &in);                        \
    template void svdInPlace<T, Tr>(Array<Tr> & s, Array<T> & u,         \
                                    Array<T> & vt, Array<T> & in);

INSTANTIATE_SVD(float, float)
INSTANTIATE_SVD(double, double)
INSTANTIATE_SVD(cfloat, float)
INSTANTIATE_SVD(cdouble, double)

}  // namespace metal
}  // namespace arrayfire
