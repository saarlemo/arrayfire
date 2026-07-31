/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <qr.hpp>

#include <err_metal.hpp>

#if defined(WITH_LINEAR_ALGEBRA)
#include <copy.hpp>
#include <identity.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <kernel/qr.hpp>
#include <queue.hpp>
#include <triangle.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace metal {

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];

    const dim4 NullShape(0, 0, 0, 0);

    dim4 endPadding(M - iDims[0], max(M, N) - iDims[1], 0, 0);
    q = (endPadding == NullShape
             ? copyArray(in)
             : padArrayBorders(in, NullShape, endPadding, AF_PAD_ZERO));
    q.resetDims(iDims);
    t = qr_inplace(q);

    // SPLIT into q and r
    dim4 rdims(M, N);
    r = createEmptyArray<T>(rdims);

    triangle<T>(r, q, true, false);

    Array<T> reflectors = copyArray<T>(q);
    q                     = identity<T>(dim4(M, M));
    getQueue().enqueueNative(kernel::qrGenerateMetal<T>, q, reflectors, t);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    dim4 iDims = in.dims();
    int M      = iDims[0];
    int N      = iDims[1];
    Array<T> t = createEmptyArray<T>(af::dim4(min(M, N), 1, 1, 1));

    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalQR(type)) {
        AF_ERROR("Input type is not supported by the Metal QR kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::qrFactorMetal<T>, in, t);

    return t;
}

}  // namespace metal
}  // namespace arrayfire

#else  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

template<typename T>
void qr(Array<T> &q, Array<T> &r, Array<T> &t, const Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

template<typename T>
Array<T> qr_inplace(Array<T> &in) {
    AF_ERROR("Linear Algebra is disabled on Metal", AF_ERR_NOT_CONFIGURED);
}

}  // namespace metal
}  // namespace arrayfire

#endif  // WITH_LINEAR_ALGEBRA

namespace arrayfire {
namespace metal {

#define INSTANTIATE_QR(T)                                         \
    template Array<T> qr_inplace<T>(Array<T> & in);               \
    template void qr<T>(Array<T> & q, Array<T> & r, Array<T> & t, \
                        const Array<T> &in);

INSTANTIATE_QR(float)
INSTANTIATE_QR(cfloat)
INSTANTIATE_QR(double)
INSTANTIATE_QR(cdouble)

}  // namespace metal
}  // namespace arrayfire
