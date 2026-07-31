/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/scan_by_key.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <scan_by_key.hpp>
#include <af/dim4.hpp>
#include <complex>

using af::dim4;

namespace arrayfire {
namespace metal {
template<af_op_t op, typename Ti, typename Tk, typename To>
Array<To> scan(const Array<Tk>& key, const Array<Ti>& in, const int dim,
               bool inclusive_scan) {
    const dim4& dims = in.dims();
    Array<To> out    = createEmptyArray<To>(dims);

    const af_dtype keyType =
        static_cast<af_dtype>(af::dtype_traits<Tk>::af_type);
    const af_dtype valueType =
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type);
    if (!kernel::supportsMetalScanByKey(keyType, valueType))
        AF_ERROR("Type is not supported by the Metal scan-by-key kernel",
                 AF_ERR_NOT_SUPPORTED);

    getQueue().enqueueNative(kernel::scanByKeyMetal<op, Ti, Tk, To>, out, key,
                             in, dim, inclusive_scan);
    return out;
}

#define INSTANTIATE_SCAN_BY_KEY(ROp, Ti, Tk, To)                  \
    template Array<To> scan<ROp, Ti, Tk, To>(                     \
        const Array<Tk>& key, const Array<Ti>& in, const int dim, \
        bool inclusive_scan);

#define INSTANTIATE_SCAN_BY_KEY_ALL(ROp, Tk)           \
    INSTANTIATE_SCAN_BY_KEY(ROp, float, Tk, float)     \
    INSTANTIATE_SCAN_BY_KEY(ROp, double, Tk, double)   \
    INSTANTIATE_SCAN_BY_KEY(ROp, cfloat, Tk, cfloat)   \
    INSTANTIATE_SCAN_BY_KEY(ROp, cdouble, Tk, cdouble) \
    INSTANTIATE_SCAN_BY_KEY(ROp, int, Tk, int)         \
    INSTANTIATE_SCAN_BY_KEY(ROp, uint, Tk, uint)       \
    INSTANTIATE_SCAN_BY_KEY(ROp, intl, Tk, intl)       \
    INSTANTIATE_SCAN_BY_KEY(ROp, uintl, Tk, uintl)

#define INSTANTIATE_SCAN_BY_KEY_ALL_OP(ROp) \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, int)   \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, uint)  \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, intl)  \
    INSTANTIATE_SCAN_BY_KEY_ALL(ROp, uintl)

INSTANTIATE_SCAN_BY_KEY_ALL_OP(af_add_t)
INSTANTIATE_SCAN_BY_KEY_ALL_OP(af_mul_t)
INSTANTIATE_SCAN_BY_KEY_ALL_OP(af_min_t)
INSTANTIATE_SCAN_BY_KEY_ALL_OP(af_max_t)
}  // namespace metal
}  // namespace arrayfire
