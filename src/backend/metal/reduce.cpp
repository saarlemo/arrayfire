/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <kernel/reduce.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <reduce.hpp>
#include <af/dim4.hpp>

using af::dim4;
using arrayfire::common::half;

namespace arrayfire {
namespace metal {

template<af_op_t op, typename Ti, typename To>
Array<To> reduce(const Array<Ti> &in, const int dim, bool change_nan,
                 double nanval) {
    dim4 odims = in.dims();
    odims[dim] = 1;

    Array<To> out = createEmptyArray<To>(odims);
    const af_dtype inputType =
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type);
    const af_dtype outputType =
        static_cast<af_dtype>(af::dtype_traits<To>::af_type);
    if (!kernel::supportsMetalReduce(inputType, outputType))
        AF_ERROR("Types are not supported by the Metal reduction kernel",
                 AF_ERR_NOT_SUPPORTED);

    getQueue().enqueueNative(kernel::reduceMetal<op, Ti, To>, out, in, dim,
                             false, change_nan, nanval);
    return out;
}

template<af_op_t op, typename Ti, typename Tk, typename To>
void reduce_by_key(Array<Tk> &keys_out, Array<To> &vals_out,
                   const Array<Tk> &keys, const Array<Ti> &vals, const int dim,
                   bool change_nan, double nanval) {
    dim4 okdims = keys.dims();
    dim4 ovdims = vals.dims();

    Array<Tk> fullsz_okeys = createEmptyArray<Tk>(okdims);
    Array<int> reducedCount = createEmptyArray<int>(1);
    const af_dtype keyType =
        static_cast<af_dtype>(af::dtype_traits<Tk>::af_type);
    const af_dtype inputType =
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type);
    const af_dtype outputType =
        static_cast<af_dtype>(af::dtype_traits<To>::af_type);
    if (!kernel::supportsMetalReduceByKey(keyType, inputType, outputType))
        AF_ERROR("Types are not supported by the Metal reduce-by-key kernel",
                 AF_ERR_NOT_SUPPORTED);
    getQueue().enqueueNative(kernel::reduceByKeyCompactMetal<Tk>, fullsz_okeys,
                             reducedCount, keys);
    getQueue().sync();
    const int n_reduced = reducedCount.getHostPtr()[0];

    okdims[0]   = n_reduced;
    ovdims[dim] = n_reduced;

    std::vector<af_seq> index;
    for (int i = 0; i < keys.ndims(); ++i) {
        af_seq s = {0.0, static_cast<double>(okdims[i]) - 1, 1.0};
        index.push_back(s);
    }
    Array<Tk> okeys = createSubArray<Tk>(fullsz_okeys, index, true);
    Array<To> ovals = createEmptyArray<To>(ovdims);

    getQueue().enqueueNative(kernel::reduceByKeyMetal<op, Ti, Tk, To>, ovals,
                             keys, vals, dim, n_reduced, change_nan, nanval);

    keys_out = okeys;
    vals_out = ovals;
}

template<af_op_t op, typename Ti, typename To>
Array<To> reduce_all(const Array<Ti> &in, bool change_nan, double nanval) {
    Array<To> out = createEmptyArray<To>(1);
    const af_dtype inputType =
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type);
    const af_dtype outputType =
        static_cast<af_dtype>(af::dtype_traits<To>::af_type);
    if (!kernel::supportsMetalReduce(inputType, outputType))
        AF_ERROR("Types are not supported by the Metal reduction kernel",
                 AF_ERR_NOT_SUPPORTED);

    getQueue().enqueueNative(kernel::reduceMetal<op, Ti, To>, out, in, 0, true,
                             change_nan, nanval);
    return out;
}

#define INSTANTIATE(ROp, Ti, To)                                               \
    template Array<To> reduce<ROp, Ti, To>(const Array<Ti> &in, const int dim, \
                                           bool change_nan, double nanval);    \
    template Array<To> reduce_all<ROp, Ti, To>(                                \
        const Array<Ti> &in, bool change_nan, double nanval);                  \
    template void reduce_by_key<ROp, Ti, int, To>(                             \
        Array<int> & keys_out, Array<To> & vals_out, const Array<int> &keys,   \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval); \
    template void reduce_by_key<ROp, Ti, uint, To>(                            \
        Array<uint> & keys_out, Array<To> & vals_out, const Array<uint> &keys, \
        const Array<Ti> &vals, const int dim, bool change_nan, double nanval);

// min
INSTANTIATE(af_min_t, float, float)
INSTANTIATE(af_min_t, double, double)
INSTANTIATE(af_min_t, cfloat, cfloat)
INSTANTIATE(af_min_t, cdouble, cdouble)
INSTANTIATE(af_min_t, int, int)
INSTANTIATE(af_min_t, uint, uint)
INSTANTIATE(af_min_t, intl, intl)
INSTANTIATE(af_min_t, uintl, uintl)
INSTANTIATE(af_min_t, char, char)
INSTANTIATE(af_min_t, schar, schar)
INSTANTIATE(af_min_t, uchar, uchar)
INSTANTIATE(af_min_t, short, short)
INSTANTIATE(af_min_t, ushort, ushort)
INSTANTIATE(af_min_t, half, half)

// max
INSTANTIATE(af_max_t, float, float)
INSTANTIATE(af_max_t, double, double)
INSTANTIATE(af_max_t, cfloat, cfloat)
INSTANTIATE(af_max_t, cdouble, cdouble)
INSTANTIATE(af_max_t, int, int)
INSTANTIATE(af_max_t, uint, uint)
INSTANTIATE(af_max_t, intl, intl)
INSTANTIATE(af_max_t, uintl, uintl)
INSTANTIATE(af_max_t, char, char)
INSTANTIATE(af_max_t, schar, schar)
INSTANTIATE(af_max_t, uchar, uchar)
INSTANTIATE(af_max_t, short, short)
INSTANTIATE(af_max_t, ushort, ushort)
INSTANTIATE(af_max_t, half, half)

// sum
INSTANTIATE(af_add_t, float, float)
INSTANTIATE(af_add_t, double, double)
INSTANTIATE(af_add_t, cfloat, cfloat)
INSTANTIATE(af_add_t, cdouble, cdouble)
INSTANTIATE(af_add_t, int, int)
INSTANTIATE(af_add_t, int, float)
INSTANTIATE(af_add_t, uint, uint)
INSTANTIATE(af_add_t, uint, float)
INSTANTIATE(af_add_t, intl, intl)
INSTANTIATE(af_add_t, intl, double)
INSTANTIATE(af_add_t, uintl, uintl)
INSTANTIATE(af_add_t, uintl, double)
INSTANTIATE(af_add_t, char, int)
INSTANTIATE(af_add_t, char, float)
INSTANTIATE(af_add_t, schar, int)
INSTANTIATE(af_add_t, schar, float)
INSTANTIATE(af_add_t, uchar, uint)
INSTANTIATE(af_add_t, uchar, float)
INSTANTIATE(af_add_t, short, int)
INSTANTIATE(af_add_t, short, float)
INSTANTIATE(af_add_t, ushort, uint)
INSTANTIATE(af_add_t, ushort, float)
INSTANTIATE(af_add_t, half, float)
INSTANTIATE(af_add_t, half, half)

// mul
INSTANTIATE(af_mul_t, float, float)
INSTANTIATE(af_mul_t, double, double)
INSTANTIATE(af_mul_t, cfloat, cfloat)
INSTANTIATE(af_mul_t, cdouble, cdouble)
INSTANTIATE(af_mul_t, int, int)
INSTANTIATE(af_mul_t, uint, uint)
INSTANTIATE(af_mul_t, intl, intl)
INSTANTIATE(af_mul_t, uintl, uintl)
INSTANTIATE(af_mul_t, char, int)
INSTANTIATE(af_mul_t, schar, int)
INSTANTIATE(af_mul_t, uchar, uint)
INSTANTIATE(af_mul_t, short, int)
INSTANTIATE(af_mul_t, ushort, uint)
INSTANTIATE(af_mul_t, half, float)

// count
INSTANTIATE(af_notzero_t, float, uint)
INSTANTIATE(af_notzero_t, double, uint)
INSTANTIATE(af_notzero_t, cfloat, uint)
INSTANTIATE(af_notzero_t, cdouble, uint)
INSTANTIATE(af_notzero_t, int, uint)
INSTANTIATE(af_notzero_t, uint, uint)
INSTANTIATE(af_notzero_t, intl, uint)
INSTANTIATE(af_notzero_t, uintl, uint)
INSTANTIATE(af_notzero_t, char, uint)
INSTANTIATE(af_notzero_t, schar, uint)
INSTANTIATE(af_notzero_t, uchar, uint)
INSTANTIATE(af_notzero_t, short, uint)
INSTANTIATE(af_notzero_t, ushort, uint)
INSTANTIATE(af_notzero_t, half, uint)

// anytrue
INSTANTIATE(af_or_t, float, char)
INSTANTIATE(af_or_t, double, char)
INSTANTIATE(af_or_t, cfloat, char)
INSTANTIATE(af_or_t, cdouble, char)
INSTANTIATE(af_or_t, int, char)
INSTANTIATE(af_or_t, uint, char)
INSTANTIATE(af_or_t, intl, char)
INSTANTIATE(af_or_t, uintl, char)
INSTANTIATE(af_or_t, char, char)
INSTANTIATE(af_or_t, schar, char)
INSTANTIATE(af_or_t, uchar, char)
INSTANTIATE(af_or_t, short, char)
INSTANTIATE(af_or_t, ushort, char)
INSTANTIATE(af_or_t, half, char)

// alltrue
INSTANTIATE(af_and_t, float, char)
INSTANTIATE(af_and_t, double, char)
INSTANTIATE(af_and_t, cfloat, char)
INSTANTIATE(af_and_t, cdouble, char)
INSTANTIATE(af_and_t, int, char)
INSTANTIATE(af_and_t, uint, char)
INSTANTIATE(af_and_t, intl, char)
INSTANTIATE(af_and_t, uintl, char)
INSTANTIATE(af_and_t, char, char)
INSTANTIATE(af_and_t, schar, char)
INSTANTIATE(af_and_t, uchar, char)
INSTANTIATE(af_and_t, short, char)
INSTANTIATE(af_and_t, ushort, char)
INSTANTIATE(af_and_t, half, char)

}  // namespace metal
}  // namespace arrayfire
