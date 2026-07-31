/*******************************************************
 * Copyright (c) 2015, Arrayfire
 * all rights reserved.
 *
 * This file is distributed under 3-clause bsd license.
 * the complete license agreement can be obtained at:
 * http://Arrayfire.com/licenses/bsd-3-clause
 ********************************************************/

#include <Array.hpp>
#include <kernel/susan.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/features.h>
#include <cmath>
#include <memory>
#include <type_traits>

using af::features;
using std::shared_ptr;

namespace arrayfire {
namespace metal {

template<typename T>
unsigned susan(Array<float> &x_out, Array<float> &y_out, Array<float> &resp_out,
               const Array<T> &in, const unsigned radius, const float diff_thr,
               const float geom_thr, const float feature_ratio,
               const unsigned edge) {
    dim4 idims                = in.dims();
    const unsigned corner_lim = in.elements() * feature_ratio;

    auto x_corners    = createEmptyArray<float>(dim4(corner_lim));
    auto y_corners    = createEmptyArray<float>(dim4(corner_lim));
    auto resp_corners = createEmptyArray<float>(dim4(corner_lim));
    auto response     = createEmptyArray<T>(dim4(in.elements()));
    auto corners_found                      = memAlloc<unsigned>(1);
    bufferData<unsigned>(corners_found.get())[0] = 0;

    if constexpr (std::is_same<T, double>::value) {
        AF_ERROR("Double input is not supported by the Metal SUSAN kernel",
                 AF_ERR_NOT_SUPPORTED);
    } else {
        getQueue().enqueueNative(kernel::susanResponseMetal<T>, response, in,
                                 idims[0], idims[1], radius, diff_thr, geom_thr,
                                 edge);
        getQueue().enqueueNative(
            kernel::susanNonMax<T>, x_corners, y_corners, resp_corners,
            response, bufferData<unsigned>(corners_found.get()), idims[0],
            idims[1], edge, corner_lim);
    }
    getQueue().sync();

    const unsigned corners_out =
        min(bufferData<unsigned>(corners_found.get())[0], corner_lim);
    if (corners_out == 0) {
        x_out    = createEmptyArray<float>(dim4());
        y_out    = createEmptyArray<float>(dim4());
        resp_out = createEmptyArray<float>(dim4());
        return 0;
    } else {
        x_out    = x_corners;
        y_out    = y_corners;
        resp_out = resp_corners;
        x_out.resetDims(dim4(corners_out));
        y_out.resetDims(dim4(corners_out));
        resp_out.resetDims(dim4(corners_out));
        return corners_out;
    }
}

#define INSTANTIATE(T)                                                        \
    template unsigned susan<T>(                                               \
        Array<float> & x_out, Array<float> & y_out, Array<float> & score_out, \
        const Array<T> &in, const unsigned radius, const float diff_thr,      \
        const float geom_thr, const float feature_ratio, const unsigned edge);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace metal
}  // namespace arrayfire
