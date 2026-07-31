/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <convolve.hpp>
#include <copy.hpp>
#include <gradient.hpp>
#include <harris.hpp>
#include <kernel/harris.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <sort_index.hpp>
#include <af/dim4.hpp>
#include <cstring>
#include <type_traits>

using af::dim4;

namespace arrayfire {
namespace metal {

template<typename T, typename convAccT>
unsigned harris(Array<float> &x_out, Array<float> &y_out,
                Array<float> &resp_out, const Array<T> &in,
                const unsigned max_corners, const float min_response,
                const float sigma, const unsigned filter_len,
                const float k_thr) {
    if constexpr (!std::is_same<T, float>::value) {
        AF_ERROR("Metal Harris supports only single-precision input",
                 AF_ERR_NOT_SUPPORTED);
        return 0;
    } else {
        dim4 idims = in.dims();

    // Window filter
    Array<convAccT> filter =
        createEmptyArray<convAccT>(dim4(filter_len));
    convAccT *hostFilter = filter.getHostPtr();
    // Decide between rectangular or circular filter
    if (sigma < 0.5f) {
        for (unsigned i = 0; i < filter_len; i++) {
            hostFilter[i] = static_cast<T>(1) / (filter_len);
        }
    } else {
        gaussian1D<convAccT>(hostFilter, static_cast<int>(filter_len), sigma);
    }
    unsigned border_len = filter_len / 2 + 1;

    Array<T> ix = createEmptyArray<T>(idims);
    Array<T> iy = createEmptyArray<T>(idims);

    // Compute first order derivatives
    gradient<T>(iy, ix, in);

    Array<T> ixx = createEmptyArray<T>(idims);
    Array<T> ixy = createEmptyArray<T>(idims);
    Array<T> iyy = createEmptyArray<T>(idims);

    // Compute second-order derivatives
        getQueue().enqueueNative(kernel::harrisSecondOrderMetal, ixx, ixy, iyy,
                                  ix, iy);

    // Convolve second-order derivatives with proper window filter
    ixx = convolve2<T, convAccT>(ixx, filter, filter, false);
    ixy = convolve2<T, convAccT>(ixy, filter, filter, false);
    iyy = convolve2<T, convAccT>(iyy, filter, filter, false);

    const unsigned corner_lim = in.elements() * 0.2f;

    Array<T> responses = createEmptyArray<T>(dim4(in.elements()));

        getQueue().enqueueNative(kernel::harrisResponseMetal, responses,
                                  idims[0], idims[1], ixx, ixy, iyy, k_thr,
                                  border_len);

    Array<float> xCorners    = createEmptyArray<float>(dim4(corner_lim));
    Array<float> yCorners    = createEmptyArray<float>(dim4(corner_lim));
    Array<float> respCorners = createEmptyArray<float>(dim4(corner_lim));

    const unsigned min_r =
        (max_corners > 0) ? 0U : static_cast<unsigned>(min_response);

    // Performs non-maximal suppression
        unsigned corners_found = 0;
        getQueue().enqueueNative(kernel::harrisNonMaxMetal, responses, xCorners,
                                  yCorners, respCorners, &corners_found,
                                  idims[0], idims[1], static_cast<float>(min_r),
                                  border_len, corner_lim);

    const unsigned corners_out =
        min(corners_found, (max_corners > 0) ? max_corners : corner_lim);
    if (corners_out == 0) { return 0; }

    if (max_corners > 0 && corners_found > corners_out) {
        respCorners.resetDims(dim4(corners_found));
        Array<float> harris_sorted =
            createEmptyArray<float>(dim4(corners_found));
        Array<unsigned> harris_idx =
            createEmptyArray<unsigned>(dim4(corners_found));

        // Sort Harris responses
        sort_index<float>(harris_sorted, harris_idx, respCorners, 0, false);

        x_out    = createEmptyArray<float>(dim4(corners_out));
        y_out    = createEmptyArray<float>(dim4(corners_out));
        resp_out = createEmptyArray<float>(dim4(corners_out));

        // Keep only the corners with higher Harris responses
            getQueue().enqueueNative(kernel::harrisKeepCornersMetal, x_out,
                                      y_out, resp_out, xCorners, yCorners,
                                      harris_sorted, harris_idx, corners_out);
    } else if (max_corners == 0 && corners_found < corner_lim) {
        x_out    = createEmptyArray<float>(dim4(corners_out));
        y_out    = createEmptyArray<float>(dim4(corners_out));
        resp_out = createEmptyArray<float>(dim4(corners_out));

        const size_t bytes = corners_out * sizeof(float);
        const auto copy    = [bytes](Array<float> &destination,
                                 const Array<float> &source) {
            const BufferParam dst = destination.bufferParam();
            const BufferParam src = source.bufferParam();
            if (!copyBuffer(dst.buffer, dst.offset, src.buffer, src.offset,
                            bytes)) {
                AF_ERROR("Could not copy Metal Harris buffers",
                         AF_ERR_RUNTIME);
            }
        };
        copy(x_out, xCorners);
        copy(y_out, yCorners);
        copy(resp_out, respCorners);
    } else {
        x_out    = xCorners;
        y_out    = yCorners;
        resp_out = respCorners;
        x_out.resetDims(dim4(corners_out));
        y_out.resetDims(dim4(corners_out));
        resp_out.resetDims(dim4(corners_out));
    }

        return corners_out;
    }
}

#define INSTANTIATE(T, convAccT)                                              \
    template unsigned harris<T, convAccT>(                                    \
        Array<float> & x_out, Array<float> & y_out, Array<float> & score_out, \
        const Array<T> &in, const unsigned max_corners,                       \
        const float min_response, const float sigma,                          \
        const unsigned block_size, const float k_thr);

INSTANTIATE(double, double)
INSTANTIATE(float, float)

}  // namespace metal
}  // namespace arrayfire
