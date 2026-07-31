/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_metal.hpp>
#include <kernel/nearest_neighbour.hpp>
#include <math.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <topk.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace metal {

template<typename T, typename To>
void nearest_neighbour(Array<uint>& idx, Array<To>& dist, const Array<T>& query,
                       const Array<T>& train, const uint dist_dim,
                       const uint n_dist, const af_match_type dist_type) {
    uint sample_dim   = (dist_dim == 0) ? 1 : 0;
    const dim4& qDims = query.dims();
    const dim4& tDims = train.dims();
    const dim4 outDims(n_dist, qDims[sample_dim]);
    const dim4 distDims(tDims[sample_dim], qDims[sample_dim]);

    Array<To> tmp_dists = createEmptyArray<To>(distDims);

    idx  = createEmptyArray<uint>(outDims);
    dist = createEmptyArray<To>(outDims);

    const af_dtype inputType =
        static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    const af_dtype outputType =
        static_cast<af_dtype>(af::dtype_traits<To>::af_type);
    if (!kernel::supportsMetalNearestNeighbour(inputType, outputType,
                                               dist_type)) {
        AF_ERROR("Nearest-neighbour type combination is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
    getQueue().enqueueNative(kernel::nearestNeighbourMetal<T, To>, tmp_dists,
                             query, train, dist_dim, dist_type);

    metal::topk(dist, idx, tmp_dists, n_dist, 0, AF_TOPK_MIN);
}

#define INSTANTIATE(T, To)                                             \
    template void nearest_neighbour<T, To>(                            \
        Array<uint> & idx, Array<To> & dist, const Array<T>& query,    \
        const Array<T>& train, const uint dist_dim, const uint n_dist, \
        const af_match_type dist_type);

INSTANTIATE(float, float)
INSTANTIATE(double, double)
INSTANTIATE(int, int)
INSTANTIATE(uint, uint)
INSTANTIATE(intl, intl)
INSTANTIATE(uintl, uintl)
INSTANTIATE(schar, int)
INSTANTIATE(uchar, uint)
INSTANTIATE(ushort, uint)
INSTANTIATE(short, int)

INSTANTIATE(uintl, uint)  // For Hamming

}  // namespace metal
}  // namespace arrayfire
