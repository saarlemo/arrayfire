/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <index.hpp>
#include <kernel/topk.hpp>
#include <platform.hpp>
#include <sort_index.hpp>

#include <type_traits>
#include <vector>

using arrayfire::common::half;
using std::vector;

namespace arrayfire {
namespace metal {

namespace {

vector<af_index_t> indexForTopK(const int k) {
    af_index_t first;
    first.idx.seq = af_seq{0.0, static_cast<double>(k) - 1.0, 1.0};
    first.isSeq   = true;
    first.isBatch = false;

    af_index_t span;
    span.idx.seq = af_span;
    span.isSeq   = true;
    span.isBatch = false;

    return {first, span, span, span};
}

}  // namespace

template<typename T>
void topk(Array<T>& vals, Array<unsigned>& idxs, const Array<T>& in,
          const int k, const int dim, const af::topkFunction order) {
    if constexpr (!std::is_same<T, double>::value) {
        auto outputDims = in.dims();
        outputDims[0]   = k;
        vals            = createEmptyArray<T>(outputDims);
        idxs            = createEmptyArray<unsigned>(outputDims);
        getQueue().enqueueNative(kernel::topKMetal<T>, vals, idxs, in, k,
                                 bool(order & AF_TOPK_MIN));
        return;
    } else {
        auto indices   = createEmptyArray<unsigned>(in.dims());
        auto selection = indexForTopK(k);
        auto sorted    = createEmptyArray<T>(in.dims());
        sort_index(sorted, indices, in, dim, order & AF_TOPK_MIN);
        vals = index<T>(sorted, selection.data());
        idxs = index<unsigned>(indices, selection.data());
    }
}

#define INSTANTIATE(T)                                                  \
    template void topk<T>(Array<T>&, Array<unsigned>&, const Array<T>&, \
                          const int, const int, const af::topkFunction);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(long long)
INSTANTIATE(unsigned long long)
INSTANTIATE(half)
}  // namespace metal
}  // namespace arrayfire
