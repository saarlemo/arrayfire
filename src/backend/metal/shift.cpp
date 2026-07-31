/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <shift.hpp>

#include <jit/ShiftNode.hpp>
#include <types.hpp>

#include <array>
#include <memory>
#include <string>

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> shift(const Array<T> &in, const int shifts[4]) {
    // Shift nodes directly reference a buffer node, so materialize the input
    // before attaching the shifted view to a larger JIT expression.
    in.eval();

    af::dim4 outDims = in.dims();
    std::array<int, 4> normalized{};
    for (int dim = 0; dim < 4; ++dim) {
        normalized[dim] =
            -(shifts[dim] % static_cast<int>(outDims[dim])) +
            outDims[dim] * (shifts[dim] > 0);
    }

    auto node = std::make_shared<jit::ShiftNode>(
        static_cast<af::dtype>(af::dtype_traits<T>::af_type),
        std::static_pointer_cast<jit::BufferNode>(in.getNode()), normalized);
    return createNodeArray<T>(outDims, std::move(node));
}

#define INSTANTIATE(T) \
    template Array<T> shift<T>(const Array<T> &in, const int shifts[4]);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(short)
INSTANTIATE(ushort)

#undef INSTANTIATE

}  // namespace metal
}  // namespace arrayfire
