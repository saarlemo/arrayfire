/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <utility.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

template<typename T>
void exampleFunction(Param<T> out, CParam<T> a, CParam<T> b,
                     const af_someenum_t method) {
    UNUSED(method);
    dim4 oDims = out.dims();

    dim4 aStrides = a.strides();  // you can retrieve strides
    dim4 bStrides = b.strides();
    dim4 oStrides = out.strides();

    const T* src1 =
        a.get();  // metal::Param<T>::get returns the pointer to the
                  // memory allocated for that Param (with proper offsets)
    const T* src2 =
        b.get();  // metal::Param<T>::get returns the pointer to the
                  // memory allocated for that Param (with proper offsets)
    T* dst = out.get();

    for (dim_t w = 0; w < oDims[3]; ++w) {
        for (dim_t z = 0; z < oDims[2]; ++z) {
            for (dim_t y = 0; y < oDims[1]; ++y) {
                for (dim_t x = 0; x < oDims[0]; ++x) {
                    const dim_t src1Idx = x * aStrides[0] + y * aStrides[1] +
                                          z * aStrides[2] + w * aStrides[3];
                    const dim_t src2Idx = x * bStrides[0] + y * bStrides[1] +
                                          z * bStrides[2] + w * bStrides[3];
                    const dim_t dstIdx = x * oStrides[0] + y * oStrides[1] +
                                         z * oStrides[2] + w * oStrides[3];
                    dst[dstIdx] = src1[src1Idx] + src2[src2Idx];
                }
            }
        }
    }
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
