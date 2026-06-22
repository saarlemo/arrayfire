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
#include <gradient.hpp>
#include <kernel/gradient.hpp>
#include <math.hpp>
#include <metal_compute.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <stdexcept>

namespace arrayfire {
namespace metal {

template<typename T>
void gradient(Array<T> &grad0, Array<T> &grad1, const Array<T> &in) {
    const af_dtype type = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (kernel::supportsMetalGradient(type)) {
        getQueue().enqueue(kernel::gradientMetal<T>, grad0, grad1, in);
    } else {
        // Apple GPUs do not expose FP64 in Metal.
        getQueue().enqueue(kernel::gradient<T>, grad0, grad1, in);
    }
}

#define INSTANTIATE(T)                                            \
    template void gradient<T>(Array<T> & grad0, Array<T> & grad1, \
                              const Array<T> &in);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}  // namespace metal
}  // namespace arrayfire
