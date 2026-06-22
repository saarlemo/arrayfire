/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>  // header with cpu backend specific
                      // Array class implementation that inherits
                      // ArrayInfo base class

#include <exampleFunction.hpp>         // cpu backend function header
#include <kernel/exampleFunction.hpp>  // Function implementation header
#include <metal_compute_example_function.hpp>

#include <err_metal.hpp>  // error check functions and Macros
                          // specific to cpu backend
#include <platform.hpp>
#include <af/dim4.hpp>

#include <type_traits>

using af::dim4;

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> exampleFunction(const Array<T> &a, const Array<T> &b,
                         const af_someenum_t method) {
    dim4 outputDims = a.dims();

    Array<T> out = createEmptyArray<T>(outputDims);
    // Please use the create***Array<T> helper
    // functions defined in Array.hpp to create
    // different types of Arrays. Please check the
    // file to know what are the different types you
    // can create.

    if constexpr (std::is_same_v<T, float>) {
        bool nonnegativeStrides = true;
        for (int i = 0; i < 4; ++i) {
            nonnegativeStrides &= a.strides()[i] >= 0 && b.strides()[i] >= 0;
        }
        if (nonnegativeStrides) {
            getQueue().enqueue(kernel::exampleFunctionMetal, out, a, b);
        } else {
            getQueue().enqueue(kernel::exampleFunction<T>, out, a, b, method);
        }
    } else {
        getQueue().enqueue(kernel::exampleFunction<T>, out, a, b, method);
    }

    return out;  // return the result
}

#define INSTANTIATE(T)                                                         \
    template Array<T> exampleFunction<T>(const Array<T> &a, const Array<T> &b, \
                                         const af_someenum_t method);

// INSTANTIATIONS for all the types which
// are present in the switch case statement
// in src/api/c/exampleFunction.cpp should be available
INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

}  // namespace metal
}  // namespace arrayfire
