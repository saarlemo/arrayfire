/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#ifndef AFMETAL_LAPACK
#define AFMETAL_LAPACK

#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#define LAPACK_PREFIX LAPACKE_
#define ORDER_TYPE int
#define AF_LAPACK_COL_MAJOR LAPACK_COL_MAJOR
#define LAPACK_NAME(fn) LAPACKE_##fn

#include <Array.hpp>
#include <copy.hpp>
#include <err_metal.hpp>
#include <queue.hpp>

#include <algorithm>
#include <limits>
#include <optional>

#ifdef USE_MKL
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#else
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#include <common/lapacke.hpp>
#undef AF_LAPACK_COL_MAJOR
#define AF_LAPACK_COL_MAJOR 0
#else  // NETLIB LAPACKE
#include <lapacke.h>
#endif
#endif

namespace arrayfire {
namespace metal {

template<typename T>
size_t lapackStorageSpan(const Array<T>& input) {
    const af::dim4 dims = input.dims();
    const af::dim4 strides = input.strides();
    size_t span = static_cast<size_t>(input.getOffset());
    for (int axis = 0; axis < 4; ++axis) {
        if (dims[axis] > 0) {
            span += static_cast<size_t>(dims[axis] - 1) * strides[axis];
        }
    }
    return span + 1;
}

template<typename T>
Array<T> copyToLapackArray(const Array<T>& input,
                           const af::dim4& outputDims) {
    const af::dim4 dims = input.dims();
    for (int axis = 0; axis < 4; ++axis) {
        if (outputDims[axis] < dims[axis]) {
            AF_ERROR("Accelerate LAPACK staging cannot truncate input",
                     AF_ERR_ARG);
        }
    }

    Array<T> output = createEmptyArray<T>(outputDims);
    if (output.elements() == 0) { return output; }

    input.eval();
    output.eval();
    getQueue().sync();

    const T* source = input.getHostPtr();
    size_t sourceOffset = 0;
    std::optional<Array<T>> sourceStaging;
    if (!source && input.elements() > 0) {
        const size_t span = lapackStorageSpan(input);
        const size_t bytes = span * sizeof(T);
        if (bytes > input.getAllocatedBytes()) {
            AF_ERROR("Accelerate LAPACK input exceeds its Metal buffer",
                     AF_ERR_RUNTIME);
        }
        sourceStaging.emplace(createEmptyArray<T>(af::dim4(span, 1, 1, 1)));
        sourceStaging->eval();
        if (!copyBuffer(sourceStaging->getBuffer(), 0, input.getBuffer(), 0,
                        bytes)) {
            AF_ERROR("Could not stage Metal input for Accelerate LAPACK",
                     AF_ERR_RUNTIME);
        }
        getQueue().sync();
        source = sourceStaging->getHostPtr(false);
        sourceOffset = static_cast<size_t>(input.getOffset());
    }

    T* destination = output.getHostPtr(false);
    if (!destination || (input.elements() > 0 && !source)) {
        AF_ERROR("Accelerate LAPACK requires CPU-visible staging storage",
                 AF_ERR_RUNTIME);
    }
    std::fill(destination, destination + output.elements(), T{});

    const af::dim4 sourceStrides = input.strides();
    const af::dim4 outputStrides = output.strides();
    for (dim_t w = 0; w < dims[3]; ++w) {
        for (dim_t z = 0; z < dims[2]; ++z) {
            for (dim_t y = 0; y < dims[1]; ++y) {
                for (dim_t x = 0; x < dims[0]; ++x) {
                    const size_t sourceIndex =
                        sourceOffset + static_cast<size_t>(
                                           x * sourceStrides[0] +
                                           y * sourceStrides[1] +
                                           z * sourceStrides[2] +
                                           w * sourceStrides[3]);
                    const size_t outputIndex =
                        static_cast<size_t>(x * outputStrides[0] +
                                            y * outputStrides[1] +
                                            z * outputStrides[2] +
                                            w * outputStrides[3]);
                    destination[outputIndex] = source[sourceIndex];
                }
            }
        }
    }
    return output;
}

template<typename T>
Array<T> copyToLapackArray(const Array<T>& input) {
    return copyToLapackArray(input, input.dims());
}

template<typename T>
void copyFromLapackArray(Array<T>& output, const Array<T>& input) {
    if (output.dims() != input.dims()) {
        AF_ERROR("Accelerate LAPACK copy-back shape mismatch", AF_ERR_ARG);
    }
    if (output.elements() == 0) { return; }

    input.eval();
    output.eval();
    getQueue().sync();
    const T* source = input.getHostPtr(false);
    if (!source) {
        AF_ERROR("Accelerate LAPACK result is not CPU-visible", AF_ERR_RUNTIME);
    }

    const af::dim4 dims = output.dims();
    const af::dim4 strides = output.strides();
    T* destination = output.getHostPtr();
    std::optional<Array<T>> destinationStaging;
    size_t destinationOffset = 0;
    if (!destination) {
        const size_t span = lapackStorageSpan(output);
        const size_t bytes = span * sizeof(T);
        if (bytes > output.getAllocatedBytes()) {
            AF_ERROR("Accelerate LAPACK output exceeds its Metal buffer",
                     AF_ERR_RUNTIME);
        }
        destinationStaging.emplace(
            createEmptyArray<T>(af::dim4(span, 1, 1, 1)));
        destinationStaging->eval();
        if (!copyBuffer(destinationStaging->getBuffer(), 0, output.getBuffer(),
                        0, bytes)) {
            AF_ERROR("Could not stage Metal output from Accelerate LAPACK",
                     AF_ERR_RUNTIME);
        }
        getQueue().sync();
        destination = destinationStaging->getHostPtr(false);
        destinationOffset = static_cast<size_t>(output.getOffset());
    }
    if (!destination) {
        AF_ERROR("Accelerate LAPACK requires CPU-visible result staging",
                 AF_ERR_RUNTIME);
    }

    for (dim_t w = 0; w < dims[3]; ++w) {
        for (dim_t z = 0; z < dims[2]; ++z) {
            for (dim_t y = 0; y < dims[1]; ++y) {
                for (dim_t x = 0; x < dims[0]; ++x) {
                    const size_t sourceIndex = static_cast<size_t>(
                        x + y * dims[0] + z * dims[0] * dims[1] +
                        w * dims[0] * dims[1] * dims[2]);
                    const size_t destinationIndex = destinationOffset +
                        static_cast<size_t>(x * strides[0] + y * strides[1] +
                                            z * strides[2] + w * strides[3]);
                    destination[destinationIndex] = source[sourceIndex];
                }
            }
        }
    }

    if (destinationStaging && destinationStaging->elements() > 0) {
        const size_t bytes = destinationStaging->elements() * sizeof(T);
        if (!copyBuffer(output.getBuffer(), 0, destinationStaging->getBuffer(),
                        0, bytes)) {
            AF_ERROR("Could not copy Accelerate LAPACK results to Metal",
                     AF_ERR_RUNTIME);
        }
        getQueue().sync();
    }
}

}  // namespace metal
}  // namespace arrayfire

#endif
