/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <flood_fill.hpp>

#include <err_metal.hpp>
#include <kernel/flood_fill.hpp>

using af::connectivity;

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> floodFill(const Array<T>& image, const Array<uint>& seedsX,
                   const Array<uint>& seedsY, const T newValue,
                   const T lowValue, const T highValue,
                   const af::connectivity nlookup) {
    UNUSED(nlookup);
    Array<T> out = createEmptyArray<T>(image.dims());
    Array<uint> state = createEmptyArray<uint>(image.dims());
    Array<uint> changed = createEmptyArray<uint>(af::dim4(1));
    const size_t outputBytes =
        static_cast<size_t>(out.elements()) * sizeof(T);
    const size_t stateBytes =
        static_cast<size_t>(state.elements()) * sizeof(uint);
    const size_t seedXBytes =
        static_cast<size_t>(seedsX.elements()) * sizeof(uint);
    const size_t seedYBytes =
        static_cast<size_t>(seedsY.elements()) * sizeof(uint);
    const size_t changedBytes = sizeof(uint);

    kernel::launchMetalFloodFillStateInit(state.bufferParam(), stateBytes,
                                          static_cast<size_t>(state.elements()));
    kernel::launchMetalFloodFillSeedState(
        state.bufferParam(), stateBytes, seedsX.bufferParam(), seedXBytes,
        seedsY.bufferParam(), seedYBytes, image.dims(), image.strides(),
        seedsX.dims(), seedsX.strides(), seedsY.strides());

    uint* changedOnHost = changed.getHostPtr();
    do {
        *changedOnHost = 0;
        kernel::launchMetalFloodFillStep(
            state.bufferParam(), stateBytes, image.bufferParam(),
            static_cast<size_t>(image.elements()) * sizeof(T),
            changed.bufferParam(), changedBytes, image.dims(), image.strides(),
            &lowValue, &highValue, sizeof(T),
            static_cast<af_dtype>(af::dtype_traits<T>::af_type));
    } while (*changedOnHost != 0);

    kernel::launchMetalFloodFillFinalize(
        out.bufferParam(), outputBytes, out.dims(), state.bufferParam(),
        stateBytes, state.getData(), &newValue, sizeof(T),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
    return out;
}

#define INSTANTIATE(T)                                                         \
    template Array<T> floodFill(const Array<T>&, const Array<uint>&,           \
                                const Array<uint>&, const T, const T, const T, \
                                const af::connectivity);

INSTANTIATE(float)
INSTANTIATE(uint)
INSTANTIATE(ushort)
INSTANTIATE(uchar)

}  // namespace metal
}  // namespace arrayfire
