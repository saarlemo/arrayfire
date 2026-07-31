/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <common/half.hpp>
#include <kernel/random_engine.hpp>
#include <af/dim4.hpp>

using arrayfire::common::half;

namespace arrayfire {
namespace metal {

void initMersenneState(Array<uint> &state, const uintl seed,
                       const Array<uint> &tbl) {
    getQueue().enqueueNative(kernel::randomMersenneInitMetal, state, tbl, seed);
}

template<typename T>
Array<T> uniformDistribution(const af::dim4 &dims,
                             const af_random_engine_type type,
                             const uintl seed, uintl &counter) {
    const af_dtype dataType = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalRandomUniform(dataType)) {
        AF_ERROR("The random type is not supported by the native Metal kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    Array<T> out = createEmptyArray<T>(dims);
    getQueue().enqueueNative(kernel::randomUniformMetal<T>, out, seed, counter,
                             type);
    counter += out.elements();
    return out;
}

template<typename T>
Array<T> normalDistribution(const af::dim4 &dims,
                            const af_random_engine_type type,
                            const uintl seed, uintl &counter) {
    const af_dtype dataType = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalRandomNormal(dataType)) {
        AF_ERROR("The normal random type is not supported by the native Metal kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    Array<T> out = createEmptyArray<T>(dims);
    getQueue().enqueueNative(kernel::randomNormalMetal<T>, out, seed, counter,
                             type);
    counter += out.elements();
    return out;
}

template<typename T>
Array<T> uniformDistribution(const af::dim4 &dims, Array<uint> pos,
                             Array<uint> sh1, Array<uint> sh2, uint mask,
                             Array<uint> recursion_table,
                             Array<uint> temper_table, Array<uint> state) {
    const af_dtype dataType = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalRandomUniform(dataType)) {
        AF_ERROR("The random type is not supported by the native Metal Mersenne kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    Array<T> out = createEmptyArray<T>(dims);
    getQueue().enqueueNative(kernel::randomMersenneUniformMetal<T>, out, state,
                             pos, sh1, sh2, mask, recursion_table,
                             temper_table);
    return out;
}

template<typename T>
Array<T> normalDistribution(const af::dim4 &dims, Array<uint> pos,
                            Array<uint> sh1, Array<uint> sh2, uint mask,
                            Array<uint> recursion_table,
                            Array<uint> temper_table, Array<uint> state) {
    const af_dtype dataType = static_cast<af_dtype>(af::dtype_traits<T>::af_type);
    if (!kernel::supportsMetalRandomNormal(dataType)) {
        AF_ERROR("The normal random type is not supported by the native Metal Mersenne kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    Array<T> out = createEmptyArray<T>(dims);
    getQueue().enqueueNative(kernel::randomMersenneNormalMetal<T>, out, state,
                             pos, sh1, sh2, mask, recursion_table,
                             temper_table);
    return out;
}

#define INSTANTIATE_UNIFORM(T)                                   \
    template Array<T> uniformDistribution<T>(                    \
        const af::dim4 &dims, const af_random_engine_type type,  \
        const uintl seed, uintl &counter);                       \
    template Array<T> uniformDistribution<T>(                    \
        const af::dim4 &dims, Array<uint> pos, Array<uint> sh1,  \
        Array<uint> sh2, uint mask, Array<uint> recursion_table, \
        Array<uint> temper_table, Array<uint> state);

#define INSTANTIATE_NORMAL(T)                                                  \
    template Array<T> normalDistribution<T>(const af::dim4 &dims,              \
                                            const af_random_engine_type type,  \
                                            const uintl seed, uintl &counter); \
    template Array<T> normalDistribution<T>(                                   \
        const af::dim4 &dims, Array<uint> pos, Array<uint> sh1,                \
        Array<uint> sh2, uint mask, Array<uint> recursion_table,               \
        Array<uint> temper_table, Array<uint> state);

#define COMPLEX_UNIFORM_DISTRIBUTION(T, TR)                                    \
    template<>                                                                 \
    Array<T> uniformDistribution<T>(const af::dim4 &dims,                      \
                                    const af_random_engine_type type,          \
                                    const uintl seed, uintl &counter) {       \
        const af_dtype scalarType =                                           \
            static_cast<af_dtype>(af::dtype_traits<TR>::af_type);              \
        if (!kernel::supportsMetalRandomUniform(scalarType)) {                \
            AF_ERROR("The complex random type is not supported by Metal",     \
                     AF_ERR_NOT_SUPPORTED);                                   \
        }                                                                       \
        Array<T> out = createEmptyArray<T>(dims);                              \
        const size_t elements = out.elements() * 2;                            \
        kernel::launchMetalRandomUniform(                                      \
            out.bufferParam(), elements * sizeof(TR), elements, seed,         \
            counter, type, scalarType);                                        \
        counter += elements;                                                   \
        return out;                                                             \
    }                                                                           \
    template<>                                                                  \
    Array<T> uniformDistribution<T>(                                            \
        const af::dim4 &dims, Array<uint> pos, Array<uint> sh1,                \
        Array<uint> sh2, uint mask, Array<uint> recursion_table,               \
        Array<uint> temper_table, Array<uint> state) {                         \
        const af_dtype scalarType =                                           \
            static_cast<af_dtype>(af::dtype_traits<TR>::af_type);              \
        if (!kernel::supportsMetalRandomUniform(scalarType)) {                \
            AF_ERROR("The complex random type is not supported by Metal",     \
                     AF_ERR_NOT_SUPPORTED);                                   \
        }                                                                       \
        Array<T> out = createEmptyArray<T>(dims);                              \
        const size_t elements = out.elements() * 2;                            \
        kernel::launchMetalRandomMersenne(                                     \
            out.bufferParam(), elements * sizeof(TR), elements,               \
            state.bufferParam(), pos.bufferParam(), sh1.bufferParam(),        \
            sh2.bufferParam(), mask, recursion_table.bufferParam(),            \
            temper_table.bufferParam(), false, scalarType);                    \
        return out;                                                             \
    }

#define COMPLEX_NORMAL_DISTRIBUTION(T, TR)                                     \
    template<>                                                                  \
    Array<T> normalDistribution<T>(const af::dim4 &dims,                       \
                                   const af_random_engine_type type,           \
                                   const uintl seed, uintl &counter) {         \
        const af_dtype scalarType =                                           \
            static_cast<af_dtype>(af::dtype_traits<TR>::af_type);              \
        if (!kernel::supportsMetalRandomNormal(scalarType)) {                 \
            AF_ERROR("The complex normal type is not supported by Metal",     \
                     AF_ERR_NOT_SUPPORTED);                                   \
        }                                                                       \
        Array<T> out = createEmptyArray<T>(dims);                              \
        const size_t elements = out.elements() * 2;                            \
        kernel::launchMetalRandomNormal(                                       \
            out.bufferParam(), elements * sizeof(TR), elements, seed,         \
            counter, type, scalarType);                                        \
        counter += elements;                                                   \
        return out;                                                             \
    }                                                                           \
    template<>                                                                  \
    Array<T> normalDistribution<T>(                                            \
        const af::dim4 &dims, Array<uint> pos, Array<uint> sh1,                \
        Array<uint> sh2, uint mask, Array<uint> recursion_table,               \
        Array<uint> temper_table, Array<uint> state) {                         \
        const af_dtype scalarType =                                           \
            static_cast<af_dtype>(af::dtype_traits<TR>::af_type);              \
        if (!kernel::supportsMetalRandomNormal(scalarType)) {                 \
            AF_ERROR("The complex normal type is not supported by Metal",     \
                     AF_ERR_NOT_SUPPORTED);                                   \
        }                                                                       \
        Array<T> out = createEmptyArray<T>(dims);                              \
        const size_t elements = out.elements() * 2;                            \
        kernel::launchMetalRandomMersenne(                                     \
            out.bufferParam(), elements * sizeof(TR), elements,               \
            state.bufferParam(), pos.bufferParam(), sh1.bufferParam(),        \
            sh2.bufferParam(), mask, recursion_table.bufferParam(),            \
            temper_table.bufferParam(), true, scalarType);                     \
        return out;                                                             \
    }

INSTANTIATE_UNIFORM(float)
INSTANTIATE_UNIFORM(double)
INSTANTIATE_UNIFORM(int)
INSTANTIATE_UNIFORM(uint)
INSTANTIATE_UNIFORM(intl)
INSTANTIATE_UNIFORM(uintl)
INSTANTIATE_UNIFORM(char)
INSTANTIATE_UNIFORM(schar)
INSTANTIATE_UNIFORM(uchar)
INSTANTIATE_UNIFORM(short)
INSTANTIATE_UNIFORM(ushort)
INSTANTIATE_UNIFORM(half)

INSTANTIATE_NORMAL(float)
INSTANTIATE_NORMAL(double)
INSTANTIATE_NORMAL(half)

COMPLEX_UNIFORM_DISTRIBUTION(cdouble, double)  // NOLINT
COMPLEX_UNIFORM_DISTRIBUTION(cfloat, float)    // NOLINT

COMPLEX_NORMAL_DISTRIBUTION(cdouble, double)  // NOLINT
COMPLEX_NORMAL_DISTRIBUTION(cfloat, float)    // NOLINT

}  // namespace metal
}  // namespace arrayfire
