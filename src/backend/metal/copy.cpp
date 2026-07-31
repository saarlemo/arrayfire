/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Metal.hpp>

#include <Array.hpp>
#include <common/ArrayInfo.hpp>
#include <common/cast.hpp>
#include <common/complex.hpp>
#include <common/half.hpp>
#include <copy.hpp>
#include <err_metal.hpp>
#include <kernel/copy.hpp>
#include <platform.hpp>
#include <queue.hpp>

#include <af/defines.h>
#include <af/dim4.hpp>

#include <cstdio>
#include <cstring>
#include <type_traits>

using arrayfire::common::half;  // NOLINT(misc-unused-using-decls) bug in
                                // clang-tidy
using arrayfire::common::is_complex;

namespace arrayfire {
namespace metal {
namespace {

template<typename T>
constexpr bool supportsNativeMetalCopy =
    !std::is_same_v<T, double> && !std::is_same_v<T, cdouble>;

}  // namespace

bool copyBuffer(MTL::Buffer* destination, const size_t destinationOffset,
                MTL::Buffer* source, const size_t sourceOffset,
                const size_t bytes) {
    if (bytes == 0) { return true; }
    if (!destination || !source) { return false; }

    auto commandBuffer = NS::RetainPtr(getCommandQueue().commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->blitCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) { return false; }

    encoder->copyFromBuffer(source, sourceOffset, destination,
                            destinationOffset, bytes);
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    return true;
}

template<typename T>
void copyData(T *to, const Array<T> &from) {
    if (from.elements() == 0) { return; }

    from.eval();
    // Ensure all operations on 'from' are complete before copying data to host.
    getQueue().sync();
    if (from.isLinear()) {
        // FIXME: Check for errors / exceptions
        memcpy(to, from.getHostPtr(), from.elements() * sizeof(T));
    } else {
        dim4 ostrides = calcStrides(from.dims());
        kernel::stridedCopy<T>(to, ostrides, from.getHostPtr(), from.dims(),
                               from.strides(), from.ndims() - 1);
    }
}

template<typename T>
Array<T> copyArray(const Array<T> &A) {
    Array<T> out = createEmptyArray<T>(A.dims());
    if (A.elements() > 0) {
        const af_dtype type =
            static_cast<af_dtype>(af::dtype_traits<T>::af_type);
        if (!kernel::supportsMetalCopy(type)) {
            AF_ERROR("Metal copy does not support this type",
                     AF_ERR_NOT_SUPPORTED);
        }
        getQueue().enqueueNative(kernel::copyMetal<T>, out, A);
    }
    return out;
}

template<typename inType, typename outType>
void copyArray(Array<outType> &out, Array<inType> const &in) {
    static_assert(
        !(is_complex<inType>::value && !is_complex<outType>::value),
        "Cannot copy from complex Array<T> to a non complex Array<T>");
    if constexpr (supportsNativeMetalCopy<inType> &&
                  supportsNativeMetalCopy<outType>) {
        if constexpr (std::is_same_v<inType, outType>) {
            getQueue().enqueueNative(kernel::copyMetal<outType>, out, in);
        } else {
            const Array<outType> converted = common::cast<outType>(in);
            getQueue().enqueueNative(kernel::copyMetal<outType>, out,
                                     converted);
        }
    } else {
        AF_ERROR("Metal typed copy does not support double precision",
                 AF_ERR_NOT_SUPPORTED);
    }
}

#define INSTANTIATE(T)                                         \
    template void copyData<T>(T * data, const Array<T> &from); \
    template Array<T> copyArray<T>(const Array<T> &A);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

#define INSTANTIATE_COPY_ARRAY(SRC_T)                                 \
    template void copyArray<SRC_T, float>(Array<float> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, double>(Array<double> & dst,       \
                                           Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, cfloat>(Array<cfloat> & dst,       \
                                           Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> & dst,     \
                                            Array<SRC_T> const &src); \
    template void copyArray<SRC_T, int>(Array<int> & dst,             \
                                        Array<SRC_T> const &src);     \
    template void copyArray<SRC_T, uint>(Array<uint> & dst,           \
                                         Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, intl>(Array<intl> & dst,           \
                                         Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, uintl>(Array<uintl> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, short>(Array<short> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, ushort>(Array<ushort> & dst,       \
                                           Array<SRC_T> const &src);  \
    template void copyArray<SRC_T, schar>(Array<schar> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, uchar>(Array<uchar> & dst,         \
                                          Array<SRC_T> const &src);   \
    template void copyArray<SRC_T, char>(Array<char> & dst,           \
                                         Array<SRC_T> const &src);    \
    template void copyArray<SRC_T, half>(Array<half> & dst,           \
                                         Array<SRC_T> const &src);

INSTANTIATE_COPY_ARRAY(float)
INSTANTIATE_COPY_ARRAY(double)
INSTANTIATE_COPY_ARRAY(int)
INSTANTIATE_COPY_ARRAY(uint)
INSTANTIATE_COPY_ARRAY(intl)
INSTANTIATE_COPY_ARRAY(uintl)
INSTANTIATE_COPY_ARRAY(schar)
INSTANTIATE_COPY_ARRAY(uchar)
INSTANTIATE_COPY_ARRAY(char)
INSTANTIATE_COPY_ARRAY(ushort)
INSTANTIATE_COPY_ARRAY(short)
INSTANTIATE_COPY_ARRAY(half)

#define INSTANTIATE_COPY_ARRAY_COMPLEX(SRC_T)                        \
    template void copyArray<SRC_T, cfloat>(Array<cfloat> & dst,      \
                                           Array<SRC_T> const &src); \
    template void copyArray<SRC_T, cdouble>(Array<cdouble> & dst,    \
                                            Array<SRC_T> const &src);

INSTANTIATE_COPY_ARRAY_COMPLEX(cfloat)
INSTANTIATE_COPY_ARRAY_COMPLEX(cdouble)

template<typename T>
T getScalar(const Array<T> &in) {
    in.eval();
    getQueue().sync();
    return in.getHostPtr()[0];
}

#define INSTANTIATE_GETSCALAR(T) template T getScalar(const Array<T> &in);

INSTANTIATE_GETSCALAR(float)
INSTANTIATE_GETSCALAR(double)
INSTANTIATE_GETSCALAR(cfloat)
INSTANTIATE_GETSCALAR(cdouble)
INSTANTIATE_GETSCALAR(int)
INSTANTIATE_GETSCALAR(uint)
INSTANTIATE_GETSCALAR(schar)
INSTANTIATE_GETSCALAR(uchar)
INSTANTIATE_GETSCALAR(char)
INSTANTIATE_GETSCALAR(intl)
INSTANTIATE_GETSCALAR(uintl)
INSTANTIATE_GETSCALAR(short)
INSTANTIATE_GETSCALAR(ushort)
INSTANTIATE_GETSCALAR(half)
}  // namespace metal
}  // namespace arrayfire
