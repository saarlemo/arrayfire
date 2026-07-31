/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fft.hpp>

#include <Array.hpp>
#include <copy.hpp>
#include <err_metal.hpp>
#include <metal_fft.hpp>
#include <platform.hpp>
#include <types.hpp>
#include <af/dim4.hpp>

#include <type_traits>

using af::dim4;

namespace arrayfire {
namespace metal {

void setFFTPlanCacheSize(size_t numPlans) { UNUSED(numPlans); }

template<typename T>
void fft_inplace(Array<T> &in, const int rank, const bool direction) {
    if constexpr (!std::is_same<T, cfloat>::value) {
        AF_ERROR("Metal FFT supports only single-precision complex values",
                 AF_ERR_NOT_SUPPORTED);
    } else {
        // MPSGraph does not permit an input and output tensor to alias. Keep
        // the public in-place API while using a temporary device buffer.
        Array<cfloat> input = copyArray<cfloat>(in);
        Array<cfloat> output = createEmptyArray<cfloat>(in.dims());
        input.device();
        output.device();
        fftComplexFloat(output.getBuffer(), input.getBuffer(), in.dims(), rank,
                        !direction);
        copyArray(in, output);
    }
}

template<typename Tc, typename Tr>
Array<Tc> fft_r2c(const Array<Tr> &in, const int rank) {
    if constexpr (!(std::is_same<Tc, cfloat>::value &&
                    std::is_same<Tr, float>::value)) {
        AF_ERROR("Metal real FFT supports only single-precision values",
                 AF_ERR_NOT_SUPPORTED);
    } else {
        dim4 odims = in.dims();
        odims[0] = odims[0] / 2 + 1;
        Array<Tc> out = createEmptyArray<Tc>(odims);
        Array<float> input = copyArray<float>(in);
        input.device();
        out.device();
        fftRealToComplexFloat(out.getBuffer(), input.getBuffer(), in.dims(),
                              rank);
        return out;
    }
}

template<typename Tr, typename Tc>
Array<Tr> fft_c2r(const Array<Tc> &in, const dim4 &odims, const int rank) {
    if constexpr (!(std::is_same<Tr, float>::value &&
                    std::is_same<Tc, cfloat>::value)) {
        AF_ERROR("Metal real FFT supports only single-precision values",
                 AF_ERR_NOT_SUPPORTED);
    } else {
        Array<Tr> out = createEmptyArray<Tr>(odims);
        Array<cfloat> input = copyArray<cfloat>(in);
        input.device();
        out.device();
        fftComplexToRealFloat(out.getBuffer(), input.getBuffer(), odims, rank);
        return out;
    }
}

#define INSTANTIATE(T) \
    template void fft_inplace<T>(Array<T> &, const int, const bool);

INSTANTIATE(cfloat)
INSTANTIATE(cdouble)

#define INSTANTIATE_REAL(Tr, Tc)                                             \
    template Array<Tc> fft_r2c<Tc, Tr>(const Array<Tr> &, const int);          \
    template Array<Tr> fft_c2r<Tr, Tc>(const Array<Tc> &in, const dim4 &odi, \
                                       const int);

INSTANTIATE_REAL(float, cfloat)
INSTANTIATE_REAL(double, cdouble)

}  // namespace metal
}  // namespace arrayfire
