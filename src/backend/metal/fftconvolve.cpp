/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <fftconvolve.hpp>

#include <Array.hpp>
#include <copy.hpp>
#include <common/dispatch.hpp>
#include <err_metal.hpp>
#include <kernel/fftconvolve.hpp>
#include <metal_fft.hpp>
#include <queue.hpp>
#include <af/dim4.hpp>

#include <array>
#include <cmath>
#include <type_traits>

using af::dim4;
using std::array;
using std::ceil;

namespace arrayfire {
namespace metal {

template<typename T>
Array<T> fftconvolve(Array<T> const& signal, Array<T> const& filter,
                     const bool expand, AF_BATCH_KIND kind, const int rank) {
    // Metal's native FFT path currently uses MPSGraph's ComplexFloat32
    // transform. Integral inputs are converted to float for the convolution;
    // double precision is deliberately rejected instead of crossing to the
    // CPU FFTW fallback.
    if constexpr (!std::is_same<T, float>::value &&
                  !std::is_integral<T>::value) {
        AF_ERROR("Metal FFT convolution supports only single precision",
                 AF_ERR_NOT_SUPPORTED);
    } else {
        const dim4& sd = signal.dims();
        const dim4& fd = filter.dims();
        dim_t fftScale = 1;

        dim4 packedDims(1, 1, 1, 1);
        array<int, AF_MAX_DIMS> fftDims{};

        fftDims[rank - 1] = nextpow2(
            static_cast<unsigned>(static_cast<int>(ceil(sd[0] / 2.f)) +
                                  fd[0] - 1));
        packedDims[0] = 2 * fftDims[rank - 1];
        fftScale *= fftDims[rank - 1];

        for (int k = 1; k < rank; k++) {
            packedDims[k] = nextpow2(
                static_cast<unsigned>(sd[k] + fd[k] - 1));
            fftDims[rank - k - 1] = packedDims[k];
            fftScale *= fftDims[rank - k - 1];
        }

        dim_t sbatch = 1, fbatch = 1;
        for (int k = rank; k < AF_MAX_DIMS; k++) {
            sbatch *= sd[k];
            fbatch *= fd[k];
        }
        packedDims[rank] = (sbatch + fbatch);

        Array<float> packed = createEmptyArray<float>(packedDims);

        dim4 paddedSigDims(packedDims[0], (1 < rank ? packedDims[1] : sd[1]),
                           (2 < rank ? packedDims[2] : sd[2]),
                           (3 < rank ? packedDims[3] : sd[3]));
        dim4 paddedFilDims(packedDims[0], (1 < rank ? packedDims[1] : fd[1]),
                           (2 < rank ? packedDims[2] : fd[2]),
                           (3 < rank ? packedDims[3] : fd[3]));
        dim4 paddedSigStrides = calcStrides(paddedSigDims);
        dim4 paddedFilStrides = calcStrides(paddedFilDims);

        // Number of packed complex elements in dimension 0.
        dim_t sig_half_d0 = divup(sd[0], 2);

        getQueue().enqueueNative(kernel::packData<T>, packed, paddedSigDims,
                                 paddedSigStrides, signal);
        const dim_t offset = paddedSigStrides[3] * paddedSigDims[3];
        getQueue().enqueueNative(kernel::padArray<T>, packed, paddedFilDims,
                                 paddedFilStrides, filter, offset);

        // MPSGraph consumes a logical complex tensor. The physical packed
        // buffer has two float values per complex element in dimension 0.
        dim4 logicalFftDims = packedDims;
        logicalFftDims[0] /= 2;
        Array<float> transformed = createEmptyArray<float>(packedDims);
        packed.device();
        transformed.device();
        fftComplexFloat(transformed.getBuffer(), packed.getBuffer(),
                        logicalFftDims, rank, false);
        copyArray(packed, transformed);

        getQueue().enqueueNative(kernel::fftConvolveMultiplyMetal, packed,
                                 paddedSigDims, paddedSigStrides, paddedFilDims,
                                 paddedFilStrides, kind, offset);

        transformed = createEmptyArray<float>(packedDims);
        transformed.device();
        fftComplexFloat(transformed.getBuffer(), packed.getBuffer(),
                        logicalFftDims, rank, true);
        copyArray(packed, transformed);

        dim4 oDims(1);
        if (expand) {
            for (int d = 0; d < AF_MAX_DIMS; ++d) {
                if (kind == AF_BATCH_NONE || kind == AF_BATCH_RHS) {
                    oDims[d] = sd[d] + fd[d] - 1;
                } else {
                    oDims[d] = (d < rank ? sd[d] + fd[d] - 1 : sd[d]);
                }
            }
        } else {
            oDims = sd;
            if (kind == AF_BATCH_RHS) {
                for (int i = rank; i < AF_MAX_DIMS; ++i) { oDims[i] = fd[i]; }
            }
        }

        Array<T> out = createEmptyArray<T>(oDims);
        getQueue().enqueueNative(kernel::reorder<T>, out, packed, filter,
                                 sig_half_d0, fftScale, paddedSigDims,
                                 paddedSigStrides, paddedFilDims,
                                 paddedFilStrides, kind, rank, expand);
        return out;
    }
}

#define INSTANTIATE(T)                                                 \
    template Array<T> fftconvolve<T>(Array<T> const&, Array<T> const&,    \
                                     const bool, AF_BATCH_KIND, const int);

INSTANTIATE(double)
INSTANTIATE(float)
INSTANTIATE(uint)
INSTANTIATE(int)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(uintl)
INSTANTIATE(intl)
INSTANTIATE(ushort)
INSTANTIATE(short)

}  // namespace metal
}  // namespace arrayfire
