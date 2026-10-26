/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <anisotropic_diffusion.hpp>
#include <Array.hpp>
#include <err_metal.hpp>
#include <kernel/anisotropic_diffusion.hpp>
#include <platform.hpp>

#include <type_traits>
#include <utility>

namespace arrayfire {
namespace metal {
template<typename T>
void anisotropicDiffusion(Array<T>& inout, const float dt, const float mct,
                          const af::fluxFunction fftype,
                          const af::diffusionEq eq) {
    if constexpr (std::is_same<T, float>::value) {
        inout.eval();
        Array<T> output = createEmptyArray<T>(inout.dims());
        getQueue().enqueueNative(kernel::anisotropicDiffusionMetal, output,
                                 inout, inout.getData(), dt, mct, fftype,
                                 eq == AF_DIFFUSION_MCDE);
        inout = std::move(output);
    } else {
        AF_ERROR("Anisotropic diffusion type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    }
}

#define INSTANTIATE(T)                                     \
    template void anisotropicDiffusion<T>(                 \
        Array<T> & inout, const float dt, const float mct, \
        const af::fluxFunction fftype, const af::diffusionEq eq);

INSTANTIATE(double)
INSTANTIATE(float)
}  // namespace metal
}  // namespace arrayfire
