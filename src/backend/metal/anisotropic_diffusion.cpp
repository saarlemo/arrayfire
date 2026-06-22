/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <anisotropic_diffusion.hpp>
#include <kernel/anisotropic_diffusion.hpp>
#include <metal_compute_anisotropic_diffusion.hpp>
#include <platform.hpp>

#include <type_traits>

namespace arrayfire {
namespace metal {
template<typename T>
void anisotropicDiffusion(Array<T>& inout, const float dt, const float mct,
                          const af::fluxFunction fftype,
                          const af::diffusionEq eq) {
    if constexpr (std::is_same<T, float>::value) {
        getQueue().enqueue(kernel::anisotropicDiffusionMetal, inout, dt, mct,
                           fftype, eq == AF_DIFFUSION_MCDE);
    } else if (eq == AF_DIFFUSION_MCDE) {
        getQueue().enqueue(kernel::anisotropicDiffusion<T, true>, inout, dt,
                           mct, fftype);
    } else {
        getQueue().enqueue(kernel::anisotropicDiffusion<T, false>, inout, dt,
                           mct, fftype);
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
