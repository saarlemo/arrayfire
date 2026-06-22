/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cstddef>
#include <string>

namespace arrayfire {
namespace metal {

struct MetalDeviceInfo {
    std::string name;
    size_t recommendedMemoryBytes    = 0;
    size_t maxThreadgroupMemoryBytes = 0;
    bool unifiedMemory               = false;
    bool lowPower                    = false;
    bool headless                    = false;
    bool removable                   = false;
};

int getMetalDeviceCount() noexcept;
MetalDeviceInfo getMetalDeviceInfo(int device) noexcept;

}  // namespace metal
}  // namespace arrayfire
