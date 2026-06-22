/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal.hpp>

#include <metal_device.hpp>

#include <algorithm>

namespace arrayfire {
namespace metal {
namespace {

std::string toString(NS::String* value) {
    const char* utf8 = value ? value->utf8String() : nullptr;
    return utf8 ? utf8 : "Unknown Metal Device";
}

MTL::Device* defaultDevice() noexcept {
    return MTL::CreateSystemDefaultDevice();
}

}  // namespace

int getMetalDeviceCount() noexcept {
    NS::Array* devices = MTL::CopyAllDevices();
    if (devices) {
        const int count = static_cast<int>(devices->count());
        devices->release();
        return count;
    }

    MTL::Device* device = defaultDevice();
    if (!device) { return 0; }
    device->release();
    return 1;
}

MetalDeviceInfo getMetalDeviceInfo(int device) noexcept {
    MetalDeviceInfo info;
    NS::Array* devices = MTL::CopyAllDevices();

    MTL::Device* selected = nullptr;
    if (devices && device >= 0 && device < static_cast<int>(devices->count())) {
        selected =
            devices->object<MTL::Device>(static_cast<NS::UInteger>(device));
    }

    MTL::Device* fallback = nullptr;
    if (!selected) {
        fallback = defaultDevice();
        selected = fallback;
    }

    if (selected) {
        info.name = toString(selected->name());
        info.recommendedMemoryBytes =
            static_cast<size_t>(selected->recommendedMaxWorkingSetSize());
        info.maxThreadgroupMemoryBytes =
            static_cast<size_t>(selected->maxThreadgroupMemoryLength());
        info.unifiedMemory = selected->hasUnifiedMemory();
        info.lowPower      = selected->isLowPower();
        info.headless      = selected->isHeadless();
        info.removable     = selected->isRemovable();
    }

    if (fallback) { fallback->release(); }
    if (devices) { devices->release(); }
    return info;
}

}  // namespace metal
}  // namespace arrayfire
