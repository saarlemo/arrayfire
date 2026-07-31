/*******************************************************
 * Copyright (c) 2019, ArrayFire
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

#include <common/DefaultMemoryManager.hpp>
#include <common/err_common.hpp>
#include <common/graphics_common.hpp>
#include <device_manager.hpp>
#include <memory.hpp>
#include <af/version.h>

#include <algorithm>

using arrayfire::common::MemoryManagerBase;
namespace arrayfire {
namespace metal {

DeviceManager::DeviceManager()
    : queues()
    , nativeDevices()
    , nativeCommandQueues()
    , lastCommandBuffers()
    , commandQueueFailed()
    , fgMngr(new common::ForgeManager())
    , memManager() {
    NS::Array* devices = MTL::CopyAllDevices();
    if (devices) {
        nativeDevices.reserve(devices->count());
        nativeCommandQueues.reserve(devices->count());
        for (NS::UInteger i = 0; i < devices->count(); ++i) {
            MTL::Device* device = devices->object<MTL::Device>(i);
            if (!device) { continue; }
            device->retain();
            nativeDevices.push_back(device);
            nativeCommandQueues.push_back(device->newCommandQueue());
            lastCommandBuffers.push_back(nullptr);
            commandQueueFailed.push_back(false);
        }
        devices->release();
    }

    if (nativeDevices.empty()) {
        MTL::Device* device = MTL::CreateSystemDefaultDevice();
        if (device) {
            nativeDevices.push_back(device);
            nativeCommandQueues.push_back(device->newCommandQueue());
            lastCommandBuffers.push_back(nullptr);
            commandQueueFailed.push_back(false);
        }
    }

    queues.reserve(nativeDevices.size());
    for (size_t device = 0; device < nativeDevices.size(); ++device) {
        queues.emplace_back(new queue(static_cast<int>(device)));
    }
}

DeviceManager::~DeviceManager() {
    for (const auto& hostQueue : queues) { hostQueue->sync(); }
    for (MTL::CommandBuffer* commandBuffer : lastCommandBuffers) {
        if (commandBuffer) {
            commandBuffer->waitUntilCompleted();
            commandBuffer->release();
        }
    }
    for (MTL::CommandQueue* commandQueue : nativeCommandQueues) {
        if (commandQueue) { commandQueue->release(); }
    }
    for (MTL::Device* device : nativeDevices) {
        if (device) { device->release(); }
    }
}

DeviceManager& DeviceManager::getInstance() {
    static auto* my_instance = new DeviceManager();
    return *my_instance;
}

int DeviceManager::deviceCount() const {
    return static_cast<int>(nativeDevices.size());
}

void DeviceManager::resetMemoryManager() {
    // Replace with default memory manager
    std::unique_ptr<MemoryManagerBase> mgr(new common::DefaultMemoryManager(
        std::max(1, deviceCount()), common::MAX_BUFFERS,
        AF_MEM_DEBUG || AF_METAL_MEM_DEBUG));
    setMemoryManager(std::move(mgr));
}

void DeviceManager::setMemoryManager(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    std::lock_guard<std::mutex> l(mutex);
    // It's possible we're setting a memory manager and the default memory
    // manager still hasn't been initialized, so initialize it anyways so we
    // don't inadvertently reset to it when we first call memoryManager()
    memoryManager();
    // Calls shutdown() on the existing memory manager
    if (memManager) { memManager->shutdownAllocator(); }
    memManager = std::move(newMgr);
    // Set the backend memory manager for this new manager to register native
    // functions correctly.
    std::unique_ptr<metal::Allocator> deviceMemoryManager(
        new metal::Allocator());
    memManager->setAllocator(std::move(deviceMemoryManager));
    memManager->initialize();
}

void DeviceManager::setMemoryManagerPinned(
    std::unique_ptr<MemoryManagerBase> newMgr) {
    std::lock_guard<std::mutex> l(mutex);
    pinnedMemoryManager();
    if (pinnedMemManager) { pinnedMemManager->shutdownAllocator(); }
    pinnedMemManager = std::move(newMgr);
    std::unique_ptr<metal::AllocatorPinned> pinnedAllocator(
        new metal::AllocatorPinned());
    pinnedMemManager->setAllocator(std::move(pinnedAllocator));
    pinnedMemManager->initialize();
}

void DeviceManager::resetMemoryManagerPinned() {
    std::unique_ptr<MemoryManagerBase> mgr(new common::DefaultMemoryManager(
        std::max(1, deviceCount()), common::MAX_BUFFERS,
        AF_MEM_DEBUG || AF_METAL_MEM_DEBUG));
    setMemoryManagerPinned(std::move(mgr));
}

}  // namespace metal
}  // namespace arrayfire
