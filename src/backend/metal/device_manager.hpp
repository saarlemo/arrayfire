/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <platform.hpp>
#include <queue.hpp>
#include <memory>
#include <mutex>
#include <vector>

using arrayfire::common::MemoryManagerBase;

namespace MTL {
class CommandBuffer;
class CommandQueue;
class Device;
}  // namespace MTL

#ifndef AF_METAL_MEM_DEBUG
#define AF_METAL_MEM_DEBUG 0
#endif

namespace arrayfire {
namespace metal {

class DeviceManager {
   public:
    static const int MAX_DEVICES          = 32;
    static const bool IS_DOUBLE_SUPPORTED  = false;

    // TODO(umar): Half is not supported for BLAS and FFT on x86_64
    static const bool IS_HALF_SUPPORTED = true;

    static DeviceManager& getInstance();

    friend queue& getQueue(int device);
    friend MTL::Device& getDevice(int device);
    friend MTL::CommandQueue& getCommandQueue(int device);
    friend void submitCommandBuffer(MTL::CommandBuffer* commandBuffer,
                                    int device);
    friend void syncCommandQueue(int device);

    friend MemoryManagerBase& memoryManager();
    friend MemoryManagerBase& pinnedMemoryManager();

    friend void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

    friend void resetMemoryManager();

    friend void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

    void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

    friend void resetMemoryManagerPinned();

    void resetMemoryManagerPinned();

    friend arrayfire::common::ForgeManager& forgeManager();

    void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

    void resetMemoryManager();

    int deviceCount() const;

   private:
    DeviceManager();
    ~DeviceManager();
    // Following two declarations are required to
    // avoid copying accidental copy/assignment
    // of instance returned by getInstance to other
    // variables
    DeviceManager(DeviceManager const&)  = delete;
    void operator=(DeviceManager const&) = delete;

    // Attributes
    std::vector<std::unique_ptr<queue>> queues;
    std::vector<MTL::Device*> nativeDevices;
    std::vector<MTL::CommandQueue*> nativeCommandQueues;
    std::vector<MTL::CommandBuffer*> lastCommandBuffers;
    std::vector<bool> commandQueueFailed;
    std::unique_ptr<arrayfire::common::ForgeManager> fgMngr;
    std::unique_ptr<MemoryManagerBase> memManager;
    std::unique_ptr<MemoryManagerBase> pinnedMemManager;
    std::mutex mutex;
    std::mutex commandMutex;
};

}  // namespace metal
}  // namespace arrayfire
