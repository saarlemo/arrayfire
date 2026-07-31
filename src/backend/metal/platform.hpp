/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <queue.hpp>
#include <string>

namespace MTL {
class CommandQueue;
class CommandBuffer;
class Device;
}  // namespace MTL

namespace arrayfire {
namespace common {
class ForgeManager;
class MemoryManagerBase;
}  // namespace common
}  // namespace arrayfire

using arrayfire::common::MemoryManagerBase;

namespace arrayfire {
namespace metal {

int getBackend();

std::string getDeviceInfo() noexcept;

bool isDoubleSupported(int device);

bool isHalfSupported(int device);

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute);

int& getMaxJitSize();

int getDeviceCount();

void init();

int getActiveDeviceId();

size_t getDeviceMemorySize(int device);

size_t getHostMemorySize();

int setDevice(int device);

queue& getQueue(int device = -1);

MTL::Device& getDevice(int device = -1);

MTL::CommandQueue& getCommandQueue(int device = -1);

/// Commit a native command buffer and retain it as the synchronization point for
/// its Metal command queue.
void submitCommandBuffer(MTL::CommandBuffer* commandBuffer, int device = -1);

/// Wait for all native work submitted to the Metal command queue.
void syncCommandQueue(int device = -1);

/// Return a handle to the queue for the device.
///
/// \param[in] device The device of the returned queue
/// \returns The handle to the queue
queue* getQueueHandle(int device);

void sync(int device);

bool& evalFlag();

MemoryManagerBase& memoryManager();

void setMemoryManager(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManager();

MemoryManagerBase& pinnedMemoryManager();

void setMemoryManagerPinned(std::unique_ptr<MemoryManagerBase> mgr);

void resetMemoryManagerPinned();

arrayfire::common::ForgeManager& forgeManager();

}  // namespace metal
}  // namespace arrayfire
