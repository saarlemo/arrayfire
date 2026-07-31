/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <memory.hpp>

#include <Metal.hpp>
#include <common/DefaultMemoryManager.hpp>
#include <common/Logger.hpp>
#include <common/half.hpp>
#include <err_metal.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <spdlog/spdlog.h>
#include <types.hpp>
#include <af/dim4.hpp>

#include <utility>

using af::dim4;
using arrayfire::common::bytesToString;
using arrayfire::common::half;
using std::function;
using std::move;
using std::unique_ptr;

namespace arrayfire {
namespace metal {
void *bufferContents(MTL::Buffer *buffer) noexcept {
    return buffer ? buffer->contents() : nullptr;
}

size_t bufferLength(const MTL::Buffer *buffer) noexcept {
    return buffer ? const_cast<MTL::Buffer *>(buffer)->length() : 0;
}

void retainBuffer(MTL::Buffer *buffer) noexcept {
    if (buffer) { buffer->retain(); }
}

void releaseBuffer(MTL::Buffer *buffer) noexcept {
    if (buffer) { buffer->release(); }
}

float getMemoryPressure() { return memoryManager().getMemoryPressure(); }
float getMemoryPressureThreshold() {
    return memoryManager().getMemoryPressureThreshold();
}

bool jitTreeExceedsMemoryPressure(size_t bytes) {
    return memoryManager().jitTreeExceedsMemoryPressure(bytes);
}

void setMemStepSize(size_t step_bytes) {
    memoryManager().setMemStepSize(step_bytes);
}

size_t getMemStepSize() { return memoryManager().getMemStepSize(); }

void signalMemoryCleanup() { memoryManager().signalMemoryCleanup(); }

void shutdownMemoryManager() { memoryManager().shutdown(); }

void shutdownPinnedMemoryManager() { pinnedMemoryManager().shutdown(); }

void printMemInfo(const char *msg, const int device) {
    memoryManager().printInfo(msg, device);
}

template<typename T>
unique_ptr<MTL::Buffer, function<void(MTL::Buffer *)>> memAlloc(
    const size_t &elements) {
    // TODO: make memAlloc aware of array shapes
    if (!elements) {
        return unique_ptr<MTL::Buffer, function<void(MTL::Buffer *)>>(nullptr,
                                                                      memFree);
    }
    dim4 dims(elements);
    auto *ptr = static_cast<MTL::Buffer *>(
        memoryManager().alloc(false, 1, dims.get(), sizeof(T)));
    return unique_ptr<MTL::Buffer, function<void(MTL::Buffer *)>>(ptr, memFree);
}

void *memAllocUser(const size_t &bytes) {
    dim4 dims(bytes);
    return memoryManager().alloc(true, 1, dims.get(), 1);
}

void memFree(MTL::Buffer *ptr) {
    return memoryManager().unlock(static_cast<void *>(ptr), false);
}

void memFreeUser(void *ptr) { memoryManager().unlock(ptr, true); }

void memLock(const MTL::Buffer *ptr) { memoryManager().userLock(ptr); }

bool isLocked(const void *ptr) { return memoryManager().isUserLocked(ptr); }

void memUnlock(const MTL::Buffer *ptr) { memoryManager().userUnlock(ptr); }

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers) {
    memoryManager().usageInfo(alloc_bytes, alloc_buffers, lock_bytes,
                              lock_buffers);
}

template<typename T>
T *pinnedAlloc(const size_t &elements) {
    dim4 dims(elements);
    return static_cast<T *>(
        pinnedMemoryManager().alloc(false, 1, dims.get(), sizeof(T)));
}

void pinnedFree(void *ptr) { pinnedMemoryManager().unlock(ptr, false); }

#define INSTANTIATE(T)                                             \
    template std::unique_ptr<MTL::Buffer,                          \
                             std::function<void(MTL::Buffer *)>>   \
    memAlloc<T>(const size_t &elements);                           \
    template T *pinnedAlloc(const size_t &elements);

INSTANTIATE(float)
INSTANTIATE(cfloat)
INSTANTIATE(double)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(char)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(ushort)
INSTANTIATE(short)
INSTANTIATE(half)

template<>
void *pinnedAlloc<void>(const size_t &elements) {
    dim4 dims(elements);
    return pinnedMemoryManager().alloc(false, 1, dims.get(), 1);
}

Allocator::Allocator() { logger = common::loggerFactory("mem"); }

void Allocator::shutdown() {
    const int previous = metal::getActiveDeviceId();
    for (int n = 0; n < metal::getDeviceCount(); n++) {
        try {
            metal::setDevice(n);
            shutdownMemoryManager();
        } catch (const AfError &err) {
            continue;  // Do not throw any errors while shutting down
        }
    }
    metal::setDevice(previous);
}

int Allocator::getActiveDeviceId() {
    return static_cast<int>(metal::getActiveDeviceId());
}

size_t Allocator::getMaxMemorySize(int id) {
    return metal::getDeviceMemorySize(id);
}

void *Allocator::nativeAlloc(const size_t bytes) {
    MTL::Device *device = &getDevice();
    MTL::Buffer *ptr =
        device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    AF_TRACE("nativeAlloc: {:>7} {}", bytesToString(bytes),
             static_cast<void *>(ptr));
    if (!ptr) { AF_ERROR("Unable to allocate memory", AF_ERR_NO_MEM); }
    return ptr;
}

void Allocator::nativeFree(void *ptr) {
    AF_TRACE("nativeFree: {: >8} {}", " ", ptr);
    // Make sure this pointer is not being used on the queue before freeing the
    // memory.
    getQueue().sync();
    static_cast<MTL::Buffer *>(ptr)->release();
}

AllocatorPinned::AllocatorPinned()
    : pinnedMaps(std::max(1, metal::getDeviceCount())) {
    logger = common::loggerFactory("mem");
}

void AllocatorPinned::shutdown() {
    const int previous = metal::getActiveDeviceId();
    for (int n = 0; n < metal::getDeviceCount(); ++n) {
        try {
            metal::setDevice(n);
            shutdownPinnedMemoryManager();
        } catch (const AfError &) { continue; }
    }
    metal::setDevice(previous);
}

int AllocatorPinned::getActiveDeviceId() {
    return static_cast<int>(metal::getActiveDeviceId());
}

size_t AllocatorPinned::getMaxMemorySize(int id) {
    return metal::getDeviceMemorySize(id);
}

void *AllocatorPinned::nativeAlloc(const size_t bytes) {
    MTL::Buffer *buffer =
        getDevice().newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!buffer || !buffer->contents()) {
        if (buffer) { buffer->release(); }
        AF_ERROR("Unable to allocate pinned Metal memory", AF_ERR_NO_MEM);
    }

    void *ptr = buffer->contents();
    AF_TRACE("Pinned::nativeAlloc: {:>7} {}", bytesToString(bytes), ptr);
    pinnedMaps[metal::getActiveDeviceId()].emplace(ptr, buffer);
    return ptr;
}

void AllocatorPinned::nativeFree(void *ptr) {
    AF_TRACE("Pinned::nativeFree:          {}", ptr);
    auto &buffers = pinnedMaps[metal::getActiveDeviceId()];
    auto iter     = buffers.find(ptr);
    if (iter == buffers.end()) { return; }

    iter->second->release();
    buffers.erase(iter);
}
}  // namespace metal
}  // namespace arrayfire
