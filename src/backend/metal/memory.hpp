/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <common/AllocatorInterface.hpp>
#include <af/defines.h>

#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace MTL {
class Buffer;
}

namespace arrayfire {
namespace metal {

void *bufferContents(MTL::Buffer *buffer) noexcept;
size_t bufferLength(const MTL::Buffer *buffer) noexcept;
void retainBuffer(MTL::Buffer *buffer) noexcept;
void releaseBuffer(MTL::Buffer *buffer) noexcept;

using buffer_ptr =
    std::unique_ptr<MTL::Buffer, std::function<void(MTL::Buffer *)>>;

template<typename T>
buffer_ptr memAlloc(const size_t &elements);

template<typename T>
T *bufferData(MTL::Buffer *buffer) noexcept {
    return static_cast<T *>(bufferContents(buffer));
}

template<typename T>
const T *bufferData(const MTL::Buffer *buffer) noexcept {
    return static_cast<const T *>(
        bufferContents(const_cast<MTL::Buffer *>(buffer)));
}
void *memAllocUser(const size_t &bytes);

// Need these as 2 separate function and not a default argument
// This is because it is used as the deleter in shared pointer
// which cannot support default arguments
void memFree(MTL::Buffer *ptr);
void memFreeUser(void *ptr);

void memLock(const MTL::Buffer *ptr);
void memUnlock(const MTL::Buffer *ptr);
bool isLocked(const void *ptr);

template<typename T>
T *pinnedAlloc(const size_t &elements);
void pinnedFree(void *ptr);

void deviceMemoryInfo(size_t *alloc_bytes, size_t *alloc_buffers,
                      size_t *lock_bytes, size_t *lock_buffers);
void signalMemoryCleanup();
void shutdownMemoryManager();
void shutdownPinnedMemoryManager();
void pinnedGarbageCollect();

void printMemInfo(const char *msg, const int device);

float getMemoryPressure();
float getMemoryPressureThreshold();
bool jitTreeExceedsMemoryPressure(size_t bytes);
void setMemStepSize(size_t step_bytes);
size_t getMemStepSize(void);

class Allocator final : public common::AllocatorInterface {
   public:
    Allocator();
    ~Allocator() = default;
    void shutdown() override;
    int getActiveDeviceId() override;
    size_t getMaxMemorySize(int id) override;
    void *nativeAlloc(const size_t bytes) override;
    void nativeFree(void *ptr) override;
};

class AllocatorPinned final : public common::AllocatorInterface {
   public:
    AllocatorPinned();
    ~AllocatorPinned() = default;
    void shutdown() override;
    int getActiveDeviceId() override;
    size_t getMaxMemorySize(int id) override;
    void *nativeAlloc(const size_t bytes) override;
    void nativeFree(void *ptr) override;

   private:
    std::vector<std::map<void *, MTL::Buffer *>> pinnedMaps;
};

}  // namespace metal
}  // namespace arrayfire
