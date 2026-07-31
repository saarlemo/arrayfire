/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Metal.hpp>

#include <Array.hpp>
#include <build_version.hpp>
#include <common/DefaultMemoryManager.hpp>
#include <common/MemoryManagerBase.hpp>
#include <common/defines.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <common/host_memory.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <handle.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <af/metal.h>
#include <af/version.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

using arrayfire::common::ForgeManager;
using arrayfire::common::getEnvVar;
using arrayfire::common::ltrim;
using arrayfire::common::MemoryManagerBase;
using std::endl;
using std::ostringstream;
using std::stoi;
using std::string;
using std::unique_ptr;

namespace arrayfire {
namespace metal {

namespace {

struct MetalDeviceInfo {
    string name;
    size_t recommendedMemoryBytes = 0;
    bool unifiedMemory            = false;
    bool lowPower                 = false;
    bool headless                 = false;
};

string toString(NS::String* value) {
    const char* utf8 = value ? value->utf8String() : nullptr;
    return utf8 ? utf8 : "Unknown Metal Device";
}

MetalDeviceInfo getMetalDeviceInfo(const int device) noexcept {
    MetalDeviceInfo info;
    if (device < 0 || device >= getDeviceCount()) { return info; }

    MTL::Device& selected = getDevice(device);
    info.name              = toString(selected.name());
    info.recommendedMemoryBytes =
        static_cast<size_t>(selected.recommendedMaxWorkingSetSize());
    info.unifiedMemory = selected.hasUnifiedMemory();
    info.lowPower      = selected.isLowPower();
    info.headless      = selected.isHeadless();
    return info;
}

}  // namespace

static string get_system() {
    string arch = (sizeof(void*) == 4) ? "32-bit " : "64-bit ";

    return arch +
#if defined(OS_LNX)
           "Linux";
#elif defined(OS_WIN)
           "Windows";
#elif defined(OS_MAC)
           "Mac OSX";
#endif
}

int getBackend() { return AF_BACKEND_METAL; }

string getDeviceInfo() noexcept {
    ostringstream info;

    info << "ArrayFire v" << AF_VERSION << " (Metal, " << get_system()
         << ", build " << AF_REVISION << ")" << endl;

    const int active = getActiveDeviceId();
    for (int device = 0; device < getDeviceCount(); ++device) {
        const MetalDeviceInfo minfo = getMetalDeviceInfo(device);
        string model =
            minfo.name.empty() ? "Unknown Metal Device" : minfo.name;
        const size_t memMB = getDeviceMemorySize(device) / 1048576;

        info << (device == active ? "[" : "-") << device
             << (device == active ? "] " : "- ") << "Apple: "
             << ltrim(model);
        if (memMB) {
            info << ", " << memMB << " MB, ";
        } else {
            info << ", Unknown MB, ";
        }
        info << "Metal";
        if (minfo.unifiedMemory) { info << ", unified memory"; }
        if (minfo.lowPower) { info << ", low power"; }
        if (minfo.headless) { info << ", headless"; }
#ifndef NDEBUG
        info << ", " << AF_COMPILER_STR;
#endif
        info << endl;
    }

    return info.str();
}

bool isDoubleSupported(int device) {
    UNUSED(device);
    return DeviceManager::IS_DOUBLE_SUPPORTED;
}

bool isHalfSupported(int device) {
    UNUSED(device);
    return DeviceManager::IS_HALF_SUPPORTED;
}

void devprop(char* d_name, char* d_platform, char* d_toolkit, char* d_compute) {
    const MetalDeviceInfo minfo =
        getMetalDeviceInfo(static_cast<int>(getActiveDeviceId()));

    snprintf(
        d_name, 64, "%s",
        (minfo.name.empty() ? "Unknown Metal Device" : minfo.name).c_str());
    snprintf(d_platform, 10, "Metal");
    snprintf(d_toolkit, 64, "%s", "metal-cpp");
    snprintf(d_compute, 10, "%s", minfo.unifiedMemory ? "UMA" : "GPU");
}

int& getMaxJitSize() {
    constexpr int MAX_JIT_LEN = 100;
    thread_local int length   = 0;
    if (length <= 0) {
        string env_var = getEnvVar("AF_METAL_MAX_JIT_LEN");
        if (!env_var.empty()) {
            int input_len = stoi(env_var);
            length        = input_len > 0 ? input_len : MAX_JIT_LEN;
        } else {
            length = MAX_JIT_LEN;
        }
    }
    return length;
}

int getDeviceCount() { return DeviceManager::getInstance().deviceCount(); }

void init() {
    thread_local const auto& instance = DeviceManager::getInstance();
    UNUSED(instance);
}

int& activeDeviceId() {
    thread_local int device = [] {
        const string deviceEnv = getEnvVar("AF_METAL_DEFAULT_DEVICE");
        if (deviceEnv.empty()) { return 0; }

        std::istringstream stream(deviceEnv);
        int defaultDevice = -1;
        stream >> defaultDevice;
        return defaultDevice >= 0 && defaultDevice < getDeviceCount()
                   ? defaultDevice
                   : 0;
    }();
    return device;
}

int getActiveDeviceId() { return activeDeviceId(); }

size_t getDeviceMemorySize(int device) {
    const MetalDeviceInfo minfo = getMetalDeviceInfo(device);
    if (minfo.recommendedMemoryBytes) { return minfo.recommendedMemoryBytes; }
    return common::getHostMemorySize();
}

size_t getHostMemorySize() { return common::getHostMemorySize(); }

int setDevice(int device) {
    if (device < 0 || device >= getDeviceCount()) { return -1; }
    const int previous = getActiveDeviceId();
    activeDeviceId()   = device;
    return previous;
}

queue& getQueue(int device) {
    if (device < 0) { device = getActiveDeviceId(); }
    DeviceManager& manager = DeviceManager::getInstance();
    if (device >= static_cast<int>(manager.queues.size())) {
        AF_ERROR("No Metal host queue is available", AF_ERR_RUNTIME);
    }
    return *manager.queues[device];
}

MTL::Device& getDevice(int device) {
    if (device < 0) { device = getActiveDeviceId(); }
    DeviceManager& manager = DeviceManager::getInstance();
    if (device < 0 ||
        device >= static_cast<int>(manager.nativeDevices.size())) {
        AF_ERROR("No Metal device is available", AF_ERR_RUNTIME);
    }
    return *manager.nativeDevices[device];
}

MTL::CommandQueue& getCommandQueue(int device) {
    if (device < 0) { device = getActiveDeviceId(); }
    DeviceManager& manager = DeviceManager::getInstance();
    if (device < 0 ||
        device >= static_cast<int>(manager.nativeCommandQueues.size()) ||
        !manager.nativeCommandQueues[device]) {
        AF_ERROR("Could not create a Metal command queue", AF_ERR_RUNTIME);
    }
    return *manager.nativeCommandQueues[device];
}

void submitCommandBuffer(MTL::CommandBuffer* commandBuffer, int device) {
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }

    if (device < 0) { device = getActiveDeviceId(); }
    DeviceManager& manager = DeviceManager::getInstance();
    if (device < 0 ||
        device >= static_cast<int>(manager.lastCommandBuffers.size())) {
        AF_ERROR("No Metal command queue is available", AF_ERR_RUNTIME);
    }

    commandBuffer->addCompletedHandler(
        [&manager, device](MTL::CommandBuffer* completed) {
            if (completed->status() == MTL::CommandBufferStatusError) {
                std::lock_guard<std::mutex> lock(manager.commandMutex);
                manager.commandQueueFailed[device] = true;
            }
        });

    MTL::CommandBuffer* previous = nullptr;
    {
        std::lock_guard<std::mutex> lock(manager.commandMutex);
        commandBuffer->retain();
        commandBuffer->commit();
        previous = manager.lastCommandBuffers[device];
        manager.lastCommandBuffers[device] = commandBuffer;
    }
    if (previous) { previous->release(); }
}

void syncCommandQueue(int device) {
    if (device < 0) { device = getActiveDeviceId(); }
    DeviceManager& manager = DeviceManager::getInstance();
    if (device < 0 ||
        device >= static_cast<int>(manager.lastCommandBuffers.size())) {
        AF_ERROR("No Metal command queue is available", AF_ERR_RUNTIME);
    }

    MTL::CommandBuffer* commandBuffer = nullptr;
    {
        std::lock_guard<std::mutex> lock(manager.commandMutex);
        commandBuffer = manager.lastCommandBuffers[device];
        if (commandBuffer) { commandBuffer->retain(); }
    }
    if (!commandBuffer) { return; }

    commandBuffer->waitUntilCompleted();
    bool failed = commandBuffer->status() == MTL::CommandBufferStatusError;
    {
        std::lock_guard<std::mutex> lock(manager.commandMutex);
        manager.commandQueueFailed[device] =
            manager.commandQueueFailed[device] || failed;
        failed = manager.commandQueueFailed[device];
    }
    commandBuffer->release();
    if (failed) {
        AF_ERROR("Metal command buffer execution failed", AF_ERR_RUNTIME);
    }
}

queue* getQueueHandle(int device) { return &getQueue(device); }

void sync(int device) { getQueue(device).sync(); }

bool& evalFlag() {
    thread_local bool flag = true;
    return flag;
}

MemoryManagerBase& memoryManager() {
    static std::once_flag flag;
    DeviceManager& inst = DeviceManager::getInstance();

    std::call_once(flag, [&]() {
        inst.memManager = unique_ptr<common::DefaultMemoryManager>(
            new common::DefaultMemoryManager(
                std::max(1, getDeviceCount()), common::MAX_BUFFERS,
                AF_MEM_DEBUG || AF_METAL_MEM_DEBUG));
        unique_ptr<Allocator> deviceMemoryManager(new Allocator());
        inst.memManager->setAllocator(move(deviceMemoryManager));
        inst.memManager->initialize();
    });

    return *(inst.memManager.get());
}

MemoryManagerBase& pinnedMemoryManager() {
    static std::once_flag flag;
    DeviceManager& inst = DeviceManager::getInstance();

    std::call_once(flag, [&]() {
        inst.pinnedMemManager = unique_ptr<common::DefaultMemoryManager>(
            new common::DefaultMemoryManager(
                std::max(1, getDeviceCount()), common::MAX_BUFFERS,
                AF_MEM_DEBUG || AF_METAL_MEM_DEBUG));
        unique_ptr<AllocatorPinned> pinnedAllocator(new AllocatorPinned());
        inst.pinnedMemManager->setAllocator(move(pinnedAllocator));
        inst.pinnedMemManager->initialize();
    });

    return *(inst.pinnedMemManager.get());
}

void setMemoryManager(unique_ptr<MemoryManagerBase> mgr) {
    return DeviceManager::getInstance().setMemoryManager(move(mgr));
}

void resetMemoryManager() {
    return DeviceManager::getInstance().resetMemoryManager();
}

void setMemoryManagerPinned(unique_ptr<MemoryManagerBase> mgr) {
    return DeviceManager::getInstance().setMemoryManagerPinned(move(mgr));
}

void resetMemoryManagerPinned() {
    return DeviceManager::getInstance().resetMemoryManagerPinned();
}

ForgeManager& forgeManager() { return *(DeviceManager::getInstance().fgMngr); }

}  // namespace metal
}  // namespace arrayfire

namespace {

struct ExportedBuffer {
    size_t references;
    bool managed;
    int device;
};

std::mutex exportedBuffersMutex;
std::unordered_map<MTL::Buffer*, ExportedBuffer> exportedBuffers;

void acquireMetalBuffer(MTL::Buffer* buffer, int device) {
    std::lock_guard<std::mutex> lock(exportedBuffersMutex);
    auto found = exportedBuffers.find(buffer);
    if (found != exportedBuffers.end()) {
        ++found->second.references;
        return;
    }

    const bool managed = detail::memoryManager().allocated(buffer) != 0;
    if (managed) {
        detail::memLock(buffer);
    } else {
        detail::retainBuffer(buffer);
    }
    exportedBuffers.emplace(buffer, ExportedBuffer{1, managed, device});
}

void releaseMetalBuffer(MTL::Buffer* buffer) {
    bool managed = false;
    int device   = -1;
    {
        std::lock_guard<std::mutex> lock(exportedBuffersMutex);
        auto found = exportedBuffers.find(buffer);
        if (found == exportedBuffers.end()) {
            AF_ERROR("Metal buffer has no interop reference", AF_ERR_ARG);
        }
        if (--found->second.references != 0) { return; }
        managed = found->second.managed;
        device  = found->second.device;
        exportedBuffers.erase(found);
    }

    if (managed) {
        const int previous = detail::setDevice(device);
        detail::memUnlock(buffer);
        if (previous >= 0) { detail::setDevice(previous); }
    } else {
        detail::releaseBuffer(buffer);
    }
}

template<typename T>
void getMetalBuffer(af_mtl_buffer* buffer, size_t* offset,
                    const af_array arr) {
    const detail::Array<T>& input = getArray<T>(arr);
    input.eval();
    MTL::Buffer* native = input.getBuffer();
    if (!native) { AF_ERROR("Metal array has no buffer", AF_ERR_NO_MEM); }
    acquireMetalBuffer(native, input.getDevId());
    *buffer = native;
    *offset = static_cast<size_t>(input.getOffset()) * sizeof(T);
}

}  // namespace

af_err afmtl_get_device(af_mtl_device* device, int id, const bool retain) {
    try {
        ARG_ASSERT(0, device != nullptr);
        MTL::Device* native = &arrayfire::metal::getDevice(id);
        if (retain) { native->retain(); }
        *device = native;
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afmtl_get_command_queue(af_mtl_command_queue* queue, int id,
                               const bool retain) {
    try {
        ARG_ASSERT(0, queue != nullptr);
        MTL::CommandQueue* native = &arrayfire::metal::getCommandQueue(id);
        if (retain) { native->retain(); }
        *queue = native;
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afmtl_set_device(af_mtl_device device) {
    try {
        ARG_ASSERT(0, device != nullptr);
        int selected = -1;
        for (int id = 0; id < arrayfire::metal::getDeviceCount(); ++id) {
            if (&arrayfire::metal::getDevice(id) == device) {
                selected = id;
                break;
            }
        }
        if (selected < 0) {
            AF_ERROR("Metal device is not managed by ArrayFire", AF_ERR_DEVICE);
        }
        arrayfire::metal::setDevice(selected);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afmtl_get_buffer(af_mtl_buffer* buffer, size_t* offset,
                        const af_array arr) {
    try {
        ARG_ASSERT(0, buffer != nullptr);
        ARG_ASSERT(1, offset != nullptr);
        ARG_ASSERT(2, arr != nullptr);

        switch (arrayfire::common::getInfo(arr).getType()) {
            case f32: getMetalBuffer<float>(buffer, offset, arr); break;
            case f64: getMetalBuffer<double>(buffer, offset, arr); break;
            case c32: getMetalBuffer<detail::cfloat>(buffer, offset, arr); break;
            case c64:
                getMetalBuffer<detail::cdouble>(buffer, offset, arr);
                break;
            case b8: getMetalBuffer<char>(buffer, offset, arr); break;
            case s32: getMetalBuffer<int>(buffer, offset, arr); break;
            case u32: getMetalBuffer<detail::uint>(buffer, offset, arr); break;
            case s8: getMetalBuffer<detail::schar>(buffer, offset, arr); break;
            case u8: getMetalBuffer<detail::uchar>(buffer, offset, arr); break;
            case s64: getMetalBuffer<detail::intl>(buffer, offset, arr); break;
            case u64: getMetalBuffer<detail::uintl>(buffer, offset, arr); break;
            case s16: getMetalBuffer<short>(buffer, offset, arr); break;
            case u16:
                getMetalBuffer<detail::ushort>(buffer, offset, arr);
                break;
            case f16:
                getMetalBuffer<arrayfire::common::half>(buffer, offset, arr);
                break;
            default: TYPE_ERROR(2, arrayfire::common::getInfo(arr).getType());
        }
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afmtl_submit_command_buffer(af_mtl_command_buffer command_buffer,
                                   int id) {
    try {
        ARG_ASSERT(0, command_buffer != nullptr);
        MTL::CommandQueue* expected =
            &arrayfire::metal::getCommandQueue(id);
        if (command_buffer->commandQueue() != expected) {
            AF_ERROR("Metal command buffer belongs to a different queue",
                     AF_ERR_ARG);
        }
        arrayfire::metal::submitCommandBuffer(command_buffer, id);
    }
    CATCHALL;
    return AF_SUCCESS;
}

af_err afmtl_release_buffer(af_mtl_buffer buffer) {
    try {
        ARG_ASSERT(0, buffer != nullptr);
        releaseMetalBuffer(buffer);
    }
    CATCHALL;
    return AF_SUCCESS;
}
