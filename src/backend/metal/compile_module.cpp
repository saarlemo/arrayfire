/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Metal.hpp>

#include <Kernel.hpp>
#include <Module.hpp>
#include <common/Logger.hpp>
#include <common/compile_module.hpp>
#include <common/defines.hpp>
#include <common/deterministicHash.hpp>
#include <common/kernel_cache.hpp>
#include <common/util.hpp>
#include <err_metal.hpp>
#include <platform.hpp>
#include <af/version.h>

#ifdef AF_METAL_PRECOMPILED_KERNELS
#include <dispatch/dispatch.h>
#include <metal_kernel_library/arrayfire_metal_kernels.hpp>
#endif

#include <nonstd/span.hpp>

#include <chrono>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using arrayfire::common::getCacheDirectory;
using arrayfire::common::loggerFactory;
using arrayfire::common::makeTempFilename;
using arrayfire::common::removeFile;
using arrayfire::common::renameFile;
using arrayfire::metal::Kernel;
using arrayfire::metal::Module;
using arrayfire::metal::ModuleState;
using nonstd::span;
using spdlog::logger;
using std::string;
using std::to_string;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

namespace {

class ScopedAutoreleasePool {
   public:
    ScopedAutoreleasePool() : pool_(NS::AutoreleasePool::alloc()->init()) {}
    ~ScopedAutoreleasePool() { pool_->release(); }

    ScopedAutoreleasePool(const ScopedAutoreleasePool&)            = delete;
    ScopedAutoreleasePool& operator=(const ScopedAutoreleasePool&) = delete;

   private:
    NS::AutoreleasePool* pool_;
};

logger* getLogger() {
    static std::shared_ptr<logger> logger(loggerFactory("jit"));
    return logger.get();
}

string errorDescription(NS::Error* error, const char* fallback) {
    if (!error || !error->localizedDescription()) { return fallback; }
    const char* value = error->localizedDescription()->utf8String();
    return value ? value : fallback;
}

string sourceWithDefinitions(span<const string> sources,
                             span<const string> options) {
    std::ostringstream output;
    for (const string& option : options) {
        size_t start = string::npos;
        if (option.compare(0, 3, "-D ") == 0) {
            start = 3;
        } else if (option.compare(0, 2, "-D") == 0) {
            start = 2;
        }
        if (start == string::npos) { continue; }

        string definition   = option.substr(start);
        const size_t equals = definition.find('=');
        if (equals != string::npos) { definition[equals] = ' '; }
        output << "#define " << definition << '\n';
    }
    for (const string& source : sources) { output << source << '\n'; }
    return output.str();
}

NS::SharedPtr<MTL::Library> compileLibrary(MTL::Device& device,
                                           span<const string> sources,
                                           span<const string> options,
                                           const bool isJIT,
                                           NS::Error** error) {
#ifdef AF_METAL_PRECOMPILED_KERNELS
    if (!isJIT) {
        dispatch_data_t data = dispatch_data_create(
            arrayfire::metal::kernel::arrayfire_metal_kernels_metallib_src.ptr,
            arrayfire::metal::kernel::arrayfire_metal_kernels_metallib_src
                .length,
            nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        NS::SharedPtr<MTL::Library> library =
            NS::TransferPtr(device.newLibrary(data, error));
        dispatch_release(data);
        return library;
    }
#else
    UNUSED(isJIT);
#endif

    const string source = sourceWithDefinitions(sources, options);
    return NS::TransferPtr(device.newLibrary(
        NS::String::string(source.c_str(), NS::UTF8StringEncoding), nullptr,
        error));
}

#ifdef AF_CACHE_KERNELS_TO_DISK

string pipelineCacheFilename(const ModuleState& module, const string& name) {
    const size_t key = deterministicHash(name, deterministicHash(module.key));
    const uint64_t registry =
        arrayfire::metal::getDevice(module.device).registryID();
    return "KER" + to_string(key) + "_MTL_" + to_string(registry) + "_AF_" +
           to_string(AF_API_VERSION_CURRENT) + ".binary.metallib";
}

bool fileExists(const string& path) {
    std::ifstream input(path, std::ios::binary);
    return input.good();
}

NS::SharedPtr<MTL::BinaryArchive> makeArchive(MTL::Device& device,
                                              const string& cacheFile,
                                              bool* loaded) {
    *loaded          = false;
    NS::Error* error = nullptr;
    auto descriptor =
        NS::TransferPtr(MTL::BinaryArchiveDescriptor::alloc()->init());
    if (fileExists(cacheFile)) {
        descriptor->setUrl(NS::URL::fileURLWithPath(
            NS::String::string(cacheFile.c_str(), NS::UTF8StringEncoding)));
        auto archive =
            NS::TransferPtr(device.newBinaryArchive(descriptor.get(), &error));
        if (archive) {
            *loaded = true;
            return archive;
        }
        AF_TRACE("Metal cache archive {} could not be loaded: {}", cacheFile,
                 errorDescription(error, "unknown Metal archive error"));
        removeFile(cacheFile);
    }

    descriptor->setUrl(nullptr);
    error = nullptr;
    auto archive =
        NS::TransferPtr(device.newBinaryArchive(descriptor.get(), &error));
    if (!archive) {
        AF_TRACE("Metal cache archive could not be created: {}",
                 errorDescription(error, "unknown Metal archive error"));
    }
    return archive;
}

void saveArchive(MTL::BinaryArchive& archive, const string& cacheFile) {
    const string tempFile = getCacheDirectory() + AF_PATH_SEPARATOR +
                            makeTempFilename() + ".binary.metallib";
    NS::Error* error = nullptr;
    const bool serialized =
        archive.serializeToURL(NS::URL::fileURLWithPath(NS::String::string(
                                   tempFile.c_str(), NS::UTF8StringEncoding)),
                               &error);
    if (!serialized) {
        AF_TRACE("Metal cache archive {} could not be serialized: {}",
                 cacheFile,
                 errorDescription(error, "unknown Metal archive error"));
        removeFile(tempFile);
        return;
    }
    if (!renameFile(tempFile, cacheFile)) {
        AF_TRACE("Metal cache archive {} could not be installed", cacheFile);
        removeFile(tempFile);
    }
}

NS::SharedPtr<MTL::ComputePipelineState> cachedPipeline(ModuleState& module,
                                                        MTL::Function& function,
                                                        const string& name) {
    const string& directory = getCacheDirectory();
    if (directory.empty()) { return {}; }

    MTL::Device& device = arrayfire::metal::getDevice(module.device);
    const string cacheFile =
        directory + AF_PATH_SEPARATOR + pipelineCacheFilename(module, name);

    bool loaded  = false;
    auto archive = makeArchive(device, cacheFile, &loaded);
    if (!archive) { return {}; }

    auto descriptor =
        NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    descriptor->setComputeFunction(&function);
    descriptor->setBinaryArchives(NS::Array::array(archive.get()));

    NS::Error* error = nullptr;
    if (loaded) {
        auto state = NS::TransferPtr(device.newComputePipelineState(
            descriptor.get(), MTL::PipelineOptionFailOnBinaryArchiveMiss,
            nullptr, &error));
        if (state) {
            AF_TRACE("Metal pipeline {} loaded from {}", name, cacheFile);
            return state;
        }
        AF_TRACE("Metal pipeline {} missed archive {}: {}", name, cacheFile,
                 errorDescription(error, "unknown Metal archive miss"));
    }

    descriptor->setBinaryArchives(nullptr);
    error = nullptr;
    if (!archive->addComputePipelineFunctions(descriptor.get(), &error)) {
        AF_TRACE("Metal pipeline {} could not be added to archive {}: {}", name,
                 cacheFile,
                 errorDescription(error, "unknown Metal archive error"));
        return {};
    }

    descriptor->setBinaryArchives(NS::Array::array(archive.get()));
    error      = nullptr;
    auto state = NS::TransferPtr(device.newComputePipelineState(
        descriptor.get(), MTL::PipelineOptionNone, nullptr, &error));
    if (!state) {
        AF_TRACE("Metal pipeline {} could not use archive {}: {}", name,
                 cacheFile,
                 errorDescription(error, "unknown Metal pipeline error"));
        return {};
    }

    saveArchive(*archive.get(), cacheFile);
    AF_TRACE("Metal pipeline {} cached in {}", name, cacheFile);
    return state;
}

#endif

NS::SharedPtr<MTL::ComputePipelineState> createPipeline(ModuleState& module,
                                                        const string& name) {
    ScopedAutoreleasePool pool;
    auto function = NS::TransferPtr(module.library->newFunction(
        NS::String::string(name.c_str(), NS::UTF8StringEncoding)));
    if (!function) {
        AF_ERROR(("Could not load Metal kernel " + name).c_str(),
                 AF_ERR_RUNTIME);
    }

#ifdef AF_CACHE_KERNELS_TO_DISK
    auto cachedState = cachedPipeline(module, *function.get(), name);
    if (cachedState) { return cachedState; }
#endif

    NS::Error* error = nullptr;
    auto state =
        NS::TransferPtr(arrayfire::metal::getDevice(module.device)
                            .newComputePipelineState(function.get(), &error));
    if (!state) {
        const string message =
            errorDescription(error, "Metal pipeline creation failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    return state;
}

}  // namespace

namespace arrayfire {
namespace common {

Module compileModule(const string& moduleKey, span<const string> sources,
                     span<const string> options,
                     span<const string> kernelInstances, const bool isJIT) {
    UNUSED(kernelInstances);
    const int deviceId  = metal::getActiveDeviceId();
    MTL::Device& device = metal::getDevice(deviceId);

    const auto begin = high_resolution_clock::now();
    ScopedAutoreleasePool pool;
    NS::Error* error = nullptr;
    auto library     = compileLibrary(device, sources, options, isJIT, &error);
    if (!library) {
        const string message =
            errorDescription(error, "Metal library compilation failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }

    AF_TRACE("{{ {:<20} : {{ compile:{:>5} ms, Metal }} }}", moduleKey,
             duration_cast<milliseconds>(high_resolution_clock::now() - begin)
                 .count());
    return Module(
        std::make_shared<ModuleState>(std::move(library), deviceId, moduleKey));
}

Module loadModuleFromDisk(const int device, const string& moduleKey,
                          const bool isJIT) {
    UNUSED(device);
    UNUSED(moduleKey);
    UNUSED(isJIT);
    // Metal persists executable pipeline states through MTLBinaryArchive in
    // getKernel(). MTLLibrary objects created from generated source cannot be
    // serialized by the public Metal API.
    return {};
}

Kernel getKernel(const Module& module, const string& name,
                 const bool sourceWasJIT) {
    UNUSED(sourceWasJIT);
    const Module::ModuleType state = module.get();
    if (!state || !state->library) {
        AF_ERROR("Metal kernel requested from an empty module",
                 AF_ERR_INTERNAL);
    }

    std::lock_guard<std::mutex> lock(state->pipelineMutex);
    const auto found = state->pipelines.find(name);
    if (found != state->pipelines.end()) {
        return Kernel(name, state, found->second.get());
    }

    auto pipeline = createPipeline(*state, name);
    auto inserted = state->pipelines.emplace(name, std::move(pipeline));
    return Kernel(name, state, inserted.first->second.get());
}

}  // namespace common
}  // namespace arrayfire
