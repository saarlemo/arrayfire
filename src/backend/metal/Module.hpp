/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Metal.hpp>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace arrayfire {
namespace metal {

struct ModuleState {
    ModuleState(NS::SharedPtr<MTL::Library> library, int device,
                std::string key)
        : library(std::move(library)), device(device), key(std::move(key)) {}

    NS::SharedPtr<MTL::Library> library;
    int device;
    std::string key;
    std::mutex pipelineMutex;
    std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>
        pipelines;
};

class Module {
   public:
    using ModuleType = std::shared_ptr<ModuleState>;

    Module() = default;
    explicit Module(ModuleType module) : module_(std::move(module)) {}

    explicit operator bool() const { return module_ && module_->library.get(); }

    void unload() { module_.reset(); }

    const ModuleType& get() const { return module_; }

   private:
    ModuleType module_;
};

}  // namespace metal
}  // namespace arrayfire
