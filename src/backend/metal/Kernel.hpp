/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Module.hpp>

#include <string>
#include <utility>

namespace arrayfire {
namespace metal {

class Kernel {
   public:
    Kernel() = default;
    Kernel(std::string name, Module::ModuleType module,
           MTL::ComputePipelineState* pipeline)
        : name_(std::move(name))
        , module_(std::move(module))
        , pipeline_(pipeline) {}

    MTL::ComputePipelineState* get() const { return pipeline_; }

    explicit operator bool() const { return pipeline_ != nullptr; }

   private:
    std::string name_;
    Module::ModuleType module_;
    MTL::ComputePipelineState* pipeline_ = nullptr;
};

}  // namespace metal
}  // namespace arrayfire
