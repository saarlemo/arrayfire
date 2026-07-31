/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/jit/Node.hpp>

#include <vector>

namespace arrayfire {
namespace metal {

struct JitOutput {
    BufferParam storage;
    af::dim4 dims;
    af::dim4 strides;
    af::dtype type;
};

bool supportsNativeJit(const std::vector<common::Node_ptr> &outputNodes,
                       size_t outputCount);

void evalNodesMetal(std::vector<JitOutput> outputs,
                    std::vector<common::Node_ptr> outputNodes);

}  // namespace metal
}  // namespace arrayfire
