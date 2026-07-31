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
#include <jit.hpp>
#include <af/traits.hpp>

#include <utility>
#include <vector>

namespace arrayfire {
namespace metal {
namespace kernel {

template<typename T>
void evalMultipleMetal(std::vector<Param<T>> arrays,
                       std::vector<common::Node_ptr> outputNodes) {
    std::vector<JitOutput> outputs;
    outputs.reserve(arrays.size());
    for (const Param<T> &array : arrays) {
        outputs.push_back(
            {array.bufferParam(), array.dims(), array.strides(),
             static_cast<af::dtype>(af::dtype_traits<T>::af_type)});
    }
    evalNodesMetal(std::move(outputs), std::move(outputNodes));
}

void arrayAddMetal(Param<float> output, CParam<float> left,
                   CParam<float> right);

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
