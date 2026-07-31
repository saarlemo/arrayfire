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

#include <kernel/KParam.hpp>

#include <functional>
#include <memory>
#include <sstream>
#include <string>

namespace arrayfire {
namespace metal {

struct JitBufferInfo {
    dim_t dims[4];
    dim_t strides[4];
};

namespace {

inline void generateParamDeclaration(std::stringstream &kerStream, int id,
                                     bool is_linear,
                                     const std::string &type) {
    // Metal buffer indices are assigned after the complete expression graph
    // has been gathered. jit.cpp emits these declarations with their final
    // [[buffer(n)]] attributes.
    UNUSED(kerStream);
    UNUSED(id);
    UNUSED(is_linear);
    UNUSED(type);
}

inline int setBufferKernelArguments(
    int start_id, bool is_linear,
    std::function<void(int id, const void *ptr, size_t arg_size,
                       bool is_buffer)> &setArg,
    const std::shared_ptr<MTL::Buffer> &buffer, const KParam &info) {
    UNUSED(is_linear);
    MTL::Buffer *native = buffer.get();
    setArg(start_id, &native, static_cast<size_t>(info.offset), true);
    const JitBufferInfo bufferInfo{
        {info.dims[0], info.dims[1], info.dims[2], info.dims[3]},
        {info.strides[0], info.strides[1], info.strides[2], info.strides[3]}};
    setArg(start_id, &bufferInfo, sizeof(bufferInfo), false);
    return start_id + 1;
}

inline void generateBufferOffsets(std::stringstream &kerStream, int id,
                                  bool is_linear,
                                  const std::string &type) {
    UNUSED(is_linear);
    UNUSED(type);
    kerStream << "long idx" << id << " = "
              << "(x < constants.dims" << id
              << "[0] ? x : 0) * constants.strides" << id << "[0] + "
              << "(y < constants.dims" << id
              << "[1] ? y : 0) * constants.strides" << id << "[1] + "
              << "(z < constants.dims" << id
              << "[2] ? z : 0) * constants.strides" << id << "[2] + "
              << "(w < constants.dims" << id
              << "[3] ? w : 0) * constants.strides" << id << "[3];\n";
}

inline void generateBufferRead(std::stringstream &kerStream, int id,
                               const std::string &type) {
    kerStream << type << " val" << id << " = in" << id << "[idx" << id
              << "];\n";
}

inline void generateShiftNodeOffsets(std::stringstream &kerStream, int id,
                                     bool is_linear,
                                     const std::string &type) {
    UNUSED(is_linear);
    UNUSED(type);
    constexpr const char *coordinates[] = {"x", "y", "z", "w"};
    for (int dim = 0; dim < 4; ++dim) {
        kerStream << "long sh_id_" << id << "_" << dim << " = "
                  << "((" << coordinates[dim] << " + constants.shift" << id
                  << "_" << dim << ") % constants.dims" << id << "[" << dim
                  << "] + constants.dims" << id << "[" << dim
                  << "]) % constants.dims" << id << "[" << dim << "];\n";
    }
    kerStream << "long idx" << id << " = "
              << "sh_id_" << id << "_0 * constants.strides" << id
              << "[0] + sh_id_" << id << "_1 * constants.strides" << id
              << "[1] + sh_id_" << id << "_2 * constants.strides" << id
              << "[2] + sh_id_" << id << "_3 * constants.strides" << id
              << "[3];\n";
}

inline void generateShiftNodeRead(std::stringstream &kerStream, int id,
                                  const std::string &type) {
    generateBufferRead(kerStream, id, type);
}

}  // namespace
}  // namespace metal
}  // namespace arrayfire
