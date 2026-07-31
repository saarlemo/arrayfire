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
#include <af/dim4.hpp>

#include <string>

namespace NS {
class Error;
}  // namespace NS

namespace MTL {
class CommandQueue;
class ComputePipelineState;
class Device;
}  // namespace MTL

namespace arrayfire {
namespace metal {
namespace kernel {

MTL::ComputePipelineState* metalPipeline(const char* functionName);
MTL::CommandQueue* metalCommandQueue();
MTL::Device* metalDevice();
std::string metalErrorDescription(NS::Error* error, const char* fallback);

void launchOutputKernel(BufferParam output, size_t outputBytes,
                        size_t dispatchTotal, const void* params,
                        size_t paramsBytes, const char* functionName,
                        const char* operationName);

void launchInputTwoOutputKernel(
    BufferParam firstOutput, BufferParam secondOutput, size_t outputBytes,
    BufferParam input, size_t inputBytes, size_t dispatchTotal,
    const void* params, size_t paramsBytes, const char* functionName,
    const char* operationName);

void launchSingleInputKernel(BufferParam output, size_t outputBytes,
                             const af::dim4& outputDims, BufferParam input,
                             size_t inputBytes, const void* params,
                             size_t paramsBytes, const char* functionName,
                             const char* operationName);

void launchTwoInputKernel(BufferParam output, size_t outputBytes,
                          BufferParam first, size_t firstBytes,
                          BufferParam second, size_t secondBytes,
                          const void* params, size_t paramsBytes,
                          size_t dispatchTotal, const char* functionName,
                          const char* operationName);

void launchThreeInputKernel(BufferParam output, size_t outputBytes,
                            BufferParam first, size_t firstBytes,
                            BufferParam second, size_t secondBytes,
                            BufferParam third, size_t thirdBytes,
                            const void* params, size_t paramsBytes,
                            size_t dispatchTotal, const char* functionName,
                            const char* operationName);

void launchFiveInputKernel(
    BufferParam output, size_t outputBytes, BufferParam first,
    size_t firstBytes, BufferParam second, size_t secondBytes,
    BufferParam third, size_t thirdBytes, BufferParam fourth,
    size_t fourthBytes, BufferParam fifth, size_t fifthBytes,
    const void* params, size_t paramsBytes, size_t dispatchTotal,
    const char* functionName, const char* operationName);

void launchFourInputThreeOutputKernel(
    BufferParam firstOutput, BufferParam secondOutput, BufferParam thirdOutput,
    size_t outputBytes, BufferParam firstInput, size_t firstInputBytes,
    BufferParam secondInput, size_t secondInputBytes, BufferParam thirdInput,
    size_t thirdInputBytes, BufferParam fourthInput, size_t fourthInputBytes,
    const void* params, size_t paramsBytes, size_t dispatchTotal,
    const char* functionName, const char* operationName);

void launchFloodFillKernel(
    BufferParam output, size_t outputBytes, BufferParam image,
    size_t imageBytes, BufferParam seedX, size_t seedXBytes, BufferParam seedY,
    size_t seedYBytes, const void* params, size_t paramsBytes,
    const void* newValue, const void* lower, const void* upper,
    size_t valueBytes, const char* functionName);

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
