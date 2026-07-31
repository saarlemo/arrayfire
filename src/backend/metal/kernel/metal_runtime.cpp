/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Metal.hpp>

#include <common/TemplateArg.hpp>
#include <common/kernel_cache.hpp>
#include <err_metal.hpp>
#include <kernel/metal_runtime.hpp>
#include <platform.hpp>
#include <metal_kernel_headers/Array.hpp>
#include <metal_kernel_headers/anisotropic_diffusion.hpp>
#include <metal_kernel_headers/approx.hpp>
#include <metal_kernel_headers/assign.hpp>
#include <metal_kernel_headers/bilateral.hpp>
#include <metal_kernel_headers/canny.hpp>
#include <metal_kernel_headers/convolve.hpp>
#include <metal_kernel_headers/copy.hpp>
#include <metal_kernel_headers/cholesky.hpp>
#include <metal_kernel_headers/diagonal.hpp>
#include <metal_kernel_headers/diff.hpp>
#include <metal_kernel_headers/dot.hpp>
#include <metal_kernel_headers/exampleFunction.hpp>
#include <metal_kernel_headers/fast.hpp>
#include <metal_kernel_headers/fftconvolve.hpp>
#include <metal_kernel_headers/flood_fill.hpp>
#include <metal_kernel_headers/gradient.hpp>
#include <metal_kernel_headers/harris.hpp>
#include <metal_kernel_headers/histogram.hpp>
#include <metal_kernel_headers/hsv_rgb.hpp>
#include <metal_kernel_headers/identity.hpp>
#include <metal_kernel_headers/iir.hpp>
#include <metal_kernel_headers/index.hpp>
#include <metal_kernel_headers/iota.hpp>
#include <metal_kernel_headers/ireduce.hpp>
#include <metal_kernel_headers/join.hpp>
#include <metal_kernel_headers/lookup.hpp>
#include <metal_kernel_headers/lu.hpp>
#include <metal_kernel_headers/solve.hpp>
#include <metal_kernel_headers/match_template.hpp>
#include <metal_kernel_headers/mean.hpp>
#include <metal_kernel_headers/meanshift.hpp>
#include <metal_kernel_headers/medfilt.hpp>
#include <metal_kernel_headers/moments.hpp>
#include <metal_kernel_headers/morph.hpp>
#include <metal_kernel_headers/nearest_neighbour.hpp>
#include <metal_kernel_headers/orb.hpp>
#include <metal_kernel_headers/qr.hpp>
#include <metal_kernel_headers/pad_borders.hpp>
#include <metal_kernel_headers/random_engine.hpp>
#include <metal_kernel_headers/range.hpp>
#include <metal_kernel_headers/reduce.hpp>
#include <metal_kernel_headers/regions.hpp>
#include <metal_kernel_headers/reorder.hpp>
#include <metal_kernel_headers/resize.hpp>
#include <metal_kernel_headers/rotate.hpp>
#include <metal_kernel_headers/scan.hpp>
#include <metal_kernel_headers/scan_by_key.hpp>
#include <metal_kernel_headers/select.hpp>
#include <metal_kernel_headers/shift.hpp>
#include <metal_kernel_headers/sift.hpp>
#include <metal_kernel_headers/sobel.hpp>
#include <metal_kernel_headers/sort.hpp>
#include <metal_kernel_headers/sort_by_key.hpp>
#include <metal_kernel_headers/sort_helper.hpp>
#include <metal_kernel_headers/sparse.hpp>
#include <metal_kernel_headers/sparse_blas.hpp>
#include <metal_kernel_headers/sparse_arith.hpp>
#include <metal_kernel_headers/susan.hpp>
#include <metal_kernel_headers/svd.hpp>
#include <metal_kernel_headers/tile.hpp>
#include <metal_kernel_headers/topk.hpp>
#include <metal_kernel_headers/transform.hpp>
#include <metal_kernel_headers/transpose.hpp>
#include <metal_kernel_headers/triangle.hpp>
#include <metal_kernel_headers/unwrap.hpp>
#include <metal_kernel_headers/wrap.hpp>

#include <algorithm>
#include <string>

namespace arrayfire {
namespace metal {
namespace kernel {
namespace {

const std::string computeSourceStorage = [] {
    std::string source;
    const auto append = [&source](const common::Source& kernelSource) {
        source.append(kernelSource.ptr, kernelSource.length - 1);
        source.push_back('\n');
    };

    append(copy_metal_src);
    append(cholesky_metal_src);
    append(anisotropic_diffusion_metal_src);
    append(Array_metal_src);
    append(approx_metal_src);
    append(assign_metal_src);
    append(bilateral_metal_src);
    append(canny_metal_src);
    append(convolve_metal_src);
    append(exampleFunction_metal_src);
    append(range_metal_src);
    append(random_engine_metal_src);
    append(reduce_metal_src);
    append(iota_metal_src);
    append(ireduce_metal_src);
    append(identity_metal_src);
    append(iir_metal_src);
    append(index_metal_src);
    append(tile_metal_src);
    append(topk_metal_src);
    append(shift_metal_src);
    append(sift_metal_src);
    append(reorder_metal_src);
    append(regions_metal_src);
    append(rotate_metal_src);
    append(resize_metal_src);
    append(scan_metal_src);
    append(scan_by_key_metal_src);
    append(select_metal_src);
    append(join_metal_src);
    append(lookup_metal_src);
    append(lu_metal_src);
    append(solve_metal_src);
    append(match_template_metal_src);
    append(moments_metal_src);
    append(morph_metal_src);
    append(meanshift_metal_src);
    append(mean_metal_src);
    append(medfilt_metal_src);
    append(nearest_neighbour_metal_src);
    append(orb_metal_src);
    append(qr_metal_src);
    append(pad_borders_metal_src);
    append(diagonal_metal_src);
    append(diff_metal_src);
    append(dot_metal_src);
    append(flood_fill_metal_src);
    append(fast_metal_src);
    append(fftconvolve_metal_src);
    append(triangle_metal_src);
    append(gradient_metal_src);
    append(hsv_rgb_metal_src);
    append(histogram_metal_src);
    append(harris_metal_src);
    append(sobel_metal_src);
    append(sort_helper_metal_src);
    append(sort_metal_src);
    append(sort_by_key_metal_src);
    append(sparse_metal_src);
    append(sparse_blas_metal_src);
    append(sparse_arith_metal_src);
    append(susan_metal_src);
    append(svd_metal_src);
    append(transpose_metal_src);
    append(transform_metal_src);
    append(unwrap_metal_src);
    append(wrap_metal_src);
    return source;
}();

std::string description(NS::Error* error, const char* fallback) {
    if (!error || !error->localizedDescription()) { return fallback; }
    const char* value = error->localizedDescription()->utf8String();
    return value ? value : fallback;
}

}  // namespace

MTL::ComputePipelineState* metalPipeline(const char* functionName) {
    const common::Source source{computeSourceStorage.data(),
                                computeSourceStorage.size(), 0};
    return common::getKernel(functionName, {{source}}, TemplateArgs()).get();
}

MTL::CommandQueue* metalCommandQueue() {
    return &arrayfire::metal::getCommandQueue();
}

MTL::Device* metalDevice() { return &arrayfire::metal::getDevice(); }

std::string metalErrorDescription(NS::Error* error, const char* fallback) {
    return description(error, fallback);
}

void launchOutputKernel(BufferParam output, const size_t outputBytes,
                        const size_t dispatchTotal, const void* params,
                        const size_t paramsBytes, const char* functionName,
                        const char* operationName) {
    if (outputBytes == 0 || dispatchTotal == 0) return;
    if (!output.buffer)
        AF_ERROR("Could not allocate a Metal buffer", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(output.buffer, output.offset, 0);
    encoder->setBytes(params, paramsBytes, 1);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(dispatchTotal, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string fallback =
            std::string("Metal ") + operationName + " dispatch failed";
        const std::string message =
            description(commandBuffer->error(), fallback.c_str());
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchInputTwoOutputKernel(
    BufferParam firstOutput, BufferParam secondOutput,
    const size_t outputBytes, BufferParam input, const size_t inputBytes,
    const size_t dispatchTotal, const void* params, const size_t paramsBytes,
    const char* functionName, const char* operationName) {
    if (outputBytes == 0 || dispatchTotal == 0) return;
    UNUSED(inputBytes);
    if (!input.buffer || !firstOutput.buffer || !secondOutput.buffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(firstOutput.buffer, firstOutput.offset, 1);
    encoder->setBuffer(secondOutput.buffer, secondOutput.offset, 2);
    encoder->setBytes(params, paramsBytes, 3);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(dispatchTotal, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string fallback =
            std::string("Metal ") + operationName + " dispatch failed";
        const std::string message =
            description(commandBuffer->error(), fallback.c_str());
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchSingleInputKernel(BufferParam output, const size_t outputBytes,
                             const af::dim4& outputDims, BufferParam input,
                             const size_t inputBytes, const void* params,
                             const size_t paramsBytes, const char* functionName,
                             const char* operationName) {
    if (outputBytes == 0) return;
    UNUSED(inputBytes);
    if (!input.buffer || !output.buffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(output.buffer, output.offset, 1);
    encoder->setBytes(params, paramsBytes, 2);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string fallback =
            std::string("Metal ") + operationName + " dispatch failed";
        const std::string message =
            description(commandBuffer->error(), fallback.c_str());
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchTwoInputKernel(BufferParam output, const size_t outputBytes,
                          BufferParam first, const size_t firstBytes,
                          BufferParam second, const size_t secondBytes,
                          const void* params, const size_t paramsBytes,
                          const size_t dispatchTotal, const char* functionName,
                          const char* operationName) {
    if (outputBytes == 0 || dispatchTotal == 0) return;
    UNUSED(firstBytes);
    UNUSED(secondBytes);
    if (!first.buffer || !second.buffer || !output.buffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(first.buffer, first.offset, 0);
    encoder->setBuffer(second.buffer, second.offset, 1);
    encoder->setBuffer(output.buffer, output.offset, 2);
    encoder->setBytes(params, paramsBytes, 3);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(dispatchTotal, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string fallback =
            std::string("Metal ") + operationName + " dispatch failed";
        const std::string message =
            description(commandBuffer->error(), fallback.c_str());
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchThreeInputKernel(BufferParam output, const size_t outputBytes,
                            BufferParam first, const size_t firstBytes,
                            BufferParam second, const size_t secondBytes,
                            BufferParam third, const size_t thirdBytes,
                            const void* params, const size_t paramsBytes,
                            const size_t dispatchTotal,
                            const char* functionName,
                            const char* operationName) {
    if (outputBytes == 0 || dispatchTotal == 0) return;
    UNUSED(firstBytes);
    UNUSED(secondBytes);
    UNUSED(thirdBytes);
    if (!first.buffer || !second.buffer || !third.buffer || !output.buffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(first.buffer, first.offset, 0);
    encoder->setBuffer(second.buffer, second.offset, 1);
    encoder->setBuffer(third.buffer, third.offset, 2);
    encoder->setBuffer(output.buffer, output.offset, 3);
    encoder->setBytes(params, paramsBytes, 4);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(dispatchTotal, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string fallback =
            std::string("Metal ") + operationName + " dispatch failed";
        const std::string message =
            description(commandBuffer->error(), fallback.c_str());
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchFiveInputKernel(
    BufferParam output, const size_t outputBytes, BufferParam first,
    const size_t firstBytes, BufferParam second, const size_t secondBytes,
    BufferParam third, const size_t thirdBytes, BufferParam fourth,
    const size_t fourthBytes, BufferParam fifth, const size_t fifthBytes,
    const void* params, const size_t paramsBytes, const size_t dispatchTotal,
    const char* functionName, const char* operationName) {
    if (outputBytes == 0 || dispatchTotal == 0) return;
    UNUSED(firstBytes);
    UNUSED(secondBytes);
    UNUSED(thirdBytes);
    UNUSED(fourthBytes);
    UNUSED(fifthBytes);
    if (!first.buffer || !second.buffer || !third.buffer || !fourth.buffer ||
        !fifth.buffer || !output.buffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(first.buffer, first.offset, 0);
    encoder->setBuffer(second.buffer, second.offset, 1);
    encoder->setBuffer(third.buffer, third.offset, 2);
    encoder->setBuffer(fourth.buffer, fourth.offset, 3);
    encoder->setBuffer(fifth.buffer, fifth.offset, 4);
    encoder->setBuffer(output.buffer, output.offset, 5);
    encoder->setBytes(params, paramsBytes, 6);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(dispatchTotal, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string fallback =
            std::string("Metal ") + operationName + " dispatch failed";
        const std::string message =
            description(commandBuffer->error(), fallback.c_str());
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchFourInputThreeOutputKernel(
    BufferParam firstOutput, BufferParam secondOutput, BufferParam thirdOutput,
    const size_t outputBytes, BufferParam firstInput,
    const size_t firstInputBytes, BufferParam secondInput,
    const size_t secondInputBytes, BufferParam thirdInput,
    const size_t thirdInputBytes, BufferParam fourthInput,
    const size_t fourthInputBytes, const void* params,
    const size_t paramsBytes, const size_t dispatchTotal,
    const char* functionName, const char* operationName) {
    if (outputBytes == 0 || dispatchTotal == 0) return;
    UNUSED(firstInputBytes);
    UNUSED(secondInputBytes);
    UNUSED(thirdInputBytes);
    UNUSED(fourthInputBytes);
    if (!firstOutput.buffer || !secondOutput.buffer || !thirdOutput.buffer ||
        !firstInput.buffer || !secondInput.buffer || !thirdInput.buffer ||
        !fourthInput.buffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);

    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(firstInput.buffer, firstInput.offset, 0);
    encoder->setBuffer(secondInput.buffer, secondInput.offset, 1);
    encoder->setBuffer(thirdInput.buffer, thirdInput.offset, 2);
    encoder->setBuffer(fourthInput.buffer, fourthInput.offset, 3);
    encoder->setBuffer(firstOutput.buffer, firstOutput.offset, 4);
    encoder->setBuffer(secondOutput.buffer, secondOutput.offset, 5);
    encoder->setBuffer(thirdOutput.buffer, thirdOutput.offset, 6);
    encoder->setBytes(params, paramsBytes, 7);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(dispatchTotal, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string fallback =
            std::string("Metal ") + operationName + " dispatch failed";
        const std::string message =
            description(commandBuffer->error(), fallback.c_str());
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchFloodFillKernel(BufferParam output, const size_t outputBytes,
                           BufferParam image, const size_t imageBytes,
                           BufferParam seedX, const size_t seedXBytes,
                           BufferParam seedY, const size_t seedYBytes,
                           const void* params, const size_t paramsBytes,
                           const void* newValue, const void* lower,
                           const void* upper, const size_t valueBytes,
                           const char* functionName) {
    if (outputBytes == 0) return;
    UNUSED(imageBytes);
    UNUSED(seedXBytes);
    UNUSED(seedYBytes);
    if (!image.buffer || !seedX.buffer || !seedY.buffer || !output.buffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);

    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(image.buffer, image.offset, 0);
    encoder->setBuffer(seedX.buffer, seedX.offset, 1);
    encoder->setBuffer(seedY.buffer, seedY.offset, 2);
    encoder->setBuffer(output.buffer, output.offset, 3);
    encoder->setBytes(params, paramsBytes, 4);
    encoder->setBytes(newValue, valueBytes, 5);
    encoder->setBytes(lower, valueBytes, 6);
    encoder->setBytes(upper, valueBytes, 7);
    encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = description(
            commandBuffer->error(), "Metal flood fill dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
