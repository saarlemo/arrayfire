/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Metal.hpp>

#include <Array.hpp>
#include <err_metal.hpp>
#include <kernel/dispatch_common.hpp>
#include <kernel/metal_runtime.hpp>
#include <Param.hpp>
#include <platform.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalRange(const af_dtype type) noexcept {
    return rangeFunctionName(type) != nullptr;
}

bool supportsMetalCopy(const af_dtype type) noexcept {
    return copyFunctionName(type) != nullptr;
}

bool supportsMetalResize(const af_dtype type) noexcept {
    return resizeFunctionName(type) != nullptr;
}

bool supportsMetalPadBorders(const af_dtype type) noexcept {
    return padFunctionName(type) != nullptr;
}

bool supportsMetalIota(const af_dtype type) noexcept {
    return iotaFunctionName(type) != nullptr;
}

bool supportsMetalIdentity(const af_dtype type) noexcept {
    return identityFunctionName(type) != nullptr;
}

bool supportsMetalTile(const af_dtype type) noexcept {
    return tileFunctionName(type) != nullptr;
}

bool supportsMetalShift(const af_dtype type) noexcept {
    return shiftFunctionName(type) != nullptr;
}

bool supportsMetalReorder(const af_dtype type) noexcept {
    return reorderFunctionName(type) != nullptr;
}

bool supportsMetalSelect(const af_dtype type) noexcept {
    return selectFunctionName(type, false) != nullptr;
}

bool supportsMetalJoin(const af_dtype type) noexcept {
    return joinFunctionName(type) != nullptr;
}

bool supportsMetalLookup(const af_dtype inputType,
                         const af_dtype indexType) noexcept {
    return lookupFunctionName(inputType) != nullptr &&
           lookupIndexType(indexType) >= 0;
}

bool supportsMetalDiagonal(const af_dtype type) noexcept {
    return diagonalFunctionName(type, true) != nullptr;
}

bool supportsMetalDiff(const af_dtype type) noexcept {
    return diffFunctionName(type, false) != nullptr;
}

bool supportsMetalTriangle(const af_dtype type) noexcept {
    return triangleFunctionName(type) != nullptr;
}

bool supportsMetalTranspose(const af_dtype type) noexcept {
    return transposeFunctionName(type, false) != nullptr;
}

bool supportsMetalUnwrap(const af_dtype type) noexcept {
    return unwrapFunctionName(type) != nullptr;
}

bool supportsMetalWrap(const af_dtype type) noexcept {
    return wrapFunctionName(type) != nullptr;
}

bool supportsMetalGradient(const af_dtype type) noexcept {
    return gradientFunctionName(type) != nullptr;
}

bool supportsMetalSobel(const af_dtype inputType) noexcept {
    return sobelFunctionName(inputType) != nullptr;
}

bool supportsMetalHsvRgb(const af_dtype type) noexcept {
    return hsvRgbFunctionName(type, true) != nullptr;
}

bool supportsMetalMoments(const af_dtype type) noexcept {
    return momentsFunctionName(type) != nullptr;
}

void launchMetalRange(BufferParam output, const size_t bytes,
                      const af::dim4& dims,
                      const af::dim4& strides, const unsigned sequenceDimension,
                      const af_dtype type) {
    if (bytes == 0) { return; }

    RangeParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]    = static_cast<uint64_t>(dims[i]);
        params.strides[i] = static_cast<uint64_t>(strides[i]);
    }
    params.sequenceDimension = sequenceDimension;

    if (!output.buffer) {
        AF_ERROR("Could not allocate a Metal range buffer", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    auto encoder = NS::RetainPtr(commandBuffer->computeCommandEncoder());
    if (!encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }

    MTL::ComputePipelineState* pipeline =
        metalPipeline(rangeFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(output.buffer, output.offset, 0);
    encoder->setBytes(&params, sizeof(params), 1);

    const size_t total = static_cast<size_t>(dims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());

    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            metalErrorDescription(commandBuffer->error(), "Metal range dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchMetalCopy(BufferParam output, const size_t outputBytes,
                     const af::dim4& dims, const af::dim4& outputStrides,
                     const dim_t outputOffset, BufferParam input,
                     const size_t inputBytes,
                     const af::dim4& inputStrides, const dim_t inputOffset,
                     const af_dtype type) {
    CopyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<int64_t>(inputStrides[i]);
    }
    params.outputOffset = static_cast<int64_t>(outputOffset);
    params.inputOffset  = static_cast<int64_t>(inputOffset);
    launchSingleInputKernel(output, outputBytes, dims, input, inputBytes,
                            &params, sizeof(params), copyFunctionName(type),
                            "copy");
}

void launchMetalResize(BufferParam output, const size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, BufferParam input,
                       const size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides,
                       const af_interp_type method, const af_dtype type) {
    ResizeParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.method = static_cast<uint32_t>(method);
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params), resizeFunctionName(type),
                            "resize");
}

void launchMetalPadBorders(BufferParam output, const size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, BufferParam input,
                           const size_t inputBytes, const af::dim4& inputDims,
                           const af::dim4& inputStrides,
                           const af::dim4& lowerPadding,
                           const af_border_type borderType,
                           const af_dtype type) {
    PadParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        params.lower[i]         = static_cast<int64_t>(lowerPadding[i]);
    }
    params.borderType = static_cast<uint32_t>(borderType);
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params), padFunctionName(type),
                            "pad borders");
}

void launchMetalHsvRgb(BufferParam output, const size_t outputBytes,
                       const af::dim4& outputStrides, BufferParam input,
                       const size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, const bool hsvToRgb,
                       const af_dtype type) {
    HsvRgbParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(inputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    af::dim4 dispatchDims = inputDims;
    dispatchDims[2]       = 1;
    launchSingleInputKernel(output, outputBytes, dispatchDims, input,
                            inputBytes, &params, sizeof(params),
                            hsvRgbFunctionName(type, hsvToRgb), "HSV/RGB");
}

void launchMetalMoments(BufferParam output, const size_t outputBytes,
                        const af::dim4& outputStrides, BufferParam input,
                        const size_t inputBytes, const af::dim4& inputDims,
                        const af::dim4& inputStrides,
                        const af_moment_type moment, const af_dtype type) {
    MomentsParams params{};
    for (int i = 0; i < 4; ++i) {
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.moment = static_cast<uint32_t>(moment);
    const af::dim4 dispatchDims(1, 1, inputDims[2], inputDims[3]);
    launchSingleInputKernel(output, outputBytes, dispatchDims, input,
                            inputBytes, &params, sizeof(params),
                            momentsFunctionName(type), "moments");
}

void launchMetalIota(BufferParam output, const size_t bytes,
                     const af::dim4& dims, const af::dim4& strides,
                     const af::dim4& sourceDims, const af_dtype type) {
    IotaParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]       = static_cast<uint64_t>(dims[i]);
        params.strides[i]    = static_cast<uint64_t>(strides[i]);
        params.sourceDims[i] = static_cast<uint64_t>(sourceDims[i]);
    }
    launchOutputKernel(output, bytes, static_cast<size_t>(dims.elements()),
                       &params, sizeof(params), iotaFunctionName(type), "iota");
}

void launchMetalIdentity(BufferParam output, const size_t bytes,
                         const af::dim4& dims, const af::dim4& strides,
                         const af_dtype type) {
    IdentityParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]    = static_cast<uint64_t>(dims[i]);
        params.strides[i] = static_cast<uint64_t>(strides[i]);
    }
    launchOutputKernel(output, bytes, static_cast<size_t>(dims.elements()),
                       &params, sizeof(params), identityFunctionName(type),
                       "identity");
}

void launchMetalTile(BufferParam output, const size_t outputBytes,
                     const af::dim4& outputDims,
                     const af::dim4& outputStrides, BufferParam input,
                     const size_t inputBytes, const af::dim4& inputDims,
                     const af::dim4& inputStrides, const af_dtype type) {
    TileParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params), tileFunctionName(type),
                            "tile");
}

void launchMetalShift(BufferParam output, const size_t outputBytes,
                      const af::dim4& outputDims,
                      const af::dim4& outputStrides, BufferParam input,
                      const size_t inputBytes, const af::dim4& inputStrides,
                      const af::dim4& shifts, const af_dtype type) {
    ShiftParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        const int64_t dim       = static_cast<int64_t>(outputDims[i]);
        const int64_t shift     = static_cast<int64_t>(shifts[i]);
        params.shifts[i]        = static_cast<uint64_t>(
            -(shift % dim) + dim * static_cast<int64_t>(shift > 0));
    }
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params), shiftFunctionName(type),
                            "shift");
}

void launchMetalReorder(BufferParam output, const size_t outputBytes,
                        const af::dim4& outputDims,
                        const af::dim4& outputStrides, BufferParam input,
                        const size_t inputBytes,
                        const af::dim4& inputStrides,
                        const af::dim4& reorderDims, const af_dtype type) {
    ReorderParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        params.reorderDims[i]   = static_cast<uint64_t>(reorderDims[i]);
    }
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params), reorderFunctionName(type),
                            "reorder");
}

void launchMetalSelect(
    BufferParam output, const size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam condition,
    const size_t conditionBytes, const af::dim4& conditionDims,
    const af::dim4& conditionStrides, BufferParam a, const size_t aBytes,
    const af::dim4& aDims, const af::dim4& aStrides, BufferParam b,
    const size_t bBytes, const af::dim4& bDims, const af::dim4& bStrides,
    const void* scalar, const bool flip, const af_dtype type) {
    if (outputBytes == 0) return;
    UNUSED(conditionBytes);
    UNUSED(aBytes);
    UNUSED(bBytes);
    SelectParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]       = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i]    = static_cast<uint64_t>(outputStrides[i]);
        params.aDims[i]            = static_cast<uint64_t>(aDims[i]);
        params.aStrides[i]         = static_cast<uint64_t>(aStrides[i]);
        params.bDims[i]            = static_cast<uint64_t>(bDims[i]);
        params.bStrides[i]         = static_cast<uint64_t>(bStrides[i]);
        params.conditionDims[i]    = static_cast<uint64_t>(conditionDims[i]);
        params.conditionStrides[i] = static_cast<uint64_t>(conditionStrides[i]);
    }
    params.flip = flip;
    if (!condition.buffer || !a.buffer || !output.buffer ||
        (!scalar && !b.buffer))
        AF_ERROR("Could not allocate Metal select buffers", AF_ERR_NO_MEM);

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = metalPipeline(selectFunctionName(type, scalar != nullptr));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(condition.buffer, condition.offset, 0);
    encoder->setBuffer(a.buffer, a.offset, 1);
    if (scalar) {
        encoder->setBuffer(output.buffer, output.offset, 2);
        encoder->setBytes(
            scalar, outputBytes / static_cast<size_t>(outputDims.elements()), 3);
    } else {
        encoder->setBuffer(b.buffer, b.offset, 2);
        encoder->setBuffer(output.buffer, output.offset, 3);
    }
    encoder->setBytes(&params, sizeof(params), 4);
    const size_t total = static_cast<size_t>(outputDims.elements());
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(total, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            metalErrorDescription(commandBuffer->error(), "Metal select dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchMetalJoinAppend(BufferParam output, const size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, BufferParam input,
                           const size_t inputBytes,
                           const af::dim4& inputDims,
                           const af::dim4& inputStrides,
                           const af::dim4& outputOffset,
                           const af_dtype type) {
    JoinParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        params.outputOffset[i]  = static_cast<uint64_t>(outputOffset[i]);
    }
    launchSingleInputKernel(output, outputBytes, inputDims, input, inputBytes,
                            &params, sizeof(params), joinFunctionName(type),
                            "join");
}

void launchMetalLookup(BufferParam output, const size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, BufferParam input,
                       const size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, BufferParam indices,
                       const size_t indexBytes, const unsigned dimension,
                       const af_dtype inputType, const af_dtype indexType) {
    LookupParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.dimension = dimension;
    params.indexType = static_cast<uint32_t>(lookupIndexType(indexType));
    launchTwoInputKernel(
        output, outputBytes, input, inputBytes, indices, indexBytes, &params,
        sizeof(params), static_cast<size_t>(outputDims.elements()),
        lookupFunctionName(inputType), "lookup");
}

void launchMetalDiagonal(BufferParam output, const size_t outputBytes,
                         const af::dim4& outputDims,
                         const af::dim4& outputStrides, BufferParam input,
                         const size_t inputBytes, const af::dim4& inputDims,
                         const af::dim4& inputStrides, const int diagonal,
                         const af_dtype type, const bool create) {
    DiagonalParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.diagonal = diagonal;
    launchSingleInputKernel(
        output, outputBytes, outputDims, input, inputBytes, &params,
        sizeof(params), diagonalFunctionName(type, create),
        create ? "diagonal create" : "diagonal extract");
}

void launchMetalDiagCreate(BufferParam output, const size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, BufferParam input,
                           const size_t inputBytes, const af::dim4& inputDims,
                           const af::dim4& inputStrides, const int diagonal,
                           const af_dtype type) {
    launchMetalDiagonal(output, outputBytes, outputDims, outputStrides, input,
                        inputBytes, inputDims, inputStrides, diagonal, type,
                        true);
}

void launchMetalDiagExtract(BufferParam output, const size_t outputBytes,
                            const af::dim4& outputDims,
                            const af::dim4& outputStrides, BufferParam input,
                            const size_t inputBytes, const af::dim4& inputDims,
                            const af::dim4& inputStrides, const int diagonal,
                            const af_dtype type) {
    launchMetalDiagonal(output, outputBytes, outputDims, outputStrides, input,
                        inputBytes, inputDims, inputStrides, diagonal, type,
                        false);
}

void launchMetalDiff(BufferParam output, const size_t outputBytes,
                     const af::dim4& outputDims,
                     const af::dim4& outputStrides, BufferParam input,
                     const size_t inputBytes, const af::dim4& inputStrides,
                     const unsigned dimension, const bool secondOrder,
                     const af_dtype type) {
    DiffParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.dimension = dimension;
    launchSingleInputKernel(
        output, outputBytes, outputDims, input, inputBytes, &params,
        sizeof(params), diffFunctionName(type, secondOrder), "diff");
}

void launchMetalTriangle(BufferParam output, const size_t outputBytes,
                         const af::dim4& outputDims,
                         const af::dim4& outputStrides, BufferParam input,
                         const size_t inputBytes,
                         const af::dim4& inputStrides, const bool upper,
                         const bool unitDiagonal, const af_dtype type) {
    TriangleParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.upper        = upper;
    params.unitDiagonal = unitDiagonal;
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params),
                            triangleFunctionName(type), "triangle");
}

void launchMetalTranspose(BufferParam output, const size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, BufferParam input,
                          const size_t inputBytes,
                          const af::dim4& inputStrides, const bool conjugate,
                          const af_dtype type) {
    TransposeParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.conjugate = conjugate;
    launchSingleInputKernel(
        output, outputBytes, outputDims, input, inputBytes, &params,
        sizeof(params), transposeFunctionName(type, false), "transpose");
}

void launchMetalTransposeInplace(BufferParam input,
                                 const size_t inputBytes,
                                 const af::dim4& dims,
                                 const af::dim4& strides,
                                 const bool conjugate, const af_dtype type) {
    TransposeInplaceParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]    = static_cast<uint64_t>(dims[i]);
        params.strides[i] = static_cast<uint64_t>(strides[i]);
    }
    params.conjugate = conjugate;
    launchOutputKernel(input, inputBytes, static_cast<size_t>(dims.elements()),
                       &params, sizeof(params),
                       transposeFunctionName(type, true), "inplace transpose");
}

void launchMetalUnwrap(BufferParam output, const size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, BufferParam input,
                       const size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, const dim_t windowX,
                       const dim_t windowY, const dim_t strideX,
                       const dim_t strideY, const dim_t paddingX,
                       const dim_t paddingY, const dim_t dilationX,
                       const dim_t dilationY,
                       const unsigned columnDimension,
                       const af_dtype type) {
    UnwrapParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.windowX         = static_cast<uint64_t>(windowX);
    params.windowY         = static_cast<uint64_t>(windowY);
    params.strideX         = static_cast<uint64_t>(strideX);
    params.strideY         = static_cast<uint64_t>(strideY);
    params.paddingX        = static_cast<int64_t>(paddingX);
    params.paddingY        = static_cast<int64_t>(paddingY);
    params.dilationX       = static_cast<uint64_t>(dilationX);
    params.dilationY       = static_cast<uint64_t>(dilationY);
    params.columnDimension = columnDimension;
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params),
                            unwrapFunctionName(type), "unwrap");
}

void launchMetalWrap(BufferParam output, const size_t outputBytes,
                     const af::dim4& outputDims,
                     const af::dim4& outputStrides, BufferParam input,
                     const size_t inputBytes, const af::dim4& inputDims,
                     const af::dim4& inputStrides, const dim_t windowX,
                     const dim_t windowY, const dim_t strideX,
                     const dim_t strideY, const dim_t paddingX,
                     const dim_t paddingY, const dim_t dilationX,
                     const dim_t dilationY,
                     const unsigned columnDimension,
                     const af_dtype type) {
    WrapParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.windowX         = static_cast<uint64_t>(windowX);
    params.windowY         = static_cast<uint64_t>(windowY);
    params.strideX         = static_cast<uint64_t>(strideX);
    params.strideY         = static_cast<uint64_t>(strideY);
    params.paddingX        = static_cast<int64_t>(paddingX);
    params.paddingY        = static_cast<int64_t>(paddingY);
    params.dilationX       = static_cast<uint64_t>(dilationX);
    params.dilationY       = static_cast<uint64_t>(dilationY);
    params.columnDimension = columnDimension;
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params), wrapFunctionName(type),
                            "wrap");
}

void launchMetalGradient(BufferParam gradient0,
                         const af::dim4& gradient0Strides,
                         BufferParam gradient1,
                         const af::dim4& gradient1Strides,
                         const size_t outputBytes, BufferParam input,
                         const size_t inputBytes,
                         const af::dim4& inputDims,
                         const af::dim4& inputStrides,
                         const af_dtype type) {
    GradientParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]             = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]     = static_cast<uint64_t>(inputStrides[i]);
        params.gradient0Strides[i] =
            static_cast<uint64_t>(gradient0Strides[i]);
        params.gradient1Strides[i] =
            static_cast<uint64_t>(gradient1Strides[i]);
    }
    launchInputTwoOutputKernel(
        gradient0, gradient1, outputBytes, input, inputBytes,
        static_cast<size_t>(inputDims.elements()), &params, sizeof(params),
        gradientFunctionName(type), "gradient");
}

void launchMetalSobel(BufferParam derivative0,
                      const af::dim4& derivative0Strides,
                      BufferParam derivative1,
                      const af::dim4& derivative1Strides,
                      const size_t outputBytes, BufferParam input,
                      const size_t inputBytes, const af::dim4& inputDims,
                      const af::dim4& inputStrides,
                      const af_dtype inputType) {
    SobelParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]         = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i] = static_cast<uint64_t>(inputStrides[i]);
        params.derivative0Strides[i] =
            static_cast<uint64_t>(derivative0Strides[i]);
        params.derivative1Strides[i] =
            static_cast<uint64_t>(derivative1Strides[i]);
    }
    launchInputTwoOutputKernel(
        derivative0, derivative1, outputBytes, input, inputBytes,
        static_cast<size_t>(inputDims.elements()), &params, sizeof(params),
        sobelFunctionName(inputType), "Sobel");
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
