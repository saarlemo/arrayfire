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
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace arrayfire {
namespace metal {
namespace kernel {

namespace {

const char* convolveFunctionName(const af_dtype inputType,
                                 const af_dtype filterType,
                                 const bool separable) {
    if (filterType == f32) {
        switch (inputType) {
            case f32: return separable ? "separable_convolve_float"
                                       : "convolve_float";
            case s32: return separable ? "separable_convolve_int_float"
                                       : "convolve_int_float";
            case u32: return separable ? "separable_convolve_uint_float"
                                       : "convolve_uint_float";
            case s64: return separable ? "separable_convolve_long_float"
                                       : "convolve_long_float";
            case u64: return separable ? "separable_convolve_ulong_float"
                                       : "convolve_ulong_float";
            case s8: return separable ? "separable_convolve_char_float"
                                      : "convolve_char_float";
            case u8:
            case b8: return separable ? "separable_convolve_uchar_float"
                                      : "convolve_uchar_float";
            case s16: return separable ? "separable_convolve_short_float"
                                       : "convolve_short_float";
            case u16: return separable ? "separable_convolve_ushort_float"
                                       : "convolve_ushort_float";
            default: return nullptr;
        }
    }
    if (inputType == c32 && filterType == c32) {
        return separable ? "separable_convolve_cfloat"
                         : "convolve_cfloat";
    }
    return nullptr;
}

const char* convolveNNFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "convolve_nn_float";
        case f16: return "convolve_nn_half";
        default: return nullptr;
    }
}

void setConvolveNNParams(ConvolveNNParams& p,
                         const af::dim4& outputDims,
                         const af::dim4& outputStrides,
                         const af::dim4& signalDims,
                         const af::dim4& signalStrides,
                         const af::dim4& filterDims,
                         const af::dim4& filterStrides,
                         const af::dim4& gradientDims,
                         const af::dim4& gradientStrides,
                         const af::dim4& stride,
                         const af::dim4& padding,
                         const af::dim4& dilation) {
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]     = static_cast<uint64_t>(outputDims[i]);
        p.outputStrides[i]  = static_cast<uint64_t>(outputStrides[i]);
        p.signalDims[i]     = static_cast<uint64_t>(signalDims[i]);
        p.signalStrides[i]  = static_cast<uint64_t>(signalStrides[i]);
        p.filterDims[i]     = static_cast<uint64_t>(filterDims[i]);
        p.filterStrides[i]  = static_cast<uint64_t>(filterStrides[i]);
        p.gradientDims[i]   = static_cast<uint64_t>(gradientDims[i]);
        p.gradientStrides[i] = static_cast<uint64_t>(gradientStrides[i]);
    }
    for (int i = 0; i < 2; ++i) {
        p.stride[i]   = static_cast<uint64_t>(stride[i]);
        p.padding[i]  = static_cast<int64_t>(padding[i]);
        p.dilation[i] = static_cast<uint64_t>(dilation[i]);
    }
}

void launchConvolveNNKernel(const std::string& functionName,
                            BufferParam input0, BufferParam input1,
                            BufferParam output, const ConvolveNNParams& p,
                            const af::dim4& outputDims) {
    if (!output.buffer || !input0.buffer || !input1.buffer) {
        AF_ERROR("Could not allocate Metal convolution buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal NN convolution command encoder",
                 AF_ERR_RUNTIME);
    }

    auto* pipeline = metalPipeline(functionName.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(input0.buffer, input0.offset, 0);
    encoder->setBuffer(input1.buffer, input1.offset, 1);
    encoder->setBuffer(output.buffer, output.offset, 2);
    encoder->setBytes(&p, sizeof(p), 3);
    const auto width = std::min<NS::UInteger>(
        256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(
        MTL::Size(static_cast<NS::UInteger>(outputDims.elements()), 1, 1),
        MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
}

}  // namespace

void launchMetalGemm(
    BufferParam output, const size_t outputBytes,
    const af::dim4& outputDims, const af::dim4& outputStrides, BufferParam lhs,
    const size_t lhsBytes, const af::dim4& lhsDims,
    const af::dim4& lhsStrides, BufferParam rhs, const size_t rhsBytes,
    const af::dim4& rhsDims, const af::dim4& rhsStrides,
    const af_mat_prop lhsOption, const af_mat_prop rhsOption,
    const float alphaReal, const float alphaImag, const float betaReal,
    const float betaImag, const af_dtype type) {
    GemmParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        p.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        p.lhsDims[i]       = static_cast<uint64_t>(lhsDims[i]);
        p.lhsStrides[i]    = static_cast<int64_t>(lhsStrides[i]);
        p.rhsDims[i]       = static_cast<uint64_t>(rhsDims[i]);
        p.rhsStrides[i]    = static_cast<int64_t>(rhsStrides[i]);
    }
    p.transposeLhs  = lhsOption != AF_MAT_NONE;
    p.conjugateLhs  = lhsOption == AF_MAT_CTRANS;
    p.transposeRhs  = rhsOption != AF_MAT_NONE;
    p.conjugateRhs  = rhsOption == AF_MAT_CTRANS;
    const int lhsRowDim = p.transposeLhs ? 1 : 0;
    const int lhsColDim = p.transposeLhs ? 0 : 1;
    const int rhsColDim = p.transposeRhs ? 0 : 1;
    p.m              = static_cast<uint32_t>(lhsDims[lhsRowDim]);
    p.n              = static_cast<uint32_t>(rhsDims[rhsColDim]);
    p.k              = static_cast<uint32_t>(lhsDims[lhsColDim]);
    p.alphaReal      = alphaReal;
    p.alphaImag      = alphaImag;
    p.betaReal       = betaReal;
    p.betaImag       = betaImag;
    if (outputDims.elements() == 0) return;
    UNUSED(outputBytes);
    UNUSED(lhsBytes);
    UNUSED(rhsBytes);
    if (!output.buffer || !lhs.buffer || !rhs.buffer) {
        AF_ERROR("Could not allocate Metal GEMM buffers", AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal GEMM command encoder",
                 AF_ERR_RUNTIME);
    }

    constexpr NS::UInteger tileM = 16;
    constexpr NS::UInteger tileN = 16;
    constexpr size_t tileElements = tileM * tileN;
    const size_t elementBytes = type == c32 ? 2 * sizeof(float) : sizeof(float);
    auto* pipeline = metalPipeline(gemmFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(lhs.buffer, lhs.offset, 0);
    encoder->setBuffer(rhs.buffer, rhs.offset, 1);
    encoder->setBuffer(output.buffer, output.offset, 2);
    encoder->setBytes(&p, sizeof(p), 3);
    encoder->setThreadgroupMemoryLength(tileElements * elementBytes, 0);
    encoder->setThreadgroupMemoryLength(tileElements * elementBytes, 1);
    const NS::UInteger groupsX =
        static_cast<NS::UInteger>(p.m / tileM + (p.m % tileM != 0));
    const NS::UInteger groupsY =
        static_cast<NS::UInteger>(p.n / tileN + (p.n % tileN != 0));
    const NS::UInteger groupsZ =
        static_cast<NS::UInteger>(outputDims[2]) *
        static_cast<NS::UInteger>(outputDims[3]);
    encoder->dispatchThreadgroups(MTL::Size(groupsX, groupsY, groupsZ),
                                  MTL::Size(tileM, tileN, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            metalErrorDescription(commandBuffer->error(),
                                  "Metal GEMM dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

bool supportsMetalIndex(const af_dtype type) noexcept {
    return indexFunctionName(type) != nullptr;
}

void launchMetalIndex(
    BufferParam output, const size_t outputBytes,
    const af::dim4& outputDims, const af::dim4& outputStrides,
    const dim_t outputOffset, BufferParam input, const size_t inputBytes,
    const af::dim4& inputDims, const af::dim4& inputStrides,
    const dim_t inputOffset, const af::dim4& offsets,
    const std::vector<af_seq>& sequences,
    const std::array<bool, 4>& isSequence,
    const std::array<BufferParam, 4>& indexBuffers,
    const std::array<dim_t, 4>& indexOffsets,
    const std::array<dim_t, 4>& indexStrides, const af_dtype type) {
    IndexParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<int64_t>(inputStrides[i]);
        params.offsets[i]       = static_cast<int64_t>(offsets[i]);
        params.steps[i]         = static_cast<int64_t>(sequences[i].step);
        params.indexOffsets[i]  = static_cast<int64_t>(indexOffsets[i]);
        params.indexStrides[i]  = static_cast<int64_t>(indexStrides[i]);
        params.isSequence[i]    = isSequence[i];
    }
    params.outputOffset = static_cast<int64_t>(outputOffset);
    params.inputOffset  = static_cast<int64_t>(inputOffset);
    launchFiveInputKernel(
        output, outputBytes, input, inputBytes, indexBuffers[0], sizeof(uint),
        indexBuffers[1], sizeof(uint), indexBuffers[2], sizeof(uint),
        indexBuffers[3], sizeof(uint), &params, sizeof(params),
        static_cast<size_t>(outputDims.elements()), indexFunctionName(type),
        "index");
}

bool supportsMetalAssign(const af_dtype type) noexcept {
    return assignFunctionName(type) != nullptr;
}

void launchMetalAssign(
    BufferParam output, const size_t outputBytes,
    const af::dim4& outputDims, const af::dim4& destinationStrides,
    const dim_t outputOffset, BufferParam rhs, const size_t rhsBytes,
    const af::dim4& rhsDims, const af::dim4& rhsStrides,
    const dim_t rhsOffset, const af::dim4& offsets,
    const std::array<bool, 4>& isSequence,
    const std::array<BufferParam, 4>& indexBuffers,
    const std::array<dim_t, 4>& indexOffsets,
    const std::array<dim_t, 4>& indexStrides, const af_dtype type) {
    AssignParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]        = static_cast<uint64_t>(outputDims[i]);
        params.destinationStrides[i] =
            static_cast<int64_t>(destinationStrides[i]);
        params.rhsDims[i]      = static_cast<uint64_t>(rhsDims[i]);
        params.rhsStrides[i]   = static_cast<int64_t>(rhsStrides[i]);
        params.offsets[i]      = static_cast<int64_t>(offsets[i]);
        params.indexOffsets[i] = static_cast<int64_t>(indexOffsets[i]);
        params.indexStrides[i] = static_cast<int64_t>(indexStrides[i]);
        params.isSequence[i]   = isSequence[i];
    }
    params.outputOffset = static_cast<int64_t>(outputOffset);
    params.rhsOffset    = static_cast<int64_t>(rhsOffset);
    launchFiveInputKernel(
        output, outputBytes, rhs, rhsBytes, indexBuffers[0], sizeof(uint),
        indexBuffers[1], sizeof(uint), indexBuffers[2], sizeof(uint),
        indexBuffers[3], sizeof(uint), &params, sizeof(params),
        static_cast<size_t>(rhsDims.elements()), assignFunctionName(type),
        "assign");
}

bool supportsMetalTransform(const af_dtype type,
                            const af_interp_type method) noexcept {
    return transformFunctionName(type) != nullptr &&
           (method == AF_INTERP_NEAREST || method == AF_INTERP_LOWER ||
            method == AF_INTERP_BILINEAR ||
            method == AF_INTERP_BILINEAR_COSINE ||
            method == AF_INTERP_BICUBIC ||
            method == AF_INTERP_BICUBIC_SPLINE);
}

void launchMetalTransform(BufferParam output, const size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides,
                          const dim_t outputOffset, BufferParam input,
                          const size_t inputBytes,
                          const af::dim4& inputDims,
                          const af::dim4& inputStrides,
                          const dim_t inputOffset, BufferParam transform,
                          const size_t transformBytes,
                          const af::dim4& transformDims,
                          const af::dim4& transformStrides,
                          const dim_t transformOffset,
                          const af_interp_type method, const bool inverse,
                          const bool perspective, const af_dtype type) {
    TransformParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<int64_t>(inputStrides[i]);
        params.transformDims[i] = static_cast<uint64_t>(transformDims[i]);
        params.transformStrides[i] =
            static_cast<int64_t>(transformStrides[i]);
    }
    params.outputOffset    = static_cast<int64_t>(outputOffset);
    params.inputOffset     = static_cast<int64_t>(inputOffset);
    params.transformOffset = static_cast<int64_t>(transformOffset);
    params.method      = static_cast<uint32_t>(method);
    params.inverse     = inverse;
    params.perspective = perspective;
    launchTwoInputKernel(output, outputBytes, input, inputBytes, transform,
                         transformBytes, &params, sizeof(params),
                         static_cast<size_t>(outputDims.elements()),
                         transformFunctionName(type), "transform");
}

bool supportsMetalSort(const af_dtype type) noexcept {
    return sortFunctionName(type) != nullptr;
}

void launchMetalSort(BufferParam inout, const size_t bytes,
                     const af::dim4& dims,
                     const af::dim4& strides, const int dimension,
                     const bool ascending, const af_dtype type) {
    SortParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]    = static_cast<uint64_t>(dims[i]);
        params.strides[i] = static_cast<uint64_t>(strides[i]);
    }
    params.ascending = ascending;
    params.dimension = static_cast<uint32_t>(dimension);
    launchSingleInputKernel(inout, bytes, dims, inout, bytes, &params,
                            sizeof(params), sortFunctionName(type), "sort");
}

bool supportsMetalSortByKey(const af_dtype keyType,
                            const af_dtype valueType) noexcept {
    return sortByKeyFunctionName(keyType, valueType) != nullptr;
}

bool supportsMetalSortIndex(const af_dtype type) noexcept {
    return sortIndexFunctionName(type) != nullptr;
}

namespace {

const char* topKFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "topk_float";
        case s32: return "topk_int";
        case u32: return "topk_uint";
        case s64: return "topk_long";
        case u64: return "topk_ulong";
        case f16: return "topk_half";
        default: return nullptr;
    }
}

}  // namespace

bool supportsMetalTopK(const af_dtype type) noexcept {
    return topKFunctionName(type) != nullptr;
}

void launchMetalTopK(BufferParam values, BufferParam indices,
                     const af::dim4& outputStrides, BufferParam input,
                     const af::dim4& inputDims,
                     const af::dim4& inputStrides, const int k,
                     const bool ascending, const af_dtype type) {
    if (!values.buffer || !indices.buffer || !input.buffer) {
        AF_ERROR("Could not allocate Metal top-k buffers", AF_ERR_NO_MEM);
    }

    TopKParams params{};
    for (int i = 0; i < 4; ++i) {
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
    }
    params.k         = static_cast<uint64_t>(k);
    params.ascending = ascending;

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }
    auto* pipeline = metalPipeline(topKFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(values.buffer, values.offset, 1);
    encoder->setBuffer(indices.buffer, indices.offset, 2);
    encoder->setBytes(&params, sizeof(params), 3);

    const size_t vectors =
        static_cast<size_t>(inputDims[1] * inputDims[2] * inputDims[3]);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(vectors, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
}

void launchMetalSortByKey(BufferParam keys, const size_t keyBytes,
                          const af::dim4& keyDims, const af::dim4& keyStrides,
                          BufferParam values, const size_t valueBytes,
                          const af::dim4& valueStrides, const int dimension,
                          const bool ascending, const af_dtype keyType,
                          const af_dtype valueType) {
    SortByKeyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]         = static_cast<uint64_t>(keyDims[i]);
        params.keyStrides[i]   = static_cast<uint64_t>(keyStrides[i]);
        params.valueStrides[i] = static_cast<uint64_t>(valueStrides[i]);
    }
    params.ascending = ascending;
    params.dimension = static_cast<uint32_t>(dimension);

    UNUSED(keyBytes);
    UNUSED(valueBytes);
    if (!keys.buffer || !values.buffer)
        AF_ERROR("Could not allocate Metal sort-by-key buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = metalPipeline(
        sortByKeyFunctionName(keyType, valueType));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(keys.buffer, keys.offset, 0);
    encoder->setBuffer(values.buffer, values.offset, 1);
    encoder->setBuffer(keys.buffer, keys.offset, 2);
    encoder->setBuffer(values.buffer, values.offset, 3);
    encoder->setBytes(&params, sizeof(params), 4);
    const size_t vectors = static_cast<size_t>(keyDims.elements() /
                                               keyDims[dimension]);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(vectors, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal sort-by-key dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchMetalSortIndex(BufferParam keys, const size_t keyBytes,
                          const af::dim4& keyDims,
                          const af::dim4& keyStrides, BufferParam values,
                          const af::dim4& valueStrides, const int dimension,
                          const bool ascending, const af_dtype keyType) {
    SortByKeyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]         = static_cast<uint64_t>(keyDims[i]);
        params.keyStrides[i]   = static_cast<uint64_t>(keyStrides[i]);
        params.valueStrides[i] = static_cast<uint64_t>(valueStrides[i]);
    }
    params.ascending = ascending;
    params.dimension = static_cast<uint32_t>(dimension);

    UNUSED(keyBytes);
    if (!keys.buffer || !values.buffer) {
        AF_ERROR("Could not allocate Metal sort-index buffers", AF_ERR_NO_MEM);
    }
    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }
    auto* pipeline = metalPipeline(sortIndexFunctionName(keyType));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(keys.buffer, keys.offset, 0);
    encoder->setBuffer(values.buffer, values.offset, 1);
    encoder->setBuffer(keys.buffer, keys.offset, 2);
    encoder->setBuffer(values.buffer, values.offset, 3);
    encoder->setBytes(&params, sizeof(params), 4);
    const size_t vectors = static_cast<size_t>(keyDims.elements() /
                                               keyDims[dimension]);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(vectors, 1, 1), MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
}

namespace {

const char* sparseMatmulFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "sparse_matmul_float";
        case c32: return "sparse_matmul_cfloat";
        default: return nullptr;
    }
}

}  // namespace

bool supportsMetalSparseMatmul(const af_dtype type) noexcept {
    return sparseMatmulFunctionName(type) != nullptr;
}

void launchMetalSparseMatmul(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam values, BufferParam rowIndex,
    BufferParam columnIndex, BufferParam rhs, const af::dim4& rhsStrides,
    const af::dim4& sparseDims, const af_mat_prop operation,
    const af_dtype type) {
    if (!output.buffer || !values.buffer || !rowIndex.buffer ||
        !columnIndex.buffer || !rhs.buffer) {
        AF_ERROR("Could not allocate Metal sparse matmul buffers",
                 AF_ERR_NO_MEM);
    }

    SparseMatmulParams params{};
    params.sourceRows    = static_cast<uint64_t>(sparseDims[0]);
    params.sourceColumns = static_cast<uint64_t>(sparseDims[1]);
    params.outputRows    = static_cast<uint64_t>(outputDims[0]);
    params.rhsColumns    = static_cast<uint64_t>(outputDims[1]);
    for (int i = 0; i < 4; ++i) {
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.rhsStrides[i]    = static_cast<uint64_t>(rhsStrides[i]);
    }
    params.operation = operation == AF_MAT_NONE
                           ? 0
                           : operation == AF_MAT_TRANS ? 1 : 2;

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }
    auto* pipeline = metalPipeline(sparseMatmulFunctionName(type));
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(output.buffer, output.offset, 0);
    encoder->setBuffer(values.buffer, values.offset, 1);
    encoder->setBuffer(rowIndex.buffer, rowIndex.offset, 2);
    encoder->setBuffer(columnIndex.buffer, columnIndex.offset, 3);
    encoder->setBuffer(rhs.buffer, rhs.offset, 4);
    encoder->setBytes(&params, sizeof(params), 5);

    constexpr NS::UInteger width = 8;
    encoder->dispatchThreads(
        MTL::Size(static_cast<NS::UInteger>(outputDims[0]),
                  static_cast<NS::UInteger>(outputDims[1]), 1),
        MTL::Size(width, width, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
}

bool supportsMetalScan(const af_dtype inputType,
                       const af_dtype outputType) noexcept {
    return scanFunctionName(inputType, outputType) != nullptr;
}

void launchMetalScan(BufferParam output, const size_t outputBytes,
                     const af::dim4& dims, const af::dim4& outputStrides,
                     BufferParam input, const size_t inputBytes,
                     const af::dim4& inputStrides, const int dimension,
                     const af_dtype inputType, const af_dtype outputType,
                     const uint32_t operation, const bool inclusive) {
    ScanParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<int64_t>(inputStrides[i]);
    }
    params.dimension = static_cast<uint32_t>(dimension);
    params.operation = operation;
    params.inclusive = inclusive;

    constexpr size_t tileLength = 4096;
    constexpr size_t threadgroupSize = 256;
    const size_t outputCount = static_cast<size_t>(dims.elements());
    const size_t scanLength = static_cast<size_t>(dims[dimension]);
    if (outputCount == 0) return;
    UNUSED(inputBytes);

    const char* baseName = scanFunctionName(inputType, outputType);
    if (!baseName)
        AF_ERROR("Scan type is not supported by Metal", AF_ERR_NOT_SUPPORTED);
    const std::string tileName = std::string(baseName) + "_tile";
    const std::string totalsName =
        std::string(baseName) + "_totals_scan";
    const std::string applyName = std::string(baseName) + "_apply";
    auto* tilePipeline = metalPipeline(tileName.c_str());
    auto* totalsPipeline = metalPipeline(totalsName.c_str());
    auto* applyPipeline = metalPipeline(applyName.c_str());
    if (scanLength >
        std::numeric_limits<size_t>::max() - (tileLength - 1))
        AF_ERROR("Metal scan tile count overflow", AF_ERR_SIZE);
    const size_t tileCount = (scanLength + tileLength - 1) / tileLength;
    constexpr size_t totalsPerBlock = 4096;
    if (tileCount >
        std::numeric_limits<size_t>::max() - (totalsPerBlock - 1))
        AF_ERROR("Metal scan totals block count overflow", AF_ERR_SIZE);
    const size_t blockCount =
        (tileCount + totalsPerBlock - 1) / totalsPerBlock;
    const size_t lineCount = outputCount / scanLength;
    if (lineCount > std::numeric_limits<size_t>::max() / tileCount)
        AF_ERROR("Metal scan scratch size overflow", AF_ERR_SIZE);
    const size_t partialCount = lineCount * tileCount;
    if (outputBytes == 0 || outputBytes % outputCount != 0)
        AF_ERROR("Metal scan output size is invalid", AF_ERR_SIZE);
    const size_t elementBytes = outputBytes / outputCount;
    if (partialCount > std::numeric_limits<size_t>::max() / elementBytes)
        AF_ERROR("Metal scan scratch size overflow", AF_ERR_SIZE);
    if (lineCount > std::numeric_limits<size_t>::max() / blockCount)
        AF_ERROR("Metal scan block scratch size overflow", AF_ERR_SIZE);
    const size_t blockPartialCount = lineCount * blockCount;
    if (blockPartialCount >
        std::numeric_limits<size_t>::max() / elementBytes)
        AF_ERROR("Metal scan block scratch size overflow", AF_ERR_SIZE);
    if (tileCount > std::numeric_limits<uint32_t>::max() ||
        blockCount > std::numeric_limits<uint32_t>::max() ||
        lineCount > std::numeric_limits<uint32_t>::max())
        AF_ERROR("Metal scan dispatch exceeds the supported grid size",
                 AF_ERR_SIZE);
    const size_t partialBytes = partialCount * elementBytes;
    const size_t blockPartialBytes = blockPartialCount * elementBytes;
    auto tileTotals = NS::TransferPtr(metalDevice()->newBuffer(
        partialBytes, MTL::ResourceStorageModePrivate));
    auto tilePrefixes = NS::TransferPtr(metalDevice()->newBuffer(
        partialBytes, MTL::ResourceStorageModePrivate));
    auto blockTotals = NS::TransferPtr(metalDevice()->newBuffer(
        blockPartialBytes, MTL::ResourceStorageModePrivate));
    if (!tileTotals || !tilePrefixes || !blockTotals)
        AF_ERROR("Could not allocate Metal scan scratch", AF_ERR_NO_MEM);
    ScanTotalsParams totalsParams{tileCount, blockCount, operation};

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    encoder->setComputePipelineState(tilePipeline);
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(output.buffer, output.offset, 1);
    encoder->setBuffer(tileTotals.get(), 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    encoder->dispatchThreadgroups(MTL::Size(tileCount, lineCount, 1),
                                  MTL::Size(threadgroupSize, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(totalsPipeline);
    encoder->setBuffer(tileTotals.get(), 0, 0);
    encoder->setBuffer(tilePrefixes.get(), 0, 1);
    encoder->setBuffer(blockTotals.get(), 0, 2);
    encoder->setBytes(&totalsParams, sizeof(totalsParams), 3);
    encoder->dispatchThreadgroups(MTL::Size(blockCount, lineCount, 1),
                                  MTL::Size(threadgroupSize, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(applyPipeline);
    encoder->setBuffer(tilePrefixes.get(), 0, 0);
    encoder->setBuffer(blockTotals.get(), 0, 1);
    encoder->setBuffer(output.buffer, output.offset, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    encoder->setBytes(&totalsParams, sizeof(totalsParams), 4);
    encoder->dispatchThreadgroups(MTL::Size(tileCount, lineCount, 1),
                                  MTL::Size(threadgroupSize, 1, 1));
    encoder->endEncoding();

    auto totalsLifetime = std::shared_ptr<MTL::Buffer>(
        tileTotals.get(), [](MTL::Buffer* buffer) { buffer->release(); });
    auto prefixLifetime = std::shared_ptr<MTL::Buffer>(
        tilePrefixes.get(), [](MTL::Buffer* buffer) { buffer->release(); });
    auto blockLifetime = std::shared_ptr<MTL::Buffer>(
        blockTotals.get(), [](MTL::Buffer* buffer) { buffer->release(); });
    tileTotals->retain();
    tilePrefixes->retain();
    blockTotals->retain();
    commandBuffer->addCompletedHandler(
        [partials = std::move(totalsLifetime),
         prefixes = std::move(prefixLifetime),
         blocks = std::move(blockLifetime)](MTL::CommandBuffer*) {
            static_cast<void>(partials);
            static_cast<void>(prefixes);
            static_cast<void>(blocks);
        });
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            metalErrorDescription(commandBuffer->error(),
                                  "Metal scan dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

bool supportsMetalScanByKey(const af_dtype keyType,
                            const af_dtype valueType) noexcept {
    return scanByKeyFunctionName(keyType, valueType) != nullptr;
}

void launchMetalScanByKey(BufferParam output, const size_t outputBytes,
                          const af::dim4& dims, const af::dim4& outputStrides,
                          BufferParam keys, const size_t keyBytes,
                          const af::dim4& keyStrides, BufferParam input,
                          const size_t inputBytes, const af::dim4& inputStrides,
                          const int dimension, const af_dtype keyType,
                          const af_dtype valueType, const uint32_t operation,
                          const bool inclusive) {
    ScanByKeyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        params.keyStrides[i]    = static_cast<int64_t>(keyStrides[i]);
        params.inputStrides[i]  = static_cast<int64_t>(inputStrides[i]);
    }
    params.dimension = static_cast<uint32_t>(dimension);
    params.operation = operation;
    params.inclusive = inclusive;

    constexpr size_t tileLength = 4096;
    constexpr size_t threadgroupSize = 256;
    const size_t outputCount = static_cast<size_t>(dims.elements());
    const size_t scanLength = static_cast<size_t>(dims[dimension]);
    if (outputCount == 0) return;
    UNUSED(outputBytes);

    const char* functionName = scanByKeyFunctionName(keyType, valueType);
    if (!functionName)
        AF_ERROR("Scan-by-key type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    const std::string tileName =
        std::string(functionName) + "_tile_scan";
    const std::string totalsName =
        std::string(functionName) + "_totals_scan";
    const std::string applyName =
        std::string(functionName) + "_carry_apply";
    auto* tilePipeline = metalPipeline(tileName.c_str());
    auto* totalsPipeline = metalPipeline(totalsName.c_str());
    auto* applyPipeline = metalPipeline(applyName.c_str());
    if (scanLength >
        std::numeric_limits<size_t>::max() - (tileLength - 1))
        AF_ERROR("Metal scan-by-key tile count overflow", AF_ERR_SIZE);
    const size_t tileCount = (scanLength + tileLength - 1) / tileLength;
    constexpr size_t summariesPerBlock = 4096;
    if (tileCount >
        std::numeric_limits<size_t>::max() - (summariesPerBlock - 1))
        AF_ERROR("Metal scan-by-key summary block count overflow", AF_ERR_SIZE);
    const size_t blockCount =
        (tileCount + summariesPerBlock - 1) / summariesPerBlock;
    const size_t lineCount = outputCount / scanLength;
    if (lineCount > std::numeric_limits<size_t>::max() / tileCount)
        AF_ERROR("Metal scan-by-key scratch size overflow", AF_ERR_SIZE);
    const size_t partialCount = lineCount * tileCount;
    if (lineCount > std::numeric_limits<size_t>::max() / blockCount)
        AF_ERROR("Metal scan-by-key block scratch size overflow", AF_ERR_SIZE);
    const size_t blockPartialCount = lineCount * blockCount;
    if (partialCount > std::numeric_limits<size_t>::max() / keyBytes ||
        partialCount > std::numeric_limits<size_t>::max() / inputBytes ||
        partialCount > std::numeric_limits<size_t>::max() /
                           sizeof(uint32_t) ||
        blockPartialCount >
            std::numeric_limits<size_t>::max() / keyBytes ||
        blockPartialCount >
            std::numeric_limits<size_t>::max() / inputBytes ||
        blockPartialCount >
            std::numeric_limits<size_t>::max() / sizeof(uint32_t))
        AF_ERROR("Metal scan-by-key scratch size overflow", AF_ERR_SIZE);
    if (tileCount > std::numeric_limits<uint32_t>::max() ||
        blockCount > std::numeric_limits<uint32_t>::max() ||
        lineCount > std::numeric_limits<uint32_t>::max())
        AF_ERROR("Metal scan-by-key dispatch exceeds the supported grid size",
                 AF_ERR_SIZE);
    const size_t keyScratchBytes = partialCount * keyBytes;
    const size_t valueScratchBytes = partialCount * inputBytes;
    const size_t metadataBytes = partialCount * sizeof(uint32_t);
    const size_t blockKeyBytes = blockPartialCount * keyBytes;
    const size_t blockValueBytes = blockPartialCount * inputBytes;
    const size_t blockMetadataBytes = blockPartialCount * sizeof(uint32_t);
    auto firstKeys = NS::TransferPtr(metalDevice()->newBuffer(
        keyScratchBytes, MTL::ResourceStorageModePrivate));
    auto lastKeys = NS::TransferPtr(metalDevice()->newBuffer(
        keyScratchBytes, MTL::ResourceStorageModePrivate));
    auto trailingValues = NS::TransferPtr(metalDevice()->newBuffer(
        valueScratchBytes, MTL::ResourceStorageModePrivate));
    auto leadingLengths = NS::TransferPtr(metalDevice()->newBuffer(
        metadataBytes, MTL::ResourceStorageModePrivate));
    auto summaryFlags = NS::TransferPtr(metalDevice()->newBuffer(
        metadataBytes, MTL::ResourceStorageModePrivate));
    auto prefixFirstKeys = NS::TransferPtr(metalDevice()->newBuffer(
        keyScratchBytes, MTL::ResourceStorageModePrivate));
    auto prefixLastKeys = NS::TransferPtr(metalDevice()->newBuffer(
        keyScratchBytes, MTL::ResourceStorageModePrivate));
    auto prefixTrailingValues = NS::TransferPtr(metalDevice()->newBuffer(
        valueScratchBytes, MTL::ResourceStorageModePrivate));
    auto prefixLeadingLengths = NS::TransferPtr(metalDevice()->newBuffer(
        metadataBytes, MTL::ResourceStorageModePrivate));
    auto prefixFlags = NS::TransferPtr(metalDevice()->newBuffer(
        metadataBytes, MTL::ResourceStorageModePrivate));
    auto blockFirstKeys = NS::TransferPtr(metalDevice()->newBuffer(
        blockKeyBytes, MTL::ResourceStorageModePrivate));
    auto blockLastKeys = NS::TransferPtr(metalDevice()->newBuffer(
        blockKeyBytes, MTL::ResourceStorageModePrivate));
    auto blockTrailingValues = NS::TransferPtr(metalDevice()->newBuffer(
        blockValueBytes, MTL::ResourceStorageModePrivate));
    auto blockLeadingLengths = NS::TransferPtr(metalDevice()->newBuffer(
        blockMetadataBytes, MTL::ResourceStorageModePrivate));
    auto blockFlags = NS::TransferPtr(metalDevice()->newBuffer(
        blockMetadataBytes, MTL::ResourceStorageModePrivate));
    if (!firstKeys || !lastKeys || !trailingValues || !leadingLengths ||
        !summaryFlags || !prefixFirstKeys || !prefixLastKeys ||
        !prefixTrailingValues || !prefixLeadingLengths || !prefixFlags ||
        !blockFirstKeys || !blockLastKeys || !blockTrailingValues ||
        !blockLeadingLengths || !blockFlags)
        AF_ERROR("Could not allocate Metal scan-by-key scratch",
                 AF_ERR_NO_MEM);
    ScanByKeyTotalsParams totalsParams{tileCount, blockCount, operation};

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    encoder->setComputePipelineState(tilePipeline);
    encoder->setBuffer(keys.buffer, keys.offset, 0);
    encoder->setBuffer(input.buffer, input.offset, 1);
    encoder->setBuffer(output.buffer, output.offset, 2);
    encoder->setBuffer(firstKeys.get(), 0, 3);
    encoder->setBuffer(lastKeys.get(), 0, 4);
    encoder->setBuffer(trailingValues.get(), 0, 5);
    encoder->setBuffer(leadingLengths.get(), 0, 6);
    encoder->setBuffer(summaryFlags.get(), 0, 7);
    encoder->setBytes(&params, sizeof(params), 8);
    encoder->dispatchThreadgroups(MTL::Size(tileCount, lineCount, 1),
                                  MTL::Size(threadgroupSize, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    std::array<MTL::Buffer*, 5> originalSummaries{
        firstKeys.get(), lastKeys.get(), trailingValues.get(),
        leadingLengths.get(), summaryFlags.get()};
    std::array<MTL::Buffer*, 5> prefixSummaries{
        prefixFirstKeys.get(), prefixLastKeys.get(), prefixTrailingValues.get(),
        prefixLeadingLengths.get(), prefixFlags.get()};
    std::array<MTL::Buffer*, 5> blockSummaries{
        blockFirstKeys.get(), blockLastKeys.get(), blockTrailingValues.get(),
        blockLeadingLengths.get(), blockFlags.get()};

    encoder->setComputePipelineState(totalsPipeline);
    for (size_t i = 0; i < 5; ++i) {
        encoder->setBuffer(originalSummaries[i], 0, i);
        encoder->setBuffer(prefixSummaries[i], 0, i + 5);
        encoder->setBuffer(blockSummaries[i], 0, i + 10);
    }
    encoder->setBytes(&totalsParams, sizeof(totalsParams), 15);
    encoder->dispatchThreadgroups(MTL::Size(blockCount, lineCount, 1),
                                  MTL::Size(threadgroupSize, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    encoder->setComputePipelineState(applyPipeline);
    encoder->setBuffer(output.buffer, output.offset, 0);
    for (size_t i = 0; i < 5; ++i) {
        encoder->setBuffer(originalSummaries[i], 0, i + 1);
        encoder->setBuffer(prefixSummaries[i], 0, i + 6);
        encoder->setBuffer(blockSummaries[i], 0, i + 11);
    }
    encoder->setBytes(&totalsParams, sizeof(totalsParams), 16);
    encoder->setBytes(&params, sizeof(params), 17);
    encoder->dispatchThreadgroups(MTL::Size(tileCount, lineCount, 1),
                                  MTL::Size(threadgroupSize, 1, 1));
    encoder->endEncoding();

    std::array<MTL::Buffer*, 15> scratchBuffers{
        firstKeys.get(),          lastKeys.get(),          trailingValues.get(),
        leadingLengths.get(),     summaryFlags.get(),      prefixFirstKeys.get(),
        prefixLastKeys.get(),     prefixTrailingValues.get(),
        prefixLeadingLengths.get(), prefixFlags.get(),     blockFirstKeys.get(),
        blockLastKeys.get(),      blockTrailingValues.get(),
        blockLeadingLengths.get(), blockFlags.get()};
    std::array<std::shared_ptr<MTL::Buffer>, 15> scratchLifetimes;
    for (size_t i = 0; i < scratchBuffers.size(); ++i) {
        scratchBuffers[i]->retain();
        scratchLifetimes[i] = std::shared_ptr<MTL::Buffer>(
            scratchBuffers[i], [](MTL::Buffer* buffer) { buffer->release(); });
    }
    commandBuffer->addCompletedHandler(
        [lifetimes = std::move(scratchLifetimes)](MTL::CommandBuffer*) {
            static_cast<void>(lifetimes);
        });
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            metalErrorDescription(commandBuffer->error(),
                                  "Metal scan-by-key dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

bool supportsMetalMean(const af_dtype inputType,
                       const af_dtype outputType) noexcept {
    return meanFunctionName(inputType, outputType) != nullptr;
}

bool supportsMetalMeanWeighted(const af_dtype valueType,
                               const af_dtype weightType) noexcept {
    return meanWeightedFunctionName(valueType, weightType) != nullptr;
}

void launchMetalMean(BufferParam output, const size_t outputBytes,
                     const af::dim4& outputDims,
                     const af::dim4& outputStrides, BufferParam input,
                     const size_t inputBytes, const af::dim4& inputDims,
                     const af::dim4& inputStrides, const int dimension,
                     const bool reduceAll, const af_dtype inputType,
                     const af_dtype outputType) {
    MeanParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<int64_t>(inputStrides[i]);
    }
    params.dimension = static_cast<uint32_t>(dimension);
    params.reduceAll = reduceAll;
    const size_t outputCount = static_cast<size_t>(outputDims.elements());
    const size_t reductionLength = static_cast<size_t>(
        reduceAll ? inputDims.elements() : inputDims[dimension]);
    constexpr size_t chunkSize = 4096;
    constexpr size_t threadgroupSize = 256;
    if (outputCount == 0) return;
    UNUSED(outputBytes);
    UNUSED(inputBytes);

    const char* baseName = meanFunctionName(inputType, outputType);
    if (!baseName)
        AF_ERROR("Mean type is not supported by Metal", AF_ERR_NOT_SUPPORTED);
    const std::string partialName = std::string(baseName) + "_partial";
    const std::string finalName = std::string(baseName) + "_final";
    auto* partialPipeline = metalPipeline(partialName.c_str());
    auto* finalPipeline = metalPipeline(finalName.c_str());
    const size_t chunkCount =
        std::max<size_t>((reductionLength + chunkSize - 1) / chunkSize, 1);
    if (outputCount > std::numeric_limits<size_t>::max() / chunkCount)
        AF_ERROR("Metal mean scratch size overflow", AF_ERR_SIZE);
    const size_t partialCount = outputCount * chunkCount;
    if (partialCount > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
        AF_ERROR("Metal mean scratch size overflow", AF_ERR_SIZE);
    const size_t partialBytes = partialCount * sizeof(uint64_t);
    const size_t groupCount = partialCount;
    if (groupCount > std::numeric_limits<uint32_t>::max() ||
        outputCount > std::numeric_limits<uint32_t>::max())
        AF_ERROR("Metal mean dispatch exceeds the supported grid size",
                 AF_ERR_SIZE);
    auto partials = NS::TransferPtr(metalDevice()->newBuffer(
        partialBytes, MTL::ResourceStorageModePrivate));
    auto partialCorrections = NS::TransferPtr(metalDevice()->newBuffer(
        partialBytes, MTL::ResourceStorageModePrivate));
    if (!partials || !partialCorrections)
        AF_ERROR("Could not allocate Metal mean scratch", AF_ERR_NO_MEM);

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    encoder->setComputePipelineState(partialPipeline);
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(partials.get(), 0, 1);
    encoder->setBuffer(partialCorrections.get(), 0, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    encoder->dispatchThreadgroups(MTL::Size(groupCount, 1, 1),
                                  MTL::Size(threadgroupSize, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(finalPipeline);
    encoder->setBuffer(partials.get(), 0, 0);
    encoder->setBuffer(partialCorrections.get(), 0, 1);
    encoder->setBuffer(output.buffer, output.offset, 2);
    encoder->setBytes(&params, sizeof(params), 3);
    encoder->dispatchThreadgroups(MTL::Size(outputCount, 1, 1),
                                  MTL::Size(threadgroupSize, 1, 1));
    encoder->endEncoding();

    auto partialLifetime = std::shared_ptr<MTL::Buffer>(
        partials.get(), [](MTL::Buffer* buffer) { buffer->release(); });
    auto correctionLifetime = std::shared_ptr<MTL::Buffer>(
        partialCorrections.get(),
        [](MTL::Buffer* buffer) { buffer->release(); });
    partials->retain();
    partialCorrections->retain();
    commandBuffer->addCompletedHandler(
        [values = std::move(partialLifetime),
         corrections = std::move(correctionLifetime)](MTL::CommandBuffer*) {
            static_cast<void>(values);
            static_cast<void>(corrections);
        });
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal mean dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchMetalMeanWeighted(
    BufferParam output, const size_t outputBytes,
    const af::dim4& outputDims, const af::dim4& outputStrides,
    BufferParam input, const size_t inputBytes,
    const af::dim4& inputDims, const af::dim4& inputStrides,
    BufferParam weights, const size_t weightBytes,
    const af::dim4& weightStrides, const int dimension,
    const bool reduceAll, const af_dtype valueType,
    const af_dtype weightType) {
    MeanParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<int64_t>(inputStrides[i]);
        params.weightStrides[i] = static_cast<int64_t>(weightStrides[i]);
    }
    params.dimension = static_cast<uint32_t>(dimension);
    params.reduceAll = reduceAll;
    const size_t outputCount = static_cast<size_t>(outputDims.elements());
    const size_t reductionLength = static_cast<size_t>(
        reduceAll ? inputDims.elements() : inputDims[dimension]);
    constexpr size_t chunkSize = 4096;
    constexpr size_t threadgroupSize = 256;
    if (outputCount == 0) return;
    UNUSED(outputBytes);
    UNUSED(inputBytes);
    UNUSED(weightBytes);

    const char* baseName = meanWeightedFunctionName(valueType, weightType);
    if (!baseName)
        AF_ERROR("Weighted mean type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    const std::string partialName = std::string(baseName) + "_partial";
    const std::string finalName = std::string(baseName) + "_final";
    auto* partialPipeline = metalPipeline(partialName.c_str());
    auto* finalPipeline = metalPipeline(finalName.c_str());
    const size_t chunkCount =
        std::max<size_t>((reductionLength + chunkSize - 1) / chunkSize, 1);
    if (outputCount > std::numeric_limits<size_t>::max() / chunkCount)
        AF_ERROR("Metal weighted-mean scratch size overflow", AF_ERR_SIZE);
    const size_t partialCount = outputCount * chunkCount;
    if (partialCount > std::numeric_limits<size_t>::max() / sizeof(uint64_t) ||
        partialCount > std::numeric_limits<size_t>::max() / sizeof(float))
        AF_ERROR("Metal weighted-mean scratch size overflow", AF_ERR_SIZE);
    const size_t groupCount = partialCount;
    if (groupCount > std::numeric_limits<uint32_t>::max() ||
        outputCount > std::numeric_limits<uint32_t>::max())
        AF_ERROR("Metal weighted-mean dispatch exceeds the supported grid size",
                 AF_ERR_SIZE);
    auto partialValues = NS::TransferPtr(metalDevice()->newBuffer(
        partialCount * sizeof(uint64_t), MTL::ResourceStorageModePrivate));
    auto partialValueCorrections = NS::TransferPtr(metalDevice()->newBuffer(
        partialCount * sizeof(uint64_t), MTL::ResourceStorageModePrivate));
    auto partialWeights = NS::TransferPtr(metalDevice()->newBuffer(
        partialCount * sizeof(float), MTL::ResourceStorageModePrivate));
    auto partialWeightCorrections = NS::TransferPtr(metalDevice()->newBuffer(
        partialCount * sizeof(float), MTL::ResourceStorageModePrivate));
    if (!partialValues || !partialValueCorrections || !partialWeights ||
        !partialWeightCorrections)
        AF_ERROR("Could not allocate Metal weighted-mean scratch",
                 AF_ERR_NO_MEM);

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    encoder->setComputePipelineState(partialPipeline);
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(weights.buffer, weights.offset, 1);
    encoder->setBuffer(partialValues.get(), 0, 2);
    encoder->setBuffer(partialValueCorrections.get(), 0, 3);
    encoder->setBuffer(partialWeights.get(), 0, 4);
    encoder->setBuffer(partialWeightCorrections.get(), 0, 5);
    encoder->setBytes(&params, sizeof(params), 6);
    encoder->dispatchThreadgroups(MTL::Size(groupCount, 1, 1),
                                  MTL::Size(threadgroupSize, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(finalPipeline);
    encoder->setBuffer(partialValues.get(), 0, 0);
    encoder->setBuffer(partialValueCorrections.get(), 0, 1);
    encoder->setBuffer(partialWeights.get(), 0, 2);
    encoder->setBuffer(partialWeightCorrections.get(), 0, 3);
    encoder->setBuffer(output.buffer, output.offset, 4);
    encoder->setBytes(&params, sizeof(params), 5);
    encoder->dispatchThreadgroups(MTL::Size(outputCount, 1, 1),
                                  MTL::Size(threadgroupSize, 1, 1));
    encoder->endEncoding();

    auto valuesLifetime = std::shared_ptr<MTL::Buffer>(
        partialValues.get(), [](MTL::Buffer* buffer) { buffer->release(); });
    auto weightsLifetime = std::shared_ptr<MTL::Buffer>(
        partialWeights.get(), [](MTL::Buffer* buffer) { buffer->release(); });
    auto valueCorrectionsLifetime = std::shared_ptr<MTL::Buffer>(
        partialValueCorrections.get(),
        [](MTL::Buffer* buffer) { buffer->release(); });
    auto weightCorrectionsLifetime = std::shared_ptr<MTL::Buffer>(
        partialWeightCorrections.get(),
        [](MTL::Buffer* buffer) { buffer->release(); });
    partialValues->retain();
    partialValueCorrections->retain();
    partialWeights->retain();
    partialWeightCorrections->retain();
    commandBuffer->addCompletedHandler(
        [values = std::move(valuesLifetime),
         weights = std::move(weightsLifetime),
         valueCorrections = std::move(valueCorrectionsLifetime),
         weightCorrections = std::move(weightCorrectionsLifetime)](
            MTL::CommandBuffer*) {
            static_cast<void>(values);
            static_cast<void>(weights);
            static_cast<void>(valueCorrections);
            static_cast<void>(weightCorrections);
        });
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal weighted-mean dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

bool supportsMetalApprox(const af_dtype valueType,
                         const af_dtype positionType) noexcept {
    return positionType == f32 && (valueType == f32 || valueType == c32);
}

void launchMetalApprox1(
    BufferParam output, const size_t outputBytes,
    const af::dim4& outputDims, const af::dim4& outputStrides,
    const dim_t outputOffset, BufferParam input, const size_t inputBytes,
    const af::dim4& inputDims, const af::dim4& inputStrides,
    const dim_t inputOffset, BufferParam positions,
    const size_t positionBytes, const af::dim4& positionDims,
    const af::dim4& positionStrides, const dim_t positionOffset,
    const int dimension, const float begin, const float step,
    const float offGrid, const af_interp_type method,
    const af_dtype valueType) {
    ApproxParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(outputDims[i]);
        p.outputStrides[i] = int64_t(outputStrides[i]);
        p.inputDims[i]     = uint64_t(inputDims[i]);
        p.inputStrides[i]  = int64_t(inputStrides[i]);
        p.xDims[i]         = uint64_t(positionDims[i]);
        p.xStrides[i]      = int64_t(positionStrides[i]);
    }
    p.outputOffset = int64_t(outputOffset);
    p.inputOffset  = int64_t(inputOffset);
    p.xOffset      = int64_t(positionOffset);
    p.xDimension = uint32_t(dimension);
    p.method     = uint32_t(method);
    p.xBegin     = begin;
    p.xStep      = step;
    p.offGrid    = offGrid;
    launchTwoInputKernel(output, outputBytes, input, inputBytes, positions,
                         positionBytes, &p, sizeof(p),
                         size_t(outputDims.elements()),
                         valueType == c32 ? "approx1_cfloat" : "approx1_float",
                         "approx1");
}

void launchMetalApprox2(
    BufferParam output, const size_t outputBytes,
    const af::dim4& outputDims, const af::dim4& outputStrides,
    const dim_t outputOffset, BufferParam input, const size_t inputBytes,
    const af::dim4& inputDims, const af::dim4& inputStrides,
    const dim_t inputOffset, BufferParam x, const size_t xBytes,
    const af::dim4& positionDims, const af::dim4& xStrides,
    const dim_t xOffset, BufferParam y, const size_t yBytes,
    const af::dim4& yStrides, const dim_t yOffset, const int xDimension,
    const float xBegin, const float xStep, const int yDimension,
    const float yBegin, const float yStep, const float offGrid,
    const af_interp_type method, const af_dtype valueType) {
    ApproxParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(outputDims[i]);
        p.outputStrides[i] = int64_t(outputStrides[i]);
        p.inputDims[i]     = uint64_t(inputDims[i]);
        p.inputStrides[i]  = int64_t(inputStrides[i]);
        p.xDims[i]         = uint64_t(positionDims[i]);
        p.xStrides[i]      = int64_t(xStrides[i]);
        p.yStrides[i]      = int64_t(yStrides[i]);
    }
    p.outputOffset = int64_t(outputOffset);
    p.inputOffset  = int64_t(inputOffset);
    p.xOffset      = int64_t(xOffset);
    p.yOffset      = int64_t(yOffset);
    p.xDimension = uint32_t(xDimension);
    p.yDimension = uint32_t(yDimension);
    p.method     = uint32_t(method);
    p.xBegin     = xBegin;
    p.xStep      = xStep;
    p.yBegin     = yBegin;
    p.yStep      = yStep;
    p.offGrid    = offGrid;
    launchThreeInputKernel(
        output, outputBytes, input, inputBytes, x, xBytes, y, yBytes, &p,
        sizeof(p), size_t(outputDims.elements()),
        valueType == c32 ? "approx2_cfloat" : "approx2_float", "approx2");
}

bool supportsMetalReduce(const af_dtype inputType,
                         const af_dtype outputType) noexcept {
    return reduceFunctionName(inputType, outputType) != nullptr;
}

bool supportsMetalReduceByKey(const af_dtype keyType, const af_dtype inputType,
                              const af_dtype outputType) noexcept {
    return reduceByKeyCompactFunctionName(keyType) != nullptr &&
           reduceByKeyFunctionName(keyType, inputType, outputType) != nullptr;
}

void launchMetalReduce(
    BufferParam output, const size_t outputBytes,
    const af::dim4& outputDims, const af::dim4& outputStrides,
    BufferParam input, const size_t inputBytes,
    const af::dim4& inputDims, const af::dim4& inputStrides,
    const int dimension, const uint32_t operation, const bool reduceAll,
    const bool changeNan, const double nanValue, const af_dtype inputType,
    const af_dtype outputType) {
    ReduceParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        p.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        p.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        p.inputStrides[i]  = static_cast<int64_t>(inputStrides[i]);
    }
    p.dimension = static_cast<uint32_t>(dimension);
    p.operation = operation;
    p.reduceAll = reduceAll;
    p.changeNan = changeNan;
    p.nanValue  = static_cast<float>(nanValue);

    const size_t outputCount = static_cast<size_t>(outputDims.elements());
    const size_t reductionLength = static_cast<size_t>(
        reduceAll ? inputDims.elements() : inputDims[dimension]);
    constexpr size_t chunkSize = 256;
    if (outputCount == 0) return;
    UNUSED(inputBytes);

    const char* baseName = reduceFunctionName(inputType, outputType);
    if (!baseName)
        AF_ERROR("Reduction type is not supported by Metal", AF_ERR_NOT_SUPPORTED);
    const std::string partialName = std::string(baseName) + "_partial";
    const std::string finalName = std::string(baseName) + "_final";
    auto* partialPipeline = metalPipeline(partialName.c_str());
    auto* finalPipeline = metalPipeline(finalName.c_str());

    const size_t chunkCount =
        std::max<size_t>((reductionLength + chunkSize - 1) / chunkSize, 1);
    if (outputCount > std::numeric_limits<size_t>::max() / chunkCount)
        AF_ERROR("Metal reduction scratch size overflow", AF_ERR_SIZE);
    const size_t partialCount = outputCount * chunkCount;
    // The largest accumulator used by any Metal reduction mapping is 8 bytes.
    if (partialCount > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
        AF_ERROR("Metal reduction scratch size overflow", AF_ERR_SIZE);
    const size_t partialBytes = partialCount * sizeof(uint64_t);
    const size_t groupCount = outputCount * chunkCount;
    if (groupCount > std::numeric_limits<uint32_t>::max() ||
        outputCount > std::numeric_limits<uint32_t>::max())
        AF_ERROR("Metal reduction dispatch exceeds the supported grid size",
                 AF_ERR_SIZE);
    auto partials = NS::TransferPtr(metalDevice()->newBuffer(
        partialBytes, MTL::ResourceStorageModePrivate));
    if (!partials)
        AF_ERROR("Could not allocate Metal reduction scratch", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);

    encoder->setComputePipelineState(partialPipeline);
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(partials.get(), 0, 1);
    encoder->setBytes(&p, sizeof(p), 2);
    encoder->dispatchThreadgroups(MTL::Size(groupCount, 1, 1),
                                  MTL::Size(chunkSize, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    encoder->setComputePipelineState(finalPipeline);
    encoder->setBuffer(partials.get(), 0, 0);
    encoder->setBuffer(output.buffer, output.offset, 1);
    encoder->setBytes(&p, sizeof(p), 2);
    encoder->dispatchThreadgroups(MTL::Size(outputCount, 1, 1),
                                  MTL::Size(chunkSize, 1, 1));
    encoder->endEncoding();

    auto partialLifetime = std::shared_ptr<MTL::Buffer>(
        partials.get(), [](MTL::Buffer* buffer) { buffer->release(); });
    partials->retain();
    commandBuffer->addCompletedHandler(
        [lifetime = std::move(partialLifetime)](MTL::CommandBuffer*) {
            static_cast<void>(lifetime);
        });
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal reduction dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchMetalReduceByKeyCompact(
    BufferParam outputKeys, const size_t outputBytes, BufferParam count,
    BufferParam inputKeys, const size_t inputBytes, const af::dim4& keyDims,
    const af::dim4& keyStrides, const af_dtype keyType) {
    ReduceByKeyCompactParams p{};
    p.length      = static_cast<uint64_t>(keyDims[0]);
    p.inputStride = static_cast<int64_t>(keyStrides[0]);
    launchInputTwoOutputKernel(
        outputKeys, count, outputBytes, inputKeys, inputBytes, 1, &p, sizeof(p),
        reduceByKeyCompactFunctionName(keyType), "reduce-by-key compaction");
}

void launchMetalReduceByKey(
    BufferParam output, const size_t outputBytes,
    const af::dim4& outputDims, const af::dim4& outputStrides,
    BufferParam keys, const size_t keyBytes, const af::dim4& keyStrides,
    BufferParam input, const size_t inputBytes, const af::dim4& inputDims,
    const af::dim4& inputStrides, const int dimension,
    const uint32_t operation, const int nReduced, const bool changeNan,
    const double nanValue, const af_dtype keyType, const af_dtype inputType,
    const af_dtype outputType) {
    ReduceByKeyParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        p.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        p.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        p.inputStrides[i]  = static_cast<int64_t>(inputStrides[i]);
    }
    p.keyStride = static_cast<int64_t>(keyStrides[0]);
    p.dimension = static_cast<uint32_t>(dimension);
    p.operation = operation;
    p.nReduced  = static_cast<uint32_t>(nReduced);
    p.changeNan = changeNan;
    p.nanValue  = static_cast<float>(nanValue);
    const size_t slices =
        static_cast<size_t>(outputDims.elements()) /
        static_cast<size_t>(std::max(nReduced, 1));
    launchTwoInputKernel(
        output, outputBytes, keys, keyBytes, input, inputBytes, &p, sizeof(p),
        slices, reduceByKeyFunctionName(keyType, inputType, outputType),
        "reduce-by-key");
}

bool supportsMetalIReduce(const af_dtype type) noexcept {
    return ireduceFunctionName(type) != nullptr;
}

void launchMetalIReduceAll(BufferParam output, size_t outputBytes,
                           BufferParam locations, size_t locationsBytes,
                           BufferParam input, size_t inputBytes,
                           const af::dim4& inputDims,
                           const af::dim4& inputStrides, bool isMax,
                           af_dtype type) {
    IReduceAllParams p{};
    for (int i = 0; i < 4; ++i) {
        p.inputDims[i]    = static_cast<uint64_t>(inputDims[i]);
        p.inputStrides[i] = static_cast<uint64_t>(inputStrides[i]);
    }
    p.isMax = isMax;
    UNUSED(locationsBytes);
    const size_t inputCount = static_cast<size_t>(inputDims.elements());
    constexpr size_t chunkSize = 256;
    UNUSED(outputBytes);
    UNUSED(inputBytes);

    const char* baseName = ireduceAllFunctionName(type);
    if (!baseName)
        AF_ERROR("Indexed reduction type is not supported by Metal",
                 AF_ERR_NOT_SUPPORTED);
    const std::string partialName = std::string(baseName) + "_partial";
    const std::string finalName = std::string(baseName) + "_final";
    auto* partialPipeline = metalPipeline(partialName.c_str());
    auto* finalPipeline = metalPipeline(finalName.c_str());
    const size_t chunkCount =
        std::max<size_t>((inputCount + chunkSize - 1) / chunkSize, 1);
    if (chunkCount > std::numeric_limits<uint32_t>::max())
        AF_ERROR("Metal indexed-reduction dispatch exceeds the supported grid size",
                 AF_ERR_SIZE);
    if (chunkCount > std::numeric_limits<size_t>::max() / sizeof(uint64_t))
        AF_ERROR("Metal indexed-reduction scratch size overflow", AF_ERR_SIZE);
    auto partialValues = NS::TransferPtr(metalDevice()->newBuffer(
        chunkCount * sizeof(uint64_t), MTL::ResourceStorageModePrivate));
    auto partialIndices = NS::TransferPtr(metalDevice()->newBuffer(
        chunkCount * sizeof(uint32_t), MTL::ResourceStorageModePrivate));
    auto partialValid = NS::TransferPtr(metalDevice()->newBuffer(
        chunkCount * sizeof(uint32_t), MTL::ResourceStorageModePrivate));
    if (!partialValues || !partialIndices || !partialValid)
        AF_ERROR("Could not allocate Metal indexed-reduction scratch",
                 AF_ERR_NO_MEM);

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    encoder->setComputePipelineState(partialPipeline);
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(partialValues.get(), 0, 1);
    encoder->setBuffer(partialIndices.get(), 0, 2);
    encoder->setBuffer(partialValid.get(), 0, 3);
    encoder->setBytes(&p, sizeof(p), 4);
    encoder->dispatchThreadgroups(MTL::Size(chunkCount, 1, 1),
                                  MTL::Size(chunkSize, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    encoder->setComputePipelineState(finalPipeline);
    encoder->setBuffer(partialValues.get(), 0, 0);
    encoder->setBuffer(partialIndices.get(), 0, 1);
    encoder->setBuffer(partialValid.get(), 0, 2);
    encoder->setBuffer(output.buffer, output.offset, 3);
    encoder->setBuffer(locations.buffer, locations.offset, 4);
    encoder->setBytes(&p, sizeof(p), 5);
    encoder->dispatchThreadgroups(MTL::Size(1, 1, 1),
                                  MTL::Size(chunkSize, 1, 1));
    encoder->endEncoding();

    auto valuesLifetime = std::shared_ptr<MTL::Buffer>(
        partialValues.get(), [](MTL::Buffer* buffer) { buffer->release(); });
    auto indicesLifetime = std::shared_ptr<MTL::Buffer>(
        partialIndices.get(), [](MTL::Buffer* buffer) { buffer->release(); });
    auto validLifetime = std::shared_ptr<MTL::Buffer>(
        partialValid.get(), [](MTL::Buffer* buffer) { buffer->release(); });
    partialValues->retain();
    partialIndices->retain();
    partialValid->retain();
    commandBuffer->addCompletedHandler(
        [values = std::move(valuesLifetime),
         indices = std::move(indicesLifetime),
         valid = std::move(validLifetime)](MTL::CommandBuffer*) {
            static_cast<void>(values);
            static_cast<void>(indices);
            static_cast<void>(valid);
        });
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(),
            "Metal all-indexed reduction dispatch failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchMetalIReduce(BufferParam output, size_t outputBytes,
                        const af::dim4& outputDims,
                        const af::dim4& outputStrides, BufferParam locations,
                        size_t locationsBytes, BufferParam input,
                        size_t inputBytes, const af::dim4& inputDims,
                        const af::dim4& inputStrides, int dimension,
                        bool isMax, af_dtype type) {
    IReduceParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        p.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        p.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        p.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    p.dimension = static_cast<uint32_t>(dimension);
    p.isMax     = isMax;
    launchInputTwoOutputKernel(
        output, locations, outputBytes, input, inputBytes,
        static_cast<size_t>(outputDims.elements()), &p, sizeof(p),
        ireduceFunctionName(type), "indexed reduction");
    UNUSED(locationsBytes);
}

void launchMetalRReduce(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam locations, size_t locationsBytes,
    BufferParam input, size_t inputBytes, const af::dim4& inputDims,
    const af::dim4& inputStrides, int dimension, BufferParam rlen,
    size_t rlenBytes, const af::dim4& rlenDims, const af::dim4& rlenStrides,
    bool isMax, af_dtype type) {
    IReduceParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        p.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        p.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        p.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        p.rlenDims[i]      = static_cast<uint64_t>(rlenDims[i]);
        p.rlenStrides[i]   = static_cast<uint64_t>(rlenStrides[i]);
    }
    p.dimension = static_cast<uint32_t>(dimension);
    p.isMax     = isMax;
    launchThreeInputKernel(
        output, outputBytes, input, inputBytes, rlen, rlenBytes, locations,
        locationsBytes, &p, sizeof(p), static_cast<size_t>(outputDims.elements()),
        rreduceFunctionName(type), "ragged indexed reduction");
}

bool supportsMetalConvolve(const af_dtype inputType,
                           const af_dtype filterType) noexcept {
    return convolveFunctionName(inputType, filterType, false) != nullptr;
}

void launchMetalConvolve(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam signal,
    const af::dim4& signalDims, const af::dim4& signalStrides,
    BufferParam filter, const af::dim4& filterDims,
    const af::dim4& filterStrides, const AF_BATCH_KIND kind, const int rank,
    const bool expand, const af_dtype inputType, const af_dtype filterType) {
    if (outputDims.elements() == 0) return;

    const char* scalarFunctionName =
        convolveFunctionName(inputType, filterType, false);
    if (!scalarFunctionName) {
        AF_ERROR("Input type is not supported by the Metal convolution kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    if (!output.buffer || !signal.buffer || !filter.buffer) {
        AF_ERROR("Could not allocate Metal convolution buffers", AF_ERR_NO_MEM);
    }

    ConvolveParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        p.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        p.signalDims[i]    = static_cast<uint64_t>(signalDims[i]);
        p.signalStrides[i] = static_cast<uint64_t>(signalStrides[i]);
        p.filterDims[i]    = static_cast<uint64_t>(filterDims[i]);
        p.filterStrides[i] = static_cast<uint64_t>(filterStrides[i]);
    }
    p.rank      = static_cast<uint32_t>(rank);
    p.expand    = expand;
    p.batchKind = static_cast<uint32_t>(kind);

    // Keep the experiment tightly scoped: only unbatched f32 rank-2
    // convolutions with 7x7 or 15x15 filters use the full-halo kernel.
    const bool unitBatchDims = outputDims[2] == 1 && outputDims[3] == 1 &&
                               signalDims[2] == 1 && signalDims[3] == 1 &&
                               filterDims[2] == 1 && filterDims[3] == 1;
    const bool supportedTileFilter = filterDims[0] == filterDims[1] &&
                                     (filterDims[0] == 7 ||
                                      filterDims[0] == 15);
    const bool useTiledKernel = inputType == f32 && filterType == f32 &&
                                rank == 2 && kind == AF_BATCH_NONE &&
                                unitBatchDims && supportedTileFilter;
    const char* functionName = scalarFunctionName;
    if (useTiledKernel) {
        functionName = filterDims[0] == 7 ? "convolve_float_tiled_7"
                                          : "convolve_float_tiled_15";
    }

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal convolution command encoder",
                 AF_ERR_RUNTIME);
    }
    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(signal.buffer, signal.offset, 0);
    encoder->setBuffer(filter.buffer, filter.offset, 1);
    encoder->setBuffer(output.buffer, output.offset, 2);
    encoder->setBytes(&p, sizeof(p), 3);
    if (useTiledKernel) {
        constexpr NS::UInteger tileWidth  = 8;
        constexpr NS::UInteger tileHeight = 8;
        const auto groupsX = static_cast<NS::UInteger>(
            (outputDims[0] + tileWidth - 1) / tileWidth);
        const auto groupsY = static_cast<NS::UInteger>(
            (outputDims[1] + tileHeight - 1) / tileHeight);
        encoder->dispatchThreadgroups(MTL::Size(groupsX, groupsY, 1),
                                      MTL::Size(tileWidth, tileHeight, 1));
    } else {
        const auto width = std::min<NS::UInteger>(
            256, pipeline->maxTotalThreadsPerThreadgroup());
        encoder->dispatchThreads(
            MTL::Size(static_cast<NS::UInteger>(outputDims.elements()), 1, 1),
            MTL::Size(width, 1, 1));
    }
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
}

bool supportsMetalSeparableConvolve(const af_dtype inputType,
                                    const af_dtype filterType) noexcept {
    return convolveFunctionName(inputType, filterType, true) != nullptr;
}

void launchMetalSeparableConvolve(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam temp,
    const af::dim4& tempDims, const af::dim4& tempStrides, BufferParam signal,
    const af::dim4& signalDims, const af::dim4& signalStrides,
    BufferParam columnFilter, const af::dim4& columnFilterDims,
    const af::dim4& columnFilterStrides, BufferParam rowFilter,
    const af::dim4& rowFilterDims, const af::dim4& rowFilterStrides,
    const bool expand, const af_dtype inputType, const af_dtype filterType) {
    const char* functionName =
        convolveFunctionName(inputType, filterType, true);
    if (!functionName) {
        AF_ERROR(
            "Input type is not supported by the Metal separable convolution kernel",
            AF_ERR_NOT_SUPPORTED);
    }
    if (!output.buffer || !temp.buffer || !signal.buffer ||
        !columnFilter.buffer || !rowFilter.buffer) {
        AF_ERROR("Could not allocate Metal separable convolution buffers",
                 AF_ERR_NO_MEM);
    }

    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR(
            "Could not create a Metal separable convolution command encoder",
            AF_ERR_RUNTIME);
    }
    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);

    SeparableConvolveParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = static_cast<uint64_t>(tempDims[i]);
        p.outputStrides[i] = static_cast<uint64_t>(tempStrides[i]);
        p.signalDims[i]    = static_cast<uint64_t>(signalDims[i]);
        p.signalStrides[i] = static_cast<uint64_t>(signalStrides[i]);
        p.filterDims[i]    = static_cast<uint64_t>(columnFilterDims[i]);
        p.filterStrides[i] = static_cast<uint64_t>(columnFilterStrides[i]);
    }
    p.convDim = 0;
    p.expand  = expand;
    encoder->setBuffer(signal.buffer, signal.offset, 0);
    encoder->setBuffer(columnFilter.buffer, columnFilter.offset, 1);
    encoder->setBuffer(temp.buffer, temp.offset, 2);
    encoder->setBytes(&p, sizeof(p), 3);
    auto width = std::min<NS::UInteger>(
        256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(
        MTL::Size(static_cast<NS::UInteger>(tempDims.elements()), 1, 1),
        MTL::Size(width, 1, 1));
    encoder->memoryBarrier(MTL::BarrierScopeBuffers);

    p.outputDims[0]    = static_cast<uint64_t>(outputDims[0]);
    p.outputDims[1]    = static_cast<uint64_t>(outputDims[1]);
    p.outputDims[2]    = static_cast<uint64_t>(outputDims[2]);
    p.outputDims[3]    = static_cast<uint64_t>(outputDims[3]);
    p.outputStrides[0] = static_cast<uint64_t>(outputStrides[0]);
    p.outputStrides[1] = static_cast<uint64_t>(outputStrides[1]);
    p.outputStrides[2] = static_cast<uint64_t>(outputStrides[2]);
    p.outputStrides[3] = static_cast<uint64_t>(outputStrides[3]);
    p.signalDims[0]    = static_cast<uint64_t>(tempDims[0]);
    p.signalDims[1]    = static_cast<uint64_t>(tempDims[1]);
    p.signalDims[2]    = static_cast<uint64_t>(tempDims[2]);
    p.signalDims[3]    = static_cast<uint64_t>(tempDims[3]);
    p.signalStrides[0] = static_cast<uint64_t>(tempStrides[0]);
    p.signalStrides[1] = static_cast<uint64_t>(tempStrides[1]);
    p.signalStrides[2] = static_cast<uint64_t>(tempStrides[2]);
    p.signalStrides[3] = static_cast<uint64_t>(tempStrides[3]);
    p.filterDims[0]    = static_cast<uint64_t>(rowFilterDims[0]);
    p.filterDims[1]    = static_cast<uint64_t>(rowFilterDims[1]);
    p.filterDims[2]    = static_cast<uint64_t>(rowFilterDims[2]);
    p.filterDims[3]    = static_cast<uint64_t>(rowFilterDims[3]);
    p.filterStrides[0] = static_cast<uint64_t>(rowFilterStrides[0]);
    p.filterStrides[1] = static_cast<uint64_t>(rowFilterStrides[1]);
    p.filterStrides[2] = static_cast<uint64_t>(rowFilterStrides[2]);
    p.filterStrides[3] = static_cast<uint64_t>(rowFilterStrides[3]);
    p.convDim          = 1;
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(temp.buffer, temp.offset, 0);
    encoder->setBuffer(rowFilter.buffer, rowFilter.offset, 1);
    encoder->setBuffer(output.buffer, output.offset, 2);
    encoder->setBytes(&p, sizeof(p), 3);
    width = std::min<NS::UInteger>(
        256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(
        MTL::Size(static_cast<NS::UInteger>(outputDims.elements()), 1, 1),
        MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
}

bool supportsMetalConvolveNN(const af_dtype type) noexcept {
    return convolveNNFunctionName(type) != nullptr;
}

void launchMetalConvolveNN(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam signal,
    const af::dim4& signalDims, const af::dim4& signalStrides,
    BufferParam filter, const af::dim4& filterDims,
    const af::dim4& filterStrides, const af::dim4& stride,
    const af::dim4& padding, const af::dim4& dilation, const af_dtype type) {
    const char* baseName = convolveNNFunctionName(type);
    if (!baseName) {
        AF_ERROR("Input type is not supported by the Metal NN convolution kernel",
                 AF_ERR_NOT_SUPPORTED);
    }

    ConvolveNNParams p{};
    setConvolveNNParams(p, outputDims, outputStrides, signalDims,
                        signalStrides, filterDims, filterStrides, af::dim4(),
                        af::dim4(), stride, padding, dilation);
    launchConvolveNNKernel(baseName, signal, filter, output, p, outputDims);
}

void launchMetalConvolveNNDataGradient(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam incomingGradient,
    const af::dim4& incomingGradientDims,
    const af::dim4& incomingGradientStrides, BufferParam filter,
    const af::dim4& filterDims, const af::dim4& filterStrides,
    const af::dim4& stride, const af::dim4& padding,
    const af::dim4& dilation, const af_dtype type) {
    const char* baseName = convolveNNFunctionName(type);
    if (!baseName) {
        AF_ERROR("Input type is not supported by the Metal NN convolution gradient kernel",
                 AF_ERR_NOT_SUPPORTED);
    }

    ConvolveNNParams p{};
    setConvolveNNParams(p, outputDims, outputStrides, incomingGradientDims,
                        incomingGradientStrides, filterDims, filterStrides,
                        incomingGradientDims, incomingGradientStrides, stride,
                        padding, dilation);
    launchConvolveNNKernel(std::string(baseName) + "_data_gradient",
                           incomingGradient, filter, output, p, outputDims);
}

void launchMetalConvolveNNFilterGradient(
    BufferParam output, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam signal,
    const af::dim4& signalDims, const af::dim4& signalStrides,
    BufferParam incomingGradient, const af::dim4& incomingGradientDims,
    const af::dim4& incomingGradientStrides, const af::dim4& stride,
    const af::dim4& padding, const af::dim4& dilation, const af_dtype type) {
    const char* baseName = convolveNNFunctionName(type);
    if (!baseName) {
        AF_ERROR("Input type is not supported by the Metal NN convolution gradient kernel",
                 AF_ERR_NOT_SUPPORTED);
    }

    ConvolveNNParams p{};
    setConvolveNNParams(p, outputDims, outputStrides, signalDims, signalStrides,
                        outputDims, outputStrides, incomingGradientDims,
                        incomingGradientStrides, stride, padding, dilation);
    launchConvolveNNKernel(std::string(baseName) + "_filter_gradient", signal,
                           incomingGradient, output, p, outputDims);
}

void launchMetalFast(
    BufferParam input, const size_t inputBytes, const af::dim4& inputDims,
    BufferParam scoreImage, BufferParam x, BufferParam y, BufferParam scores,
    unsigned* count, const float threshold, const unsigned arcLength,
    const bool nonmax, const unsigned maxFeatures, const unsigned edge,
    const af_dtype inputType) {
    FastParams p{uint64_t(inputDims[0]), uint64_t(inputDims[1]), threshold,
                 arcLength, nonmax, maxFeatures, edge};
    const size_t rows = static_cast<size_t>(inputDims[0]);
    const size_t columns = static_cast<size_t>(inputDims[1]);
    const size_t margin = 2 * static_cast<size_t>(edge);
    if (rows <= margin || columns <= margin) {
        *count = 0;
        return;
    }
    const size_t candidateCount = (rows - margin) * (columns - margin);
    if (!input.buffer || !x.buffer || !y.buffer || !scores.buffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    const char* functionName = fastFunctionName(inputType);
    if (!functionName)
        AF_ERROR("Input type is not supported by the Metal FAST kernel",
                 AF_ERR_NOT_SUPPORTED);
    auto countBuffer = NS::TransferPtr(metalDevice()->newBuffer(
        sizeof(uint32_t), MTL::ResourceStorageModeShared));
    if (!countBuffer)
        AF_ERROR("Could not allocate Metal FAST count buffer", AF_ERR_NO_MEM);
    *static_cast<uint32_t*>(countBuffer->contents()) = 0;
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
    encoder->setBuffer(scoreImage.buffer ? scoreImage.buffer : x.buffer,
                       scoreImage.buffer ? scoreImage.offset : x.offset, 1);
    encoder->setBuffer(x.buffer, x.offset, 2);
    encoder->setBuffer(y.buffer, y.offset, 3);
    encoder->setBuffer(scores.buffer, scores.offset, 4);
    encoder->setBuffer(countBuffer.get(), 0, 5);
    encoder->setBytes(&p, sizeof(p), 6);
    UNUSED(inputBytes);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(candidateCount, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    // Feature count is consumed immediately on the CPU.
    syncCommandQueue();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            metalErrorDescription(commandBuffer->error(), "Metal FAST locate failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    *count = std::min(*static_cast<uint32_t*>(countBuffer->contents()),
                      maxFeatures);
}

void fastNonMaxMetal(CParam<float> score, CParam<float> xInput,
                     CParam<float> yInput, Param<float> xOutput,
                     Param<float> yOutput, Param<float> scoreOutput,
                     unsigned* count, const unsigned totalFeatures,
                     const unsigned edge) {
    const BufferParam scoreParam = score.bufferParam();
    const BufferParam xInputParam = xInput.bufferParam();
    const BufferParam yInputParam = yInput.bufferParam();
    const BufferParam xOutputParam = xOutput.bufferParam();
    const BufferParam yOutputParam = yOutput.bufferParam();
    const BufferParam scoreOutputParam = scoreOutput.bufferParam();
    if (!scoreParam.buffer || !xInputParam.buffer || !yInputParam.buffer ||
        !xOutputParam.buffer || !yOutputParam.buffer ||
        !scoreOutputParam.buffer)
        AF_ERROR("Could not allocate Metal FAST nonmax buffers", AF_ERR_NO_MEM);

    auto countBuffer = NS::TransferPtr(metalDevice()->newBuffer(
        sizeof(uint32_t), MTL::ResourceStorageModeShared));
    if (!countBuffer)
        AF_ERROR("Could not allocate Metal FAST nonmax count buffer",
                 AF_ERR_NO_MEM);
    *static_cast<uint32_t*>(countBuffer->contents()) = 0;

    FastParams p{uint64_t(score.dims(0)), uint64_t(score.dims(1)), 0.0f, 0,
                 1, totalFeatures, edge};
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = metalPipeline("fast_nonmax_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(scoreParam.buffer, scoreParam.offset, 0);
    encoder->setBuffer(xInputParam.buffer, xInputParam.offset, 1);
    encoder->setBuffer(yInputParam.buffer, yInputParam.offset, 2);
    encoder->setBuffer(xOutputParam.buffer, xOutputParam.offset, 3);
    encoder->setBuffer(yOutputParam.buffer, yOutputParam.offset, 4);
    encoder->setBuffer(scoreOutputParam.buffer, scoreOutputParam.offset, 5);
    encoder->setBuffer(countBuffer.get(), 0, 6);
    encoder->setBytes(&p, sizeof(p), 7);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(totalFeatures, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    syncCommandQueue();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal FAST nonmax failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    *count = *static_cast<uint32_t*>(countBuffer->contents());
}

void harrisSecondOrderMetal(Param<float> ixx, Param<float> ixy,
                            Param<float> iyy, CParam<float> ix,
                            CParam<float> iy) {
    const size_t elements = size_t(ix.dims().elements());
    HarrisParams p{elements, uint32_t(ix.dims(0)), uint32_t(ix.dims(1)), 0, 0};
    const BufferParam ixp = ix.bufferParam();
    const BufferParam iyp = iy.bufferParam();
    const BufferParam ixxp = ixx.bufferParam();
    const BufferParam ixyp = ixy.bufferParam();
    const BufferParam iyyp = iyy.bufferParam();
    if (!ixp.buffer || !iyp.buffer || !ixxp.buffer || !ixyp.buffer ||
        !iyyp.buffer)
        AF_ERROR("Could not allocate Metal buffers", AF_ERR_NO_MEM);
    auto commandBuffer =
        NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto* pipeline = metalPipeline("harris_second_order_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(ixp.buffer, ixp.offset, 0);
    encoder->setBuffer(iyp.buffer, iyp.offset, 1);
    encoder->setBuffer(ixxp.buffer, ixxp.offset, 2);
    encoder->setBuffer(ixyp.buffer, ixyp.offset, 3);
    encoder->setBuffer(iyyp.buffer, iyyp.offset, 4);
    encoder->setBytes(&p, sizeof(p), 5);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(elements, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal Harris second-order failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void harrisResponseMetal(Param<float> output, const unsigned rows,
                         const unsigned columns, CParam<float> ixx,
                         CParam<float> ixy, CParam<float> iyy, const float k,
                         const unsigned border) {
    const size_t elements = size_t(output.dims().elements());
    HarrisParams p{elements, rows, columns, border, k};
    const size_t bytes = elements * sizeof(float);
    launchThreeInputKernel(output.bufferParam(), bytes, ixx.bufferParam(), bytes,
                           ixy.bufferParam(), bytes, iyy.bufferParam(), bytes,
                           &p, sizeof(p), elements,
                           "harris_response_float", "Harris response");
}

void harrisNonMaxMetal(CParam<float> response, Param<float> xOutput,
                       Param<float> yOutput, Param<float> responseOutput,
                       unsigned* count, const unsigned rows,
                       const unsigned columns, const float minResponse,
                       const unsigned border, const unsigned maxCorners) {
    const BufferParam responseParam = response.bufferParam();
    const BufferParam xParam        = xOutput.bufferParam();
    const BufferParam yParam        = yOutput.bufferParam();
    const BufferParam scoreParam    = responseOutput.bufferParam();
    if (!responseParam.buffer || !xParam.buffer || !yParam.buffer ||
        !scoreParam.buffer)
        AF_ERROR("Could not allocate Metal Harris buffers", AF_ERR_NO_MEM);

    auto countBuffer = NS::TransferPtr(metalDevice()->newBuffer(
        sizeof(uint32_t), MTL::ResourceStorageModeShared));
    if (!countBuffer)
        AF_ERROR("Could not allocate Metal Harris count buffer", AF_ERR_NO_MEM);
    *static_cast<uint32_t*>(countBuffer->contents()) = 0;

    HarrisNonMaxParams params{rows, columns, border, maxCorners, minResponse};
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);

    auto* pipeline = metalPipeline("harris_nonmax_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(responseParam.buffer, responseParam.offset, 0);
    encoder->setBuffer(xParam.buffer, xParam.offset, 1);
    encoder->setBuffer(yParam.buffer, yParam.offset, 2);
    encoder->setBuffer(scoreParam.buffer, scoreParam.offset, 3);
    encoder->setBuffer(countBuffer.get(), 0, 4);
    encoder->setBytes(&params, sizeof(params), 5);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(size_t(rows) * columns, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    syncCommandQueue();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal Harris non-max suppression failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    *count = *static_cast<uint32_t*>(countBuffer->contents());
}

void harrisKeepCornersMetal(Param<float> xOutput, Param<float> yOutput,
                            Param<float> responseOutput, CParam<float> xInput,
                            CParam<float> yInput, CParam<float> responseInput,
                            CParam<unsigned> responseIndex,
                            const unsigned corners) {
    const size_t outputBytes = size_t(corners) * sizeof(float);
    HarrisKeepParams params{corners};
    launchFourInputThreeOutputKernel(
        xOutput.bufferParam(), yOutput.bufferParam(), responseOutput.bufferParam(),
        outputBytes, xInput.bufferParam(), outputBytes, yInput.bufferParam(),
        outputBytes, responseInput.bufferParam(), outputBytes,
        responseIndex.bufferParam(), size_t(corners) * sizeof(unsigned),
        &params, sizeof(params), corners,
        "harris_keep_corners", "Harris corner selection");
}

void launchSusanResponse(
    BufferParam output, const size_t outputBytes, BufferParam input,
    const size_t inputBytes, const af::dim4& outputDims, const unsigned rows,
    const unsigned columns, const unsigned radius,
    const float differenceThreshold, const float geometricThreshold,
    const unsigned border, const af_dtype type) {
    SusanParams p{
        rows, columns, radius, border, differenceThreshold, geometricThreshold};
    const char* functionName = susanResponseFunctionName(type);
    if (!functionName) {
        AF_ERROR("Input type is not supported by the Metal SUSAN kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &p, sizeof(p), functionName, "SUSAN response");
}

void launchSusanNonMax(BufferParam xOutput, BufferParam yOutput,
                       BufferParam responseOutput, BufferParam response,
                       unsigned* count, const unsigned rows,
                       const unsigned columns, const unsigned border,
                       const unsigned maxCorners, const af_dtype type) {
    if (!xOutput.buffer || !yOutput.buffer || !responseOutput.buffer ||
        !response.buffer || !count) {
        AF_ERROR("Could not allocate Metal SUSAN buffers", AF_ERR_NO_MEM);
    }
    const char* functionName = susanNonMaxFunctionName(type);
    if (!functionName) {
        AF_ERROR("Input type is not supported by the Metal SUSAN kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    SusanNonMaxParams p{rows, columns, border, maxCorners};
    auto countBuffer = NS::TransferPtr(metalDevice()->newBuffer(
        sizeof(uint32_t), MTL::ResourceStorageModeShared));
    if (!countBuffer) {
        AF_ERROR("Could not allocate Metal SUSAN count buffer", AF_ERR_NO_MEM);
    }
    *static_cast<uint32_t*>(countBuffer->contents()) = 0;

    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }
    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(response.buffer, response.offset, 0);
    encoder->setBuffer(xOutput.buffer, xOutput.offset, 1);
    encoder->setBuffer(yOutput.buffer, yOutput.offset, 2);
    encoder->setBuffer(responseOutput.buffer, responseOutput.offset, 3);
    encoder->setBuffer(countBuffer.get(), 0, 4);
    encoder->setBytes(&p, sizeof(p), 5);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(size_t(rows) * columns, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    syncCommandQueue();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal SUSAN non-max suppression failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    *count = *static_cast<uint32_t*>(countBuffer->contents());
}

namespace {

const char* randomUniformFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "random_uniform_float";
        case s32: return "random_uniform_int";
        case u32: return "random_uniform_uint";
        case s64: return "random_uniform_long";
        case u64: return "random_uniform_ulong";
        case s8: return "random_uniform_schar";
        case b8: return "random_uniform_char";
        case u8: return "random_uniform_uchar";
        case s16: return "random_uniform_short";
        case u16: return "random_uniform_ushort";
        case f16: return "random_uniform_half";
        default: return nullptr;
    }
}

const char* randomNormalFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "random_normal_float";
        case f16: return "random_normal_half";
        default: return nullptr;
    }
}

const char* randomMersenneUniformFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "random_mersenne_uniform_float";
        case s32: return "random_mersenne_uniform_int";
        case u32: return "random_mersenne_uniform_uint";
        case s64: return "random_mersenne_uniform_long";
        case u64: return "random_mersenne_uniform_ulong";
        case s8: return "random_mersenne_uniform_schar";
        case b8: return "random_mersenne_uniform_char";
        case u8: return "random_mersenne_uniform_uchar";
        case s16: return "random_mersenne_uniform_short";
        case u16: return "random_mersenne_uniform_ushort";
        case f16: return "random_mersenne_uniform_half";
        default: return nullptr;
    }
}

const char* randomMersenneNormalFunctionName(const af_dtype type) {
    switch (type) {
        case f32: return "random_mersenne_normal_float";
        case f16: return "random_mersenne_normal_half";
        default: return nullptr;
    }
}

bool supportedCounterEngine(const af_random_engine_type type) {
    return type == AF_RANDOM_ENGINE_PHILOX_4X32_10 ||
           type == AF_RANDOM_ENGINE_THREEFRY_2X32_16;
}

}  // namespace

bool supportsMetalRandomUniform(const af_dtype type) noexcept {
    return randomUniformFunctionName(type) != nullptr;
}

bool supportsMetalRandomNormal(const af_dtype type) noexcept {
    return randomNormalFunctionName(type) != nullptr;
}

void launchMetalRandomUniform(BufferParam output, const size_t outputBytes,
                              const size_t elements,
                              const unsigned long long seed,
                              const unsigned long long counter,
                              const af_random_engine_type type,
                              const af_dtype dataType) {
    if (!supportedCounterEngine(type)) {
        AF_ERROR("The native Metal counter-based random kernel requires Philox or Threefry",
                 AF_ERR_NOT_SUPPORTED);
    }
    const char* functionName = randomUniformFunctionName(dataType);
    if (!functionName) {
        AF_ERROR("The random type is not supported by the native Metal kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    RandomParams p{elements, seed, counter, uint32_t(type)};
    launchOutputKernel(output, outputBytes, elements, &p, sizeof(p),
                       functionName, "random uniform");
}

void launchMetalRandomNormal(BufferParam output, const size_t outputBytes,
                             const size_t elements,
                             const unsigned long long seed,
                             const unsigned long long counter,
                             const af_random_engine_type type,
                             const af_dtype dataType) {
    if (!supportedCounterEngine(type)) {
        AF_ERROR("The native Metal counter-based random kernel requires Philox or Threefry",
                 AF_ERR_NOT_SUPPORTED);
    }
    const char* functionName = randomNormalFunctionName(dataType);
    if (!functionName) {
        AF_ERROR("The normal random type is not supported by the native Metal kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    RandomParams p{elements, seed, counter, uint32_t(type)};
    launchOutputKernel(output, outputBytes, elements, &p, sizeof(p),
                       functionName, "random normal");
}

void launchMetalRandomMersenne(
    BufferParam output, const size_t outputBytes, const size_t elements,
    BufferParam state, BufferParam pos, BufferParam sh1, BufferParam sh2,
    const unsigned mask, BufferParam recursionTable, BufferParam temperTable,
    const bool normal, const af_dtype dataType) {
    const char* functionName = normal
                                   ? randomMersenneNormalFunctionName(dataType)
                                   : randomMersenneUniformFunctionName(dataType);
    if (!functionName) {
        AF_ERROR("The random type is not supported by the native Metal Mersenne kernel",
                 AF_ERR_NOT_SUPPORTED);
    }
    if (outputBytes == 0 || elements == 0) return;
    if (!output.buffer || !state.buffer || !pos.buffer || !sh1.buffer ||
        !sh2.buffer || !recursionTable.buffer || !temperTable.buffer) {
        AF_ERROR("Could not allocate native Metal random buffers", AF_ERR_NO_MEM);
    }

    size_t elementSize = 0;
    switch (dataType) {
        case f32:
        case s32:
        case u32: elementSize = 4; break;
        case s64:
        case u64: elementSize = 8; break;
        case s8:
        case b8:
        case u8: elementSize = 1; break;
        case s16:
        case u16:
        case f16: elementSize = 2; break;
        default:
            AF_ERROR("The random type is not supported by the native Metal Mersenne kernel",
                     AF_ERR_NOT_SUPPORTED);
    }

    constexpr size_t threadgroupSize = 256;
    const size_t elementsPerIteration =
        threadgroupSize * (4 * sizeof(uint) / elementSize);
    const size_t minElementsPerBlock = 32 * elementsPerIteration;
    const size_t blocks =
        std::min<size_t>(32, (elements - 1) / minElementsPerBlock + 1);
    size_t elementsPerBlock = (elements - 1) / blocks + 1;
    elementsPerBlock =
        ((elementsPerBlock - 1) / elementsPerIteration + 1) *
        elementsPerIteration;

    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal random command encoder",
                 AF_ERR_RUNTIME);
    }
    auto* pipeline = metalPipeline(functionName);
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(output.buffer, output.offset, 0);
    encoder->setBuffer(state.buffer, state.offset, 1);
    encoder->setBuffer(pos.buffer, pos.offset, 2);
    encoder->setBuffer(sh1.buffer, sh1.offset, 3);
    encoder->setBuffer(sh2.buffer, sh2.offset, 4);
    encoder->setBuffer(recursionTable.buffer, recursionTable.offset, 5);
    encoder->setBuffer(temperTable.buffer, temperTable.offset, 6);
    MersenneParams p{elements, elementsPerBlock, mask,
                     static_cast<uint32_t>(blocks)};
    encoder->setBytes(&p, sizeof(p), 7);
    const auto maxThreads = pipeline->maxTotalThreadsPerThreadgroup();
    if (maxThreads < threadgroupSize) {
        AF_ERROR("Metal device does not support the Mersenne threadgroup size",
                 AF_ERR_NOT_SUPPORTED);
    }
    encoder->dispatchThreads(MTL::Size(blocks * threadgroupSize, 1, 1),
                             MTL::Size(threadgroupSize, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
}

void randomMersenneInitMetal(Param<uint> state, CParam<uint> table,
                             const unsigned long long seed) {
    MersenneInitParams p{seed};
    launchSingleInputKernel(
        state.bufferParam(), size_t(state.dims().elements()) * sizeof(uint),
        state.dims(), table.bufferParam(),
        size_t(table.dims().elements()) * sizeof(uint), &p, sizeof(p),
        "random_mersenne_init", "Mersenne initialization");
}

const char* sparseTypeSuffix(const af_dtype type) {
    switch (type) {
        case f32: return "float";
        case c32: return "cfloat";
        default: return nullptr;
    }
}

bool supportsMetalSparse(const af_dtype type) noexcept {
    return sparseTypeSuffix(type) != nullptr;
}

bool supportsMetalSparseArith(const af_dtype type) noexcept {
    return supportsMetalSparse(type);
}

static void finishSparseCommandBuffer(MTL::CommandBuffer* commandBuffer,
                                      const char* operation) {
    submitCommandBuffer(commandBuffer);
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message =
            metalErrorDescription(commandBuffer->error(), operation);
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchMetalDenseToCsr(BufferParam values, const size_t valuesBytes,
                           BufferParam rowIdx, const size_t rowIdxBytes,
                           BufferParam colIdx, const size_t colIdxBytes,
                           BufferParam input, const size_t inputBytes,
                           const af::dim4& inputDims,
                           const af::dim4& inputStrides, const size_t nonzeros,
                           const af_dtype type) {
    const char* suffix = sparseTypeSuffix(type);
    if (!suffix) AF_ERROR("Metal dense-to-CSR type is not supported", AF_ERR_NOT_SUPPORTED);
    if (!input.buffer || !rowIdx.buffer)
        AF_ERROR("Could not allocate Metal sparse buffers", AF_ERR_NO_MEM);
    if (nonzeros > 0 && (!values.buffer || !colIdx.buffer))
        AF_ERROR("Could not allocate Metal sparse buffers", AF_ERR_NO_MEM);

    SparseDenseToCsrParams p{uint32_t(inputDims[0]), uint32_t(inputDims[1]),
                             uint32_t(inputStrides[1]), uint32_t(nonzeros)};
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    std::string countName = std::string("sparse_dense_to_csr_count_") + suffix;
    auto* pipeline = metalPipeline(countName.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(rowIdx.buffer, rowIdx.offset, 1);
    encoder->setBytes(&p, sizeof(p), 2);
    auto width = std::min<NS::UInteger>(
        256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(inputDims[0], 1, 1),
                             MTL::Size(width, 1, 1));

    std::string prefixName = std::string("sparse_dense_to_csr_prefix_") + suffix;
    pipeline = metalPipeline(prefixName.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(rowIdx.buffer, rowIdx.offset, 0);
    encoder->setBytes(&p, sizeof(p), 1);
    encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

    if (nonzeros > 0) {
        std::string scatterName =
            std::string("sparse_dense_to_csr_scatter_") + suffix;
        pipeline = metalPipeline(scatterName.c_str());
        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(input.buffer, input.offset, 0);
        encoder->setBuffer(values.buffer, values.offset, 1);
        encoder->setBuffer(colIdx.buffer, colIdx.offset, 2);
        encoder->setBuffer(rowIdx.buffer, rowIdx.offset, 3);
        encoder->setBytes(&p, sizeof(p), 4);
        width = std::min<NS::UInteger>(
            256, pipeline->maxTotalThreadsPerThreadgroup());
        encoder->dispatchThreads(MTL::Size(inputDims[0], 1, 1),
                                 MTL::Size(width, 1, 1));
    }
    encoder->endEncoding();
    UNUSED(valuesBytes);
    UNUSED(rowIdxBytes);
    UNUSED(colIdxBytes);
    UNUSED(inputBytes);
    finishSparseCommandBuffer(commandBuffer.get(),
                              "Metal dense-to-CSR conversion failed");
}

void launchMetalCsrToCoo(BufferParam outputValues, const size_t outputValuesBytes,
                         BufferParam outputRows, const size_t outputRowsBytes,
                         BufferParam outputColumns, const size_t outputColumnsBytes,
                         BufferParam inputValues, const size_t inputValuesBytes,
                         BufferParam inputRows, const size_t inputRowsBytes,
                         BufferParam inputColumns, const size_t inputColumnsBytes,
                         const size_t nonzeros, const size_t rows,
                         const af_dtype type) {
    const char* suffix = sparseTypeSuffix(type);
    if (!suffix) AF_ERROR("Metal CSR-to-COO type is not supported", AF_ERR_NOT_SUPPORTED);
    if (nonzeros == 0) return;
    if (!inputValues.buffer || !inputRows.buffer || !inputColumns.buffer ||
        !outputValues.buffer || !outputRows.buffer || !outputColumns.buffer)
        AF_ERROR("Could not allocate Metal sparse buffers", AF_ERR_NO_MEM);
    SparseParams p{uint32_t(rows), 0, 0, uint32_t(nonzeros)};
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    std::string name = std::string("sparse_csr_to_coo_") + suffix;
    auto* pipeline = metalPipeline(name.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(inputValues.buffer, inputValues.offset, 0);
    encoder->setBuffer(inputRows.buffer, inputRows.offset, 1);
    encoder->setBuffer(inputColumns.buffer, inputColumns.offset, 2);
    encoder->setBuffer(outputValues.buffer, outputValues.offset, 3);
    encoder->setBuffer(outputRows.buffer, outputRows.offset, 4);
    encoder->setBuffer(outputColumns.buffer, outputColumns.offset, 5);
    encoder->setBytes(&p, sizeof(p), 6);
    const auto width = std::min<NS::UInteger>(
        256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(nonzeros, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    UNUSED(outputValuesBytes);
    UNUSED(outputRowsBytes);
    UNUSED(outputColumnsBytes);
    UNUSED(inputValuesBytes);
    UNUSED(inputRowsBytes);
    UNUSED(inputColumnsBytes);
    finishSparseCommandBuffer(commandBuffer.get(),
                              "Metal CSR-to-COO conversion failed");
}

void launchMetalCooToCsr(BufferParam outputValues, const size_t outputValuesBytes,
                         BufferParam outputRowIdx, const size_t outputRowIdxBytes,
                         BufferParam outputColumns, const size_t outputColumnsBytes,
                         BufferParam inputValues, const size_t inputValuesBytes,
                         BufferParam inputRows, const size_t inputRowsBytes,
                         BufferParam inputColumns, const size_t inputColumnsBytes,
                         BufferParam cursor, const size_t cursorBytes,
                         const size_t rows, const size_t nonzeros,
                         const af_dtype type) {
    const char* suffix = sparseTypeSuffix(type);
    if (!suffix) AF_ERROR("Metal COO-to-CSR type is not supported", AF_ERR_NOT_SUPPORTED);
    if (!outputRowIdx.buffer || (rows > 0 && !cursor.buffer))
        AF_ERROR("Could not allocate Metal sparse buffers", AF_ERR_NO_MEM);
    if (nonzeros > 0 && (!inputValues.buffer || !inputRows.buffer ||
                         !inputColumns.buffer || !outputValues.buffer ||
                         !outputColumns.buffer))
        AF_ERROR("Could not allocate Metal sparse buffers", AF_ERR_NO_MEM);

    SparseCooToCsrParams p{uint32_t(rows), uint32_t(nonzeros)};
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    auto width = NS::UInteger(1);
    std::string clearName = std::string("sparse_coo_to_csr_clear_") + suffix;
    auto* pipeline = metalPipeline(clearName.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(outputRowIdx.buffer, outputRowIdx.offset, 0);
    encoder->setBytes(&p, sizeof(p), 1);
    width = std::min<NS::UInteger>(
        256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(rows, 1, 1), MTL::Size(width, 1, 1));

    if (nonzeros > 0) {
        std::string countName = std::string("sparse_coo_to_csr_count_") + suffix;
        pipeline = metalPipeline(countName.c_str());
        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(inputRows.buffer, inputRows.offset, 0);
        encoder->setBuffer(outputRowIdx.buffer, outputRowIdx.offset, 1);
        encoder->setBytes(&p, sizeof(p), 2);
        width = std::min<NS::UInteger>(
            256, pipeline->maxTotalThreadsPerThreadgroup());
        encoder->dispatchThreads(MTL::Size(nonzeros, 1, 1),
                                 MTL::Size(width, 1, 1));
    }

    std::string prefixName = std::string("sparse_coo_to_csr_prefix_") + suffix;
    pipeline = metalPipeline(prefixName.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(outputRowIdx.buffer, outputRowIdx.offset, 0);
    encoder->setBytes(&p, sizeof(p), 1);
    encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));

    if (rows > 0 && nonzeros > 0) {
        std::string scatterName =
            std::string("sparse_coo_to_csr_scatter_sorted_") + suffix;
        pipeline = metalPipeline(scatterName.c_str());
        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(inputValues.buffer, inputValues.offset, 0);
        encoder->setBuffer(inputColumns.buffer, inputColumns.offset, 1);
        encoder->setBuffer(outputValues.buffer, outputValues.offset, 2);
        encoder->setBuffer(outputColumns.buffer, outputColumns.offset, 3);
        encoder->setBuffer(outputRowIdx.buffer, outputRowIdx.offset, 4);
        encoder->setBytes(&p, sizeof(p), 5);
        width = std::min<NS::UInteger>(
            256, pipeline->maxTotalThreadsPerThreadgroup());
        encoder->dispatchThreads(MTL::Size(rows, 1, 1),
                                 MTL::Size(width, 1, 1));
    }
    encoder->endEncoding();
    UNUSED(outputValuesBytes);
    UNUSED(outputRowIdxBytes);
    UNUSED(outputColumnsBytes);
    UNUSED(inputValuesBytes);
    UNUSED(inputRowsBytes);
    UNUSED(inputColumnsBytes);
    UNUSED(cursorBytes);
    finishSparseCommandBuffer(commandBuffer.get(),
                              "Metal COO-to-CSR conversion failed");
}

void launchMetalSparseToDense(BufferParam output, const size_t outputBytes,
                              BufferParam values, const size_t valuesBytes,
                              BufferParam rows, const size_t rowsBytes,
                              BufferParam columns, const size_t columnsBytes,
                              const af::dim4& outputDims,
                              const af::dim4& outputStrides, const bool csr,
                              const af_dtype type) {
    const char* suffix = sparseTypeSuffix(type);
    if (!suffix) AF_ERROR("Metal sparse-to-dense type is not supported", AF_ERR_NOT_SUPPORTED);
    if (!output.buffer) AF_ERROR("Could not allocate Metal sparse output", AF_ERR_NO_MEM);
    const size_t outputElements = size_t(outputDims.elements());
    const size_t nonzeros = valuesBytes / (type == c32 ? sizeof(float) * 2 : sizeof(float));
    SparseParams p{uint32_t(outputDims[0]), uint32_t(outputDims[1]),
                   uint32_t(outputStrides[1]), uint32_t(nonzeros)};
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    std::string zeroName = std::string("sparse_zero_") + suffix;
    auto* pipeline = metalPipeline(zeroName.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(output.buffer, output.offset, 0);
    encoder->setBytes(&p, sizeof(p), 1);
    auto width = std::min<NS::UInteger>(
        256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(outputElements, 1, 1),
                             MTL::Size(width, 1, 1));
    if (nonzeros > 0) {
        if (!values.buffer || !rows.buffer || !columns.buffer)
            AF_ERROR("Could not allocate Metal sparse inputs", AF_ERR_NO_MEM);
        const std::string scatterName =
            std::string(csr ? "sparse_csr_to_dense_" : "sparse_coo_to_dense_") +
            suffix;
        pipeline = metalPipeline(scatterName.c_str());
        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(values.buffer, values.offset, 0);
        encoder->setBuffer(rows.buffer, rows.offset, 1);
        encoder->setBuffer(columns.buffer, columns.offset, 2);
        encoder->setBuffer(output.buffer, output.offset, 3);
        encoder->setBytes(&p, sizeof(p), 4);
        const size_t scatterElements = csr ? outputDims[0] : nonzeros;
        width = std::min<NS::UInteger>(
            256, pipeline->maxTotalThreadsPerThreadgroup());
        encoder->dispatchThreads(MTL::Size(scatterElements, 1, 1),
                                 MTL::Size(width, 1, 1));
    }
    encoder->endEncoding();
    UNUSED(outputBytes);
    UNUSED(valuesBytes);
    UNUSED(rowsBytes);
    UNUSED(columnsBytes);
    finishSparseCommandBuffer(commandBuffer.get(),
                              "Metal sparse conversion failed");
}

void launchMetalSparseArithDense(
    BufferParam output, const size_t outputBytes, BufferParam values,
    const size_t valuesBytes, BufferParam rows, const size_t rowsBytes,
    BufferParam columns, const size_t columnsBytes, BufferParam rhs,
    const size_t rhsBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, const af::dim4& rhsStrides, const bool csr,
    const bool reverse, const unsigned operation, const af_dtype type) {
    const char* suffix = sparseTypeSuffix(type);
    if (!suffix) AF_ERROR("Metal sparse arithmetic type is not supported", AF_ERR_NOT_SUPPORTED);
    const size_t nonzeros = valuesBytes / (type == c32 ? sizeof(float) * 2 : sizeof(float));
    SparseArithParams p{uint32_t(nonzeros), uint32_t(outputDims[0]),
                        uint32_t(outputDims[1]), uint32_t(rhsStrides[1]),
                        uint32_t(csr), uint32_t(reverse), operation};
    if (!output.buffer || !values.buffer || !rows.buffer || !columns.buffer ||
        !rhs.buffer)
        AF_ERROR("Could not allocate Metal sparse arithmetic buffers", AF_ERR_NO_MEM);
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    const std::string name = std::string("sparse_arith_dense_") + suffix;
    auto* pipeline = metalPipeline(name.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(output.buffer, output.offset, 0);
    encoder->setBuffer(values.buffer, values.offset, 1);
    encoder->setBuffer(rows.buffer, rows.offset, 2);
    encoder->setBuffer(columns.buffer, columns.offset, 3);
    encoder->setBuffer(rhs.buffer, rhs.offset, 4);
    encoder->setBytes(&p, sizeof(p), 5);
    const auto width = std::min<NS::UInteger>(
        256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(nonzeros, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    UNUSED(outputBytes);
    UNUSED(rowsBytes);
    UNUSED(columnsBytes);
    UNUSED(rhsBytes);
    finishSparseCommandBuffer(commandBuffer.get(),
                              "Metal sparse-dense arithmetic failed");
}

void launchMetalSparseArithValues(
    BufferParam values, const size_t valuesBytes, BufferParam rows,
    const size_t rowsBytes, BufferParam columns, const size_t columnsBytes,
    BufferParam rhs, const size_t rhsBytes, const af::dim4& rhsDims,
    const af::dim4& rhsStrides, const bool csr, const bool reverse,
    const unsigned operation, const af_dtype type) {
    const char* suffix = sparseTypeSuffix(type);
    if (!suffix) AF_ERROR("Metal sparse arithmetic type is not supported", AF_ERR_NOT_SUPPORTED);
    const size_t nonzeros = valuesBytes / (type == c32 ? sizeof(float) * 2 : sizeof(float));
    SparseArithParams p{uint32_t(nonzeros), uint32_t(rhsDims[0]),
                        uint32_t(rhsDims[1]), uint32_t(rhsStrides[1]),
                        uint32_t(csr), uint32_t(reverse), operation};
    if (!values.buffer || !rows.buffer || !columns.buffer || !rhs.buffer)
        AF_ERROR("Could not allocate Metal sparse arithmetic buffers", AF_ERR_NO_MEM);
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    const std::string name = std::string("sparse_arith_values_") + suffix;
    auto* pipeline = metalPipeline(name.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(values.buffer, values.offset, 0);
    encoder->setBuffer(rows.buffer, rows.offset, 1);
    encoder->setBuffer(columns.buffer, columns.offset, 2);
    encoder->setBuffer(rhs.buffer, rhs.offset, 3);
    encoder->setBytes(&p, sizeof(p), 4);
    const auto width = std::min<NS::UInteger>(
        256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(nonzeros, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    UNUSED(rowsBytes);
    UNUSED(columnsBytes);
    UNUSED(rhsBytes);
    finishSparseCommandBuffer(commandBuffer.get(),
                              "Metal sparse arithmetic failed");
}

void launchMetalSparseCsrArithCount(
    BufferParam outputRows, const size_t outputRowsBytes, BufferParam lhsRows,
    const size_t lhsRowsBytes, BufferParam lhsColumns,
    const size_t lhsColumnsBytes, BufferParam rhsRows, const size_t rhsRowsBytes,
    BufferParam rhsColumns, const size_t rhsColumnsBytes, const size_t rows,
    const af_dtype type) {
    const char* suffix = sparseTypeSuffix(type);
    if (!suffix) AF_ERROR("Metal sparse arithmetic type is not supported", AF_ERR_NOT_SUPPORTED);
    if (!outputRows.buffer || !lhsRows.buffer || !lhsColumns.buffer ||
        !rhsRows.buffer || !rhsColumns.buffer)
        AF_ERROR("Could not allocate Metal sparse arithmetic buffers", AF_ERR_NO_MEM);
    SparseCsrArithParams p{uint32_t(rows), 0, 0, 0};
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    const std::string countName = std::string("sparse_csr_arith_count_") + suffix;
    auto* pipeline = metalPipeline(countName.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(outputRows.buffer, outputRows.offset, 0);
    encoder->setBuffer(lhsRows.buffer, lhsRows.offset, 1);
    encoder->setBuffer(lhsColumns.buffer, lhsColumns.offset, 2);
    encoder->setBuffer(rhsRows.buffer, rhsRows.offset, 3);
    encoder->setBuffer(rhsColumns.buffer, rhsColumns.offset, 4);
    encoder->setBytes(&p, sizeof(p), 5);
    auto width = std::min<NS::UInteger>(
        256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(rows, 1, 1), MTL::Size(width, 1, 1));
    const std::string prefixName = std::string("sparse_csr_arith_prefix_") + suffix;
    pipeline = metalPipeline(prefixName.c_str());
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(outputRows.buffer, outputRows.offset, 0);
    encoder->setBytes(&p, sizeof(p), 1);
    encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
    encoder->endEncoding();
    UNUSED(outputRowsBytes);
    UNUSED(lhsRowsBytes);
    UNUSED(lhsColumnsBytes);
    UNUSED(rhsRowsBytes);
    UNUSED(rhsColumnsBytes);
    finishSparseCommandBuffer(commandBuffer.get(),
                              "Metal sparse arithmetic count failed");
}

void launchMetalSparseCsrArith(
    BufferParam outputValues, const size_t outputValuesBytes,
    BufferParam outputColumns, const size_t outputColumnsBytes,
    BufferParam outputRows, const size_t outputRowsBytes, BufferParam lhsValues,
    const size_t lhsValuesBytes, BufferParam lhsRows, const size_t lhsRowsBytes,
    BufferParam lhsColumns, const size_t lhsColumnsBytes, BufferParam rhsValues,
    const size_t rhsValuesBytes, BufferParam rhsRows, const size_t rhsRowsBytes,
    BufferParam rhsColumns, const size_t rhsColumnsBytes, const size_t rows,
    const unsigned operation, const af_dtype type) {
    const char* suffix = sparseTypeSuffix(type);
    if (!suffix) AF_ERROR("Metal sparse arithmetic type is not supported", AF_ERR_NOT_SUPPORTED);
    if (!outputRows.buffer || !lhsValues.buffer || !lhsRows.buffer ||
        !lhsColumns.buffer || !rhsValues.buffer || !rhsRows.buffer ||
        !rhsColumns.buffer)
        AF_ERROR("Could not allocate Metal sparse arithmetic buffers", AF_ERR_NO_MEM);
    const size_t lhsNonzeros = lhsValuesBytes / (type == c32 ? sizeof(float) * 2 : sizeof(float));
    const size_t rhsNonzeros = rhsValuesBytes / (type == c32 ? sizeof(float) * 2 : sizeof(float));
    const size_t outputNonzeros = outputValuesBytes / (type == c32 ? sizeof(float) * 2 : sizeof(float));
    if (outputNonzeros > 0 && (!outputValues.buffer || !outputColumns.buffer))
        AF_ERROR("Could not allocate Metal sparse arithmetic output", AF_ERR_NO_MEM);
    SparseCsrArithParams p{uint32_t(rows), uint32_t(lhsNonzeros),
                           uint32_t(rhsNonzeros), operation};
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder)
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    if (outputNonzeros > 0) {
        const std::string name = std::string("sparse_csr_arith_merge_") + suffix;
        auto* pipeline = metalPipeline(name.c_str());
        encoder->setComputePipelineState(pipeline);
        encoder->setBuffer(outputValues.buffer, outputValues.offset, 0);
        encoder->setBuffer(outputColumns.buffer, outputColumns.offset, 1);
        encoder->setBuffer(outputRows.buffer, outputRows.offset, 2);
        encoder->setBuffer(lhsValues.buffer, lhsValues.offset, 3);
        encoder->setBuffer(lhsRows.buffer, lhsRows.offset, 4);
        encoder->setBuffer(lhsColumns.buffer, lhsColumns.offset, 5);
        encoder->setBuffer(rhsValues.buffer, rhsValues.offset, 6);
        encoder->setBuffer(rhsRows.buffer, rhsRows.offset, 7);
        encoder->setBuffer(rhsColumns.buffer, rhsColumns.offset, 8);
        encoder->setBytes(&p, sizeof(p), 9);
        const auto width = std::min<NS::UInteger>(
            256, pipeline->maxTotalThreadsPerThreadgroup());
        encoder->dispatchThreads(MTL::Size(rows, 1, 1),
                                 MTL::Size(width, 1, 1));
    }
    encoder->endEncoding();
    UNUSED(outputColumnsBytes);
    UNUSED(outputRowsBytes);
    UNUSED(lhsRowsBytes);
    UNUSED(lhsColumnsBytes);
    UNUSED(rhsRowsBytes);
    UNUSED(rhsColumnsBytes);
    finishSparseCommandBuffer(commandBuffer.get(),
                              "Metal sparse arithmetic merge failed");
}

void fftConvolveMultiplyMetal(Param<float> packed,
                              const af::dim4& signalDims,
                              const af::dim4& signalStrides,
                              const af::dim4& filterDims,
                              const af::dim4& filterStrides,
                              const AF_BATCH_KIND kind, const dim_t offset) {
    FFTConvolveParams p{};
    const af::dim4& outputDims = kind == AF_BATCH_RHS ? filterDims : signalDims;
    const af::dim4& outputStrides =
        kind == AF_BATCH_RHS ? filterStrides : signalStrides;
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(outputDims[i]);
        p.outputStrides[i] = uint64_t(outputStrides[i]);
        p.signalDims[i]    = uint64_t(signalDims[i]);
        p.signalStrides[i] = uint64_t(signalStrides[i]);
        p.filterDims[i]    = uint64_t(filterDims[i]);
        p.filterStrides[i] = uint64_t(filterStrides[i]);
    }
    p.offset = uint64_t(offset);
    p.kind   = uint32_t(kind);
    const size_t bytes = size_t(packed.dims().elements()) * sizeof(float);
    const size_t complexElements = size_t(outputDims.elements()) / 2;
    launchOutputKernel(packed.bufferParam(), bytes, complexElements, &p,
                       sizeof(p), "fftconvolve_multiply_float",
                       "FFT convolution multiply");
}

namespace {

size_t fftConvolveTypeSize(const af_dtype type) {
    switch (type) {
        case f32: return sizeof(float);
        case s32: return sizeof(int);
        case u32: return sizeof(uint);
        case s64: return sizeof(int64_t);
        case u64: return sizeof(uint64_t);
        case s8:
        case u8:
        case b8: return sizeof(char);
        case s16: return sizeof(short);
        case u16: return sizeof(unsigned short);
        default: return 0;
    }
}

void fillFFTConvolvePackParams(FFTConvolvePackParams& p,
                               const af::dim4& outputDims,
                               const af::dim4& outputStrides,
                               const af::dim4& inputDims,
                               const af::dim4& inputStrides) {
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(outputDims[i]);
        p.outputStrides[i] = uint64_t(outputStrides[i]);
        p.inputDims[i]     = uint64_t(inputDims[i]);
        p.inputStrides[i]  = uint64_t(inputStrides[i]);
    }
}

}  // namespace

void fftConvolvePackMetal(Param<float> output, const af::dim4& outputDims,
                          const af::dim4& outputStrides, BufferParam input,
                          const af::dim4& inputDims,
                          const af::dim4& inputStrides, const af_dtype type) {
    FFTConvolvePackParams p{};
    fillFFTConvolvePackParams(p, outputDims, outputStrides, inputDims,
                              inputStrides);
    const size_t outputBytes = size_t(output.dims().elements()) * sizeof(float);
    const char* function = nullptr;
    switch (type) {
        case f32: function = "fftconvolve_pack_float"; break;
        case s32: function = "fftconvolve_pack_int"; break;
        case u32: function = "fftconvolve_pack_uint"; break;
        case s64: function = "fftconvolve_pack_long"; break;
        case u64: function = "fftconvolve_pack_ulong"; break;
        case s8: function = "fftconvolve_pack_char"; break;
        case u8:
        case b8: function = "fftconvolve_pack_uchar"; break;
        case s16: function = "fftconvolve_pack_short"; break;
        case u16: function = "fftconvolve_pack_ushort"; break;
        default: AF_ERROR("Unsupported Metal FFT convolution input type",
                           AF_ERR_NOT_SUPPORTED);
    }
    launchSingleInputKernel(output.bufferParam(), outputBytes, outputDims,
                            input, size_t(inputDims.elements()) *
                                       fftConvolveTypeSize(type),
                            &p, sizeof(p), function, "FFT convolution pack");
}

void fftConvolvePadMetal(Param<float> output, const af::dim4& outputDims,
                         const af::dim4& outputStrides, BufferParam input,
                         const af::dim4& inputDims,
                         const af::dim4& inputStrides, const dim_t offset,
                         const af_dtype type) {
    FFTConvolvePadParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(outputDims[i]);
        p.outputStrides[i] = uint64_t(outputStrides[i]);
        p.inputDims[i]     = uint64_t(inputDims[i]);
        p.inputStrides[i]  = uint64_t(inputStrides[i]);
    }
    p.offset = uint64_t(offset);
    const size_t outputBytes = size_t(output.dims().elements()) * sizeof(float);
    const char* function = nullptr;
    switch (type) {
        case f32: function = "fftconvolve_pad_float"; break;
        case s32: function = "fftconvolve_pad_int"; break;
        case u32: function = "fftconvolve_pad_uint"; break;
        case s64: function = "fftconvolve_pad_long"; break;
        case u64: function = "fftconvolve_pad_ulong"; break;
        case s8: function = "fftconvolve_pad_char"; break;
        case u8:
        case b8: function = "fftconvolve_pad_uchar"; break;
        case s16: function = "fftconvolve_pad_short"; break;
        case u16: function = "fftconvolve_pad_ushort"; break;
        default: AF_ERROR("Unsupported Metal FFT convolution input type",
                           AF_ERR_NOT_SUPPORTED);
    }
    launchSingleInputKernel(output.bufferParam(), outputBytes, outputDims,
                            input, size_t(inputDims.elements()) *
                                       fftConvolveTypeSize(type),
                            &p, sizeof(p), function, "FFT convolution padding");
}

void fftConvolveReorderMetal(BufferParam output, BufferParam packed,
                             const af::dim4& outputDims,
                             const af::dim4& outputStrides,
                             const af::dim4& inputDims,
                             const af::dim4& inputStrides,
                             const af::dim4& filterDims,
                             const dim_t filterOffset,
                             const dim_t signalHalfDim0, const dim_t fftScale,
                             const AF_BATCH_KIND kind, const af_dtype type,
                             const int rank, const bool expand) {
    FFTConvolveReorderParams p{};
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i] = uint64_t(outputDims[i]);
        p.outputStrides[i] = uint64_t(outputStrides[i]);
        p.inputDims[i] = uint64_t(inputDims[i]);
        p.inputStrides[i] = uint64_t(inputStrides[i]);
        p.filterDims[i] = uint64_t(filterDims[i]);
    }
    p.filterOffset = uint64_t(filterOffset);
    p.signalHalfDim0 = uint64_t(signalHalfDim0);
    p.fftScale = uint64_t(fftScale);
    p.kind = uint32_t(kind);
    const char* base = nullptr;
    switch (type) {
        case f32: base = "float"; break;
        case s32: base = "int"; break;
        case u32: base = "uint"; break;
        case s64: base = "long"; break;
        case u64: base = "ulong"; break;
        case s8: base = "char"; break;
        case u8:
        case b8: base = "uchar"; break;
        case s16: base = "short"; break;
        case u16: base = "ushort"; break;
        default: AF_ERROR("Unsupported Metal FFT convolution output type",
                           AF_ERR_NOT_SUPPORTED);
    }
    std::string function = "fftconvolve_reorder_" + std::string(base) +
                           "_" + std::to_string(rank) +
                           (expand ? "_expand" : "_same");
    const size_t outputBytes = size_t(outputDims.elements()) *
                               fftConvolveTypeSize(type);
    const size_t packedBytes = size_t(inputDims.elements()) * sizeof(float);
    launchSingleInputKernel(output, outputBytes, outputDims,
                            packed, packedBytes, &p, sizeof(p), function.c_str(),
                            "FFT convolution reorder");
}

void orbCentroidMetal(BufferParam x, BufferParam y, BufferParam orientation,
                      const unsigned features, CParam<float> image,
                      const unsigned patchSize) {
    OrbParams p{features, uint32_t(image.dims(0)), uint32_t(image.dims(1)),
                patchSize};
    const size_t featureBytes = size_t(features) * sizeof(float);
    const size_t imageBytes   = size_t(image.dims().elements()) * sizeof(float);
    launchThreeInputKernel(orientation, featureBytes, x, featureBytes, y,
                           featureBytes, image.bufferParam(), imageBytes, &p,
                           sizeof(p), features, "orb_centroid_float",
                           "ORB centroid orientation");
}

void orbHarrisMetal(BufferParam xOutput, BufferParam yOutput,
                    BufferParam scoreOutput, BufferParam xInput,
                    BufferParam yInput, const unsigned features,
                    unsigned* count, CParam<float> image,
                    const unsigned blockSize, const float kThreshold,
                    const unsigned patchSize) {
    if (!xOutput.buffer || !yOutput.buffer || !scoreOutput.buffer ||
        !xInput.buffer || !yInput.buffer || !image.bufferParam().buffer ||
        !count) {
        AF_ERROR("Could not allocate Metal ORB Harris buffers", AF_ERR_NO_MEM);
    }

    OrbHarrisParams p{features, uint32_t(image.dims(0)),
                      uint32_t(image.dims(1)), blockSize, patchSize,
                      kThreshold};
    auto countBuffer = NS::TransferPtr(metalDevice()->newBuffer(
        sizeof(uint32_t), MTL::ResourceStorageModeShared));
    if (!countBuffer) {
        AF_ERROR("Could not allocate Metal ORB Harris count buffer",
                 AF_ERR_NO_MEM);
    }
    *static_cast<uint32_t*>(countBuffer->contents()) = 0;

    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }
    auto* pipeline = metalPipeline("orb_harris_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(xInput.buffer, xInput.offset, 0);
    encoder->setBuffer(yInput.buffer, yInput.offset, 1);
    const BufferParam imageParam = image.bufferParam();
    encoder->setBuffer(imageParam.buffer, imageParam.offset, 2);
    encoder->setBuffer(xOutput.buffer, xOutput.offset, 3);
    encoder->setBuffer(yOutput.buffer, yOutput.offset, 4);
    encoder->setBuffer(scoreOutput.buffer, scoreOutput.offset, 5);
    encoder->setBuffer(countBuffer.get(), 0, 6);
    encoder->setBytes(&p, sizeof(p), 7);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(features, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    syncCommandQueue();
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal ORB Harris response failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
    *count = *static_cast<uint32_t*>(countBuffer->contents());
}

void orbExtractMetal(BufferParam descriptor, BufferParam x,
                     BufferParam y, BufferParam orientation, BufferParam size,
                     BufferParam pattern, const unsigned features,
                     CParam<float> image, const float scale,
                     const unsigned patchSize) {
    if (!descriptor.buffer || !x.buffer || !y.buffer || !orientation.buffer ||
        !size.buffer || !pattern.buffer || !image.bufferParam().buffer) {
        AF_ERROR("Could not allocate Metal ORB descriptor buffers",
                 AF_ERR_NO_MEM);
    }
    OrbExtractParams p{features, uint32_t(image.dims(0)),
                       uint32_t(image.dims(1)), patchSize, scale};
    const BufferParam imageParam = image.bufferParam();
    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }
    auto* pipeline = metalPipeline("orb_extract_float");
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(descriptor.buffer, descriptor.offset, 0);
    encoder->setBuffer(x.buffer, x.offset, 1);
    encoder->setBuffer(y.buffer, y.offset, 2);
    encoder->setBuffer(orientation.buffer, orientation.offset, 3);
    encoder->setBuffer(size.buffer, size.offset, 4);
    encoder->setBuffer(pattern.buffer, pattern.offset, 5);
    encoder->setBuffer(imageParam.buffer, imageParam.offset, 6);
    encoder->setBytes(&p, sizeof(p), 7);
    const auto width =
        std::min<NS::UInteger>(256, pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(features, 1, 1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
}

void orbKeepMetal(BufferParam xOutput, BufferParam yOutput,
                  BufferParam scoreOutput, BufferParam xInput,
                  BufferParam yInput, BufferParam scoreInput,
                  BufferParam scoreIndex, const unsigned features) {
    HarrisKeepParams p{features};
    const size_t outputBytes = size_t(features) * sizeof(float);
    launchFourInputThreeOutputKernel(
        xOutput, yOutput, scoreOutput, outputBytes, xInput, outputBytes,
        yInput, outputBytes, scoreInput, outputBytes, scoreIndex,
        size_t(features) * sizeof(unsigned), &p, sizeof(p), features,
        "harris_keep_corners", "ORB feature selection");
}

void siftSubtractMetal(Array<float>& output, const Array<float>& first,
                       const Array<float>& second) {
    const size_t elements = size_t(output.elements());
    const size_t bytes    = elements * sizeof(float);
    SiftParams p{elements};
    launchTwoInputKernel(output.bufferParam(), bytes, first.bufferParam(), bytes,
                         second.bufferParam(), bytes, &p, sizeof(p), elements,
                         "sift_subtract_float",
                         "SIFT pyramid subtraction");
}

void siftExtremaFlagsMetal(Array<unsigned char>& output,
                           const Array<float>& previous,
                           const Array<float>& center,
                           const Array<float>& next,
                           const float threshold) {
    const size_t elements = size_t(center.elements());
    if (elements == 0) return;

    struct SiftExtremaParams {
        uint64_t elements;
        uint64_t width;
        uint64_t height;
        float threshold;
    } p{uint64_t(elements), uint64_t(center.dims()[0]),
        uint64_t(center.dims()[1]), threshold};

    launchThreeInputKernel(
        output.bufferParam(), elements * sizeof(unsigned char),
        previous.bufferParam(), elements * sizeof(float), center.bufferParam(),
        elements * sizeof(float), next.bufferParam(), elements * sizeof(float),
        &p, sizeof(p), elements, "sift_extrema_flags_float",
        "SIFT extrema detection");
}

void siftDescriptorMetal(Array<float>& output, const Array<float>& features,
                         const Array<unsigned>& outputIndices,
                         const Array<float>& image, const unsigned count,
                         const float scale) {
    if (count == 0) return;
    if (!output.bufferParam().buffer || !features.bufferParam().buffer ||
        !outputIndices.bufferParam().buffer || !image.bufferParam().buffer) {
        AF_ERROR("Could not allocate Metal SIFT descriptor buffers",
                 AF_ERR_NO_MEM);
    }

    struct SiftDescriptorParams {
        uint32_t featureCount;
        uint32_t rows;
        uint32_t columns;
        float scale;
    } p{count, uint32_t(image.dims()[0]), uint32_t(image.dims()[1]), scale};

    const BufferParam imageParam = image.bufferParam();
    const BufferParam featureParam = features.bufferParam();
    const BufferParam indexParam = outputIndices.bufferParam();
    const BufferParam outputParam = output.bufferParam();

    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal command encoder", AF_ERR_RUNTIME);
    }
    auto* pipeline = metalPipeline("sift_descriptor_float");
    if (pipeline->maxTotalThreadsPerThreadgroup() < 128) {
        AF_ERROR("Metal SIFT descriptor requires 128 threads per threadgroup",
                 AF_ERR_NOT_SUPPORTED);
    }
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(imageParam.buffer, imageParam.offset, 0);
    encoder->setBuffer(featureParam.buffer, featureParam.offset, 1);
    encoder->setBuffer(indexParam.buffer, indexParam.offset, 2);
    encoder->setBuffer(outputParam.buffer, outputParam.offset, 3);
    encoder->setBytes(&p, sizeof(p), 4);
    encoder->dispatchThreadgroups(MTL::Size(1, count, 1),
                                 MTL::Size(128, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal SIFT descriptor failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void arrayAddMetal(Param<float> output, CParam<float> left,
                   CParam<float> right) {
    ArrayAddParams p{};
    size_t leftElements = 1, rightElements = 1;
    for (int i = 0; i < 4; ++i) {
        p.outputDims[i]    = uint64_t(output.dims(i));
        p.outputStrides[i] = uint64_t(output.strides(i));
        p.leftDims[i]      = uint64_t(left.dims(i));
        p.leftStrides[i]   = uint64_t(left.strides(i));
        p.rightDims[i]     = uint64_t(right.dims(i));
        p.rightStrides[i]  = uint64_t(right.strides(i));
        leftElements += size_t(left.dims(i) - 1) * size_t(left.strides(i));
        rightElements += size_t(right.dims(i) - 1) * size_t(right.strides(i));
    }
    launchTwoInputKernel(
        output.bufferParam(), size_t(output.dims().elements()) * sizeof(float),
        left.bufferParam(), leftElements * sizeof(float), right.bufferParam(),
        rightElements * sizeof(float), &p, sizeof(p),
        size_t(output.dims().elements()), "array_add_float", "array add");
}

bool supportsMetalExampleFunction(const af_dtype type) noexcept {
    return exampleFunctionName(type) != nullptr;
}

void launchMetalExampleFunction(
    BufferParam output, const size_t outputBytes, const af::dim4& dims,
    const af::dim4& outputStrides, BufferParam left, const size_t leftBytes,
    const af::dim4& leftStrides, BufferParam right, const size_t rightBytes,
    const af::dim4& rightStrides, const af_dtype type) {
    ExampleFunctionParams p{};
    for (int i = 0; i < 4; ++i) {
        p.dims[i]          = static_cast<uint64_t>(dims[i]);
        p.outputStrides[i] = static_cast<int64_t>(outputStrides[i]);
        p.leftStrides[i]   = static_cast<int64_t>(leftStrides[i]);
        p.rightStrides[i]  = static_cast<int64_t>(rightStrides[i]);
    }
    launchTwoInputKernel(output, outputBytes, left, leftBytes, right, rightBytes,
                         &p, sizeof(p), static_cast<size_t>(dims.elements()),
                         exampleFunctionName(type), "example function");
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
