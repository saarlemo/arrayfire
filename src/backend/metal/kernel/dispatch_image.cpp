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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalHistogram(const af_dtype type) noexcept {
    return histogramFunctionName(type) != nullptr;
}

void launchMetalHistogram(BufferParam output, const size_t outputBytes,
                          const af::dim4& outputStrides, BufferParam input,
                          const size_t inputBytes, const af::dim4& inputDims,
                          const af::dim4& inputStrides, const unsigned bins,
                          const double minValue, const double maxValue,
                          const bool linear, const af_dtype type) {
    HistogramParams params{};
    for (int i = 0; i < 4; ++i) {
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.bins     = bins;
    params.minValue = static_cast<float>(minValue);
    params.maxValue = static_cast<float>(maxValue);
    params.linear   = linear;
    const af::dim4 dispatchDims(1, 1, inputDims[2], inputDims[3]);
    launchSingleInputKernel(output, outputBytes, dispatchDims, input,
                            inputBytes, &params, sizeof(params),
                            histogramFunctionName(type), "histogram");
}

bool supportsMetalMorph(const af_dtype type) noexcept {
    return morphFunctionName(type) != nullptr;
}

void launchMetalMorph(BufferParam output, const size_t outputBytes,
                      const af::dim4& outputDims, const af::dim4& outputStrides,
                      BufferParam input, const size_t inputBytes,
                      const af::dim4& inputDims, const af::dim4& inputStrides,
                      BufferParam mask, const size_t maskBytes,
                      const af::dim4& maskDims, const af::dim4& maskStrides,
                      const bool dilation, const bool volume,
                      const af_dtype type) {
    MorphParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputDims[i]     = static_cast<uint64_t>(inputDims[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
        params.maskDims[i]      = static_cast<uint64_t>(maskDims[i]);
        params.maskStrides[i]   = static_cast<uint64_t>(maskStrides[i]);
    }
    params.dilation = dilation;
    params.volume   = volume;

    launchTwoInputKernel(
        output, outputBytes, input, inputBytes, mask, maskBytes, &params,
        sizeof(params), static_cast<size_t>(outputDims.elements()),
        morphFunctionName(type), "morph");
}

bool supportsMetalNearestNeighbour(const af_dtype inputType,
                                   const af_dtype outputType,
                                   const af_match_type distanceType) noexcept {
    return nearestNeighbourFunctionName(inputType, outputType, distanceType) !=
           nullptr;
}

void launchMetalNearestNeighbour(
    BufferParam output, const size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam query, const size_t queryBytes,
    const af::dim4& queryDims, const af::dim4& queryStrides, BufferParam train,
    const size_t trainBytes, const af::dim4& trainDims,
    const af::dim4& trainStrides, const unsigned distanceDimension,
    const af_match_type distanceType, const af_dtype inputType,
    const af_dtype outputType) {
    NearestNeighbourParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.queryDims[i]     = static_cast<uint64_t>(queryDims[i]);
        params.queryStrides[i]  = static_cast<uint64_t>(queryStrides[i]);
        params.trainDims[i]     = static_cast<uint64_t>(trainDims[i]);
        params.trainStrides[i]  = static_cast<uint64_t>(trainStrides[i]);
    }
    params.distanceDimension = distanceDimension;
    params.distanceType      = static_cast<uint32_t>(distanceType);

    launchTwoInputKernel(
        output, outputBytes, query, queryBytes, train, trainBytes, &params,
        sizeof(params), static_cast<size_t>(outputDims.elements()),
        nearestNeighbourFunctionName(inputType, outputType, distanceType),
        "nearest neighbour");
}

bool supportsMetalIir(const af_dtype type) noexcept {
    return iirFunctionName(type) != nullptr;
}

void launchMetalIir(BufferParam output, const size_t outputBytes,
                    const af::dim4& outputDims, const af::dim4& outputStrides,
                    BufferParam coefficients, const size_t coefficientBytes,
                    const af::dim4& coefficientStrides, BufferParam feedback,
                    const size_t feedbackBytes, const af::dim4& feedbackDims,
                    const af::dim4& feedbackStrides, const bool feedbackBatched,
                    const af_dtype type) {
    IirParams params{};
    for (int i = 0; i < 4; ++i) {
        params.outputDims[i]    = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.coefficientStrides[i] =
            static_cast<uint64_t>(coefficientStrides[i]);
        params.feedbackDims[i]    = static_cast<uint64_t>(feedbackDims[i]);
        params.feedbackStrides[i] = static_cast<uint64_t>(feedbackStrides[i]);
    }
    params.feedbackBatched = feedbackBatched;
    const size_t series =
        static_cast<size_t>(outputDims[1] * outputDims[2] * outputDims[3]);
    launchTwoInputKernel(output, outputBytes, coefficients, coefficientBytes,
                         feedback, feedbackBytes, &params, sizeof(params),
                         series, iirFunctionName(type), "IIR");
}

bool supportsMetalDot(const af_dtype type) noexcept {
    return dotFunctionName(type) != nullptr;
}

void launchMetalDot(BufferParam output, const size_t outputBytes,
                    BufferParam lhs,
                    const size_t lhsBytes, const af::dim4& lhsDims,
                    const af::dim4& lhsStrides, BufferParam rhs,
                    const size_t rhsBytes, const af::dim4& rhsStrides,
                    const af_mat_prop lhsOption, const af_mat_prop rhsOption,
                    const af_dtype type) {
    DotParams params{};
    params.length       = static_cast<uint64_t>(lhsDims[0]);
    params.lhsStride    = static_cast<uint64_t>(lhsStrides[0]);
    params.rhsStride    = static_cast<uint64_t>(rhsStrides[0]);
    params.conjugateLhs = lhsOption == AF_MAT_CONJ;
    params.conjugateRhs = rhsOption == AF_MAT_CONJ;
    launchTwoInputKernel(output, outputBytes, lhs, lhsBytes, rhs, rhsBytes,
                         &params, sizeof(params), 1, dotFunctionName(type),
                         "dot");
}

bool supportsMetalBilateral(const af_dtype inputType,
                            const af_dtype outputType) noexcept {
    return bilateralFunctionName(inputType, outputType) != nullptr;
}

void launchMetalBilateral(BufferParam output, const size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, BufferParam input,
                          const size_t inputBytes, const af::dim4& inputStrides,
                          const float spatialSigma, const float chromaticSigma,
                          const af_dtype inputType, const af_dtype outputType) {
    BilateralParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(outputDims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.spatialSigma   = spatialSigma;
    params.chromaticSigma = chromaticSigma;
    launchSingleInputKernel(output, outputBytes, outputDims, input, inputBytes,
                            &params, sizeof(params),
                            bilateralFunctionName(inputType, outputType),
                            "bilateral");
}

bool supportsMetalMeanshift(const af_dtype type) noexcept {
    return meanshiftFunctionName(type) != nullptr;
}

void launchMetalMeanshift(BufferParam output, const size_t outputBytes,
                          const af::dim4& dims, const af::dim4& outputStrides,
                          BufferParam input, const size_t inputBytes,
                          const af::dim4& inputStrides,
                          const float spatialSigma, const float chromaticSigma,
                          const unsigned iterations, const bool color,
                          const af_dtype type) {
    MeanshiftParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.spatialSigma   = spatialSigma;
    params.chromaticSigma = chromaticSigma;
    params.iterations     = iterations;
    params.color          = color;
    af::dim4 dispatchDims = dims;
    if (color) dispatchDims[2] = 1;
    launchSingleInputKernel(output, outputBytes, dispatchDims, input,
                            inputBytes, &params, sizeof(params),
                            meanshiftFunctionName(type), "meanshift");
}

bool supportsMetalMedfilt(const af_dtype type, const dim_t windowLength,
                          const dim_t windowWidth) noexcept {
    return medfiltFunctionName(type) != nullptr && windowLength > 0 &&
           windowWidth > 0;
}

void launchMetalMedfilt(BufferParam output, const size_t outputBytes,
                        const af::dim4& dims, const af::dim4& outputStrides,
                        BufferParam input, const size_t inputBytes,
                        const af::dim4& inputStrides, const dim_t windowLength,
                        const dim_t windowWidth, const af_border_type padding,
                        const bool oneDimensional, const af_dtype type) {
    MedfiltParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.windowLength   = static_cast<uint32_t>(windowLength);
    params.windowWidth    = static_cast<uint32_t>(windowWidth);
    params.padding        = static_cast<uint32_t>(padding);
    params.oneDimensional = oneDimensional;
    launchSingleInputKernel(output, outputBytes, dims, input, inputBytes,
                            &params, sizeof(params), medfiltFunctionName(type),
                            "median filter");
}
bool supportsMetalMatchTemplate(const af_dtype inputType,
                                const af_dtype outputType) noexcept {
    return matchTemplateFunctionName(inputType, outputType) != nullptr;
}
void launchMetalMatchTemplate(
    BufferParam output, const size_t outputBytes, const af::dim4& dims,
    const af::dim4& outputStrides, BufferParam search, const size_t searchBytes,
    const af::dim4& searchStrides, BufferParam templ,
    const size_t templateBytes, const af::dim4& templateDims,
    const af::dim4& templateStrides, const af_match_type matchType,
    const af_dtype inputType, const af_dtype outputType) {
    MatchTemplateParams p{};
    for (int i = 0; i < 4; ++i) {
        p.dims[i]            = uint64_t(dims[i]);
        p.outputStrides[i]   = uint64_t(outputStrides[i]);
        p.searchStrides[i]   = uint64_t(searchStrides[i]);
        p.templateDims[i]    = uint64_t(templateDims[i]);
        p.templateStrides[i] = uint64_t(templateStrides[i]);
    }
    p.matchType = uint32_t(matchType);
    launchTwoInputKernel(output, outputBytes, search, searchBytes, templ,
                         templateBytes, &p, sizeof(p), size_t(dims.elements()),
                         matchTemplateFunctionName(inputType, outputType),
                         "match template");
}
bool supportsMetalRotate(const af_dtype t, const af_interp_type m) noexcept {
    return rotateFunctionName(t) &&
           (m == AF_INTERP_NEAREST || m == AF_INTERP_LOWER ||
            m == AF_INTERP_BILINEAR || m == AF_INTERP_BILINEAR_COSINE ||
            m == AF_INTERP_BICUBIC || m == AF_INTERP_BICUBIC_SPLINE);
}
void launchMetalRotate(BufferParam o, size_t ob, const af::dim4& od,
                       const af::dim4& os, BufferParam i, size_t ib,
                       const af::dim4& id, const af::dim4& is, float theta,
                       af_interp_type method, af_dtype type) {
    RotateParams p{};
    for (int k = 0; k < 4; ++k) {
        p.odims[k]    = uint64_t(od[k]);
        p.ostrides[k] = uint64_t(os[k]);
        p.idims[k]    = uint64_t(id[k]);
        p.istrides[k] = uint64_t(is[k]);
    }
    const float cosine = std::cos(-theta);
    const float sine   = std::sin(-theta);
    const float nx     = 0.5f * static_cast<float>(id[0] - 1);
    const float ny     = 0.5f * static_cast<float>(id[1] - 1);
    const float mx     = 0.5f * static_cast<float>(od[0] - 1);
    const float my     = 0.5f * static_cast<float>(od[1] - 1);
    const float tx     = -(mx * cosine - my * sine - nx);
    const float ty     = -(mx * sine + my * cosine - ny);
    p.transform[0]     = std::round(cosine * 1000.0f) / 1000.0f;
    p.transform[1]     = std::round(-sine * 1000.0f) / 1000.0f;
    p.transform[2]     = std::round(tx * 1000.0f) / 1000.0f;
    p.transform[3]     = std::round(sine * 1000.0f) / 1000.0f;
    p.transform[4]     = std::round(cosine * 1000.0f) / 1000.0f;
    p.transform[5]     = std::round(ty * 1000.0f) / 1000.0f;
    p.method = uint32_t(method);
    launchSingleInputKernel(o, ob, od, i, ib, &p, sizeof(p),
                            rotateFunctionName(type), "rotate");
}
bool supportsMetalLu(const af_dtype t) noexcept {
    return luFunctionName(t) != nullptr;
}

bool supportsMetalLuFactor(const af_dtype t) noexcept {
    return luFactorFunctionName(t) != nullptr;
}

void launchMetalLuFactor(BufferParam input, size_t inputBytes,
                         const af::dim4& dims, const af::dim4& strides,
                         BufferParam pivot, size_t pivotBytes,
                         const af::dim4& pivotDims, const af_dtype type) {
    UNUSED(inputBytes);
    UNUSED(pivotBytes);
    UNUSED(pivotDims);
    if (!input.buffer || !pivot.buffer) {
        AF_ERROR("Could not allocate Metal LU buffers", AF_ERR_NO_MEM);
    }
    if (dims[2] != 1 || dims[3] != 1) {
        AF_ERROR("Metal LU currently supports one matrix at a time",
                 AF_ERR_NOT_SUPPORTED);
    }
    const char* functionName = luFactorFunctionName(type);
    if (!functionName) {
        AF_ERROR("Input type is not supported by the Metal LU kernel",
                 AF_ERR_NOT_SUPPORTED);
    }

    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal LU command encoder", AF_ERR_RUNTIME);
    }

    const std::string base(functionName);
    const std::string pivotName = base + "_pivot";
    const std::string swapName = base + "_swap";
    const std::string updateName = base + "_update";
    const std::string factorName = base + "_factor";
    auto* pivotPipeline = metalPipeline(pivotName.c_str());
    auto* swapPipeline = metalPipeline(swapName.c_str());
    auto* updatePipeline = metalPipeline(updateName.c_str());
    auto* factorPipeline = metalPipeline(factorName.c_str());
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(pivot.buffer, pivot.offset, 1);

    const uint64_t rows = static_cast<uint64_t>(dims[0]);
    const uint64_t columns = static_cast<uint64_t>(dims[1]);
    const uint64_t steps = std::min(rows, columns);
    const NS::UInteger vectorWidth = std::min<NS::UInteger>(64,
        swapPipeline->maxTotalThreadsPerThreadgroup());
    for (uint64_t step = 0; step < steps; ++step) {
        LuFactorParams params{};
        params.rows = rows;
        params.columns = columns;
        params.stride0 = static_cast<uint64_t>(strides[0]);
        params.stride1 = static_cast<uint64_t>(strides[1]);
        params.step = static_cast<uint32_t>(step);
        encoder->setBytes(&params, sizeof(params), 2);

        encoder->setComputePipelineState(pivotPipeline);
        encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);

        encoder->setComputePipelineState(swapPipeline);
        encoder->dispatchThreads(MTL::Size(columns, 1, 1),
                                 MTL::Size(vectorWidth, 1, 1));
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);

        encoder->setComputePipelineState(updatePipeline);
        encoder->dispatchThreads(MTL::Size(columns, rows, 1),
                                 MTL::Size(8, 8, 1));
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);

        encoder->setComputePipelineState(factorPipeline);
        encoder->dispatchThreads(MTL::Size(rows, 1, 1),
                                 MTL::Size(vectorWidth, 1, 1));
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal LU factorization failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

bool supportsMetalQR(const af_dtype type) noexcept {
    return qrFunctionName(type) != nullptr;
}

void launchMetalQRFactor(BufferParam input, size_t inputBytes,
                         const af::dim4& dims, const af::dim4& strides,
                         BufferParam tau, size_t tauBytes,
                         const af_dtype type) {
    UNUSED(inputBytes);
    UNUSED(tauBytes);
    if (!input.buffer || !tau.buffer) {
        AF_ERROR("Could not allocate Metal QR buffers", AF_ERR_NO_MEM);
    }
    if (dims[2] != 1 || dims[3] != 1) {
        AF_ERROR("Metal QR currently supports one matrix at a time",
                 AF_ERR_NOT_SUPPORTED);
    }
    const char* functionName = qrFunctionName(type);
    if (!functionName) {
        AF_ERROR("Input type is not supported by the Metal QR kernel",
                 AF_ERR_NOT_SUPPORTED);
    }

    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal QR command encoder", AF_ERR_RUNTIME);
    }

    const std::string base(functionName);
    auto* factorPipeline = metalPipeline((base + "_factor").c_str());
    auto* applyPipeline  = metalPipeline((base + "_apply").c_str());
    encoder->setBuffer(input.buffer, input.offset, 0);
    encoder->setBuffer(tau.buffer, tau.offset, 1);

    const uint64_t rows = static_cast<uint64_t>(dims[0]);
    const uint64_t columns = static_cast<uint64_t>(dims[1]);
    const uint64_t steps = std::min(rows, columns);
    const NS::UInteger width = std::min<NS::UInteger>(
        64, applyPipeline->maxTotalThreadsPerThreadgroup());
    for (uint64_t step = 0; step < steps; ++step) {
        QRFactorParams params{};
        params.rows    = rows;
        params.columns = columns;
        params.stride0 = static_cast<uint64_t>(strides[0]);
        params.stride1 = static_cast<uint64_t>(strides[1]);
        params.step    = static_cast<uint32_t>(step);
        encoder->setBytes(&params, sizeof(params), 2);

        encoder->setComputePipelineState(factorPipeline);
        encoder->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);

        encoder->setComputePipelineState(applyPipeline);
        encoder->dispatchThreads(MTL::Size(columns, 1, 1),
                                 MTL::Size(width, 1, 1));
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal QR factorization failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchMetalQRGenerate(BufferParam output, size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, BufferParam packed,
                           size_t packedBytes, const af::dim4& packedDims,
                           const af::dim4& packedStrides, BufferParam tau,
                           size_t tauBytes, const af_dtype type) {
    UNUSED(outputBytes);
    UNUSED(packedBytes);
    UNUSED(tauBytes);
    if (!output.buffer || !packed.buffer || !tau.buffer) {
        AF_ERROR("Could not allocate Metal QR buffers", AF_ERR_NO_MEM);
    }
    if (outputDims[2] != 1 || outputDims[3] != 1 || packedDims[2] != 1 ||
        packedDims[3] != 1) {
        AF_ERROR("Metal QR currently supports one matrix at a time",
                 AF_ERR_NOT_SUPPORTED);
    }
    const char* functionName = qrFunctionName(type);
    if (!functionName) {
        AF_ERROR("Input type is not supported by the Metal QR kernel",
                 AF_ERR_NOT_SUPPORTED);
    }

    auto commandBuffer = NS::RetainPtr(metalCommandQueue()->commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal QR command encoder", AF_ERR_RUNTIME);
    }

    const std::string pipelineName = std::string(functionName) + "_generate";
    auto* pipeline = metalPipeline(pipelineName.c_str());
    encoder->setBuffer(output.buffer, output.offset, 0);
    encoder->setBuffer(packed.buffer, packed.offset, 1);
    encoder->setBuffer(tau.buffer, tau.offset, 2);

    const uint64_t rows = static_cast<uint64_t>(outputDims[0]);
    const uint64_t steps = std::min<uint64_t>(
        rows, static_cast<uint64_t>(packedDims[1]));
    const NS::UInteger width = std::min<NS::UInteger>(
        64, pipeline->maxTotalThreadsPerThreadgroup());
    for (uint64_t step = steps; step-- > 0;) {
        QRGenerateParams params{};
        params.rows          = rows;
        params.columns       = steps;
        params.outputStride0 = static_cast<uint64_t>(outputStrides[0]);
        params.outputStride1 = static_cast<uint64_t>(outputStrides[1]);
        params.packedStride0 = static_cast<uint64_t>(packedStrides[0]);
        params.packedStride1 = static_cast<uint64_t>(packedStrides[1]);
        params.step           = static_cast<uint32_t>(step);
        encoder->setBytes(&params, sizeof(params), 3);
        encoder->setComputePipelineState(pipeline);
        encoder->dispatchThreads(MTL::Size(rows, 1, 1),
                                 MTL::Size(width, 1, 1));
        encoder->memoryBarrier(MTL::BarrierScopeBuffers);
    }
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
    if (commandBuffer->status() == MTL::CommandBufferStatusError) {
        const std::string message = metalErrorDescription(
            commandBuffer->error(), "Metal Q generation failed");
        AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
    }
}

void launchMetalLuPart(BufferParam o, size_t ob, const af::dim4& od,
                       const af::dim4& os, BufferParam i, size_t ib,
                       const af::dim4& id, const af::dim4& is, bool lower,
                       af_dtype type) {
    LuParams p{};
    for (int k = 0; k < 4; ++k) {
        p.odims[k]    = uint64_t(od[k]);
        p.ostrides[k] = uint64_t(os[k]);
        p.idims[k]    = uint64_t(id[k]);
        p.istrides[k] = uint64_t(is[k]);
    }
    p.lower = lower;
    launchSingleInputKernel(o, ob, od, i, ib, &p, sizeof(p),
                            luFunctionName(type), "LU split");
}
void launchMetalConvertPivot(BufferParam o, size_t ob, const af::dim4& od,
                             BufferParam pivot, size_t pb,
                             const af::dim4& pd) {
    PivotParams p{uint64_t(od.elements()), uint64_t(pd.elements())};
    launchTwoInputKernel(o, ob, o, ob, pivot, pb, &p, sizeof(p), 1,
                         "convert_pivot", "pivot conversion");
}

void launchMetalFloodFill(
    BufferParam output, const size_t outputBytes, const af::dim4& imageDims,
    const af::dim4& imageStrides, BufferParam image, const size_t imageBytes,
    BufferParam seedX, const size_t seedXBytes, const af::dim4& seedDims,
    const af::dim4& seedXStrides, BufferParam seedY, const size_t seedYBytes,
    const af::dim4& seedYStrides, const void* newValue, const void* lower,
    const void* upper, const size_t valueBytes, const af_dtype type) {
    FloodFillParams params{};
    for (int i = 0; i < 4; ++i) {
        params.imageDims[i]    = static_cast<uint64_t>(imageDims[i]);
        params.imageStrides[i] = static_cast<uint64_t>(imageStrides[i]);
        params.seedDims[i]     = static_cast<uint64_t>(seedDims[i]);
        params.seedXStrides[i] = static_cast<uint64_t>(seedXStrides[i]);
        params.seedYStrides[i] = static_cast<uint64_t>(seedYStrides[i]);
    }
    launchFloodFillKernel(output, outputBytes, image, imageBytes, seedX,
                          seedXBytes, seedY, seedYBytes, &params,
                          sizeof(params), newValue, lower, upper, valueBytes,
                          floodFillFunctionName(type));
}

void launchMetalCannyNonmax(
    BufferParam output, const size_t outputBytes, const af::dim4& dims,
    const af::dim4& outputStrides, BufferParam magnitude,
    const size_t magnitudeBytes, const af::dim4& magnitudeStrides,
    BufferParam derivativeX, const size_t derivativeXBytes,
    const af::dim4& derivativeXStrides, BufferParam derivativeY,
    const size_t derivativeYBytes, const af::dim4& derivativeYStrides) {
    CannyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.firstStrides[i]  = static_cast<uint64_t>(magnitudeStrides[i]);
        params.secondStrides[i] = static_cast<uint64_t>(derivativeXStrides[i]);
        params.thirdStrides[i]  = static_cast<uint64_t>(derivativeYStrides[i]);
    }
    launchThreeInputKernel(
        output, outputBytes, magnitude, magnitudeBytes, derivativeX,
        derivativeXBytes, derivativeY, derivativeYBytes, &params,
        sizeof(params), static_cast<size_t>(dims.elements()),
        "canny_nonmax_float", "Canny non-maximum suppression");
}

void launchMetalCannyHysteresis(BufferParam output, const size_t outputBytes,
                                const af::dim4& dims,
                                const af::dim4& outputStrides,
                                BufferParam strong, const size_t strongBytes,
                                const af::dim4& strongStrides, BufferParam weak,
                                const size_t weakBytes,
                                const af::dim4& weakStrides) {
    CannyParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.firstStrides[i]  = static_cast<uint64_t>(strongStrides[i]);
        params.secondStrides[i] = static_cast<uint64_t>(weakStrides[i]);
    }

    UNUSED(strongBytes);
    UNUSED(weakBytes);
    auto changedBuffer    = NS::TransferPtr(metalDevice()->newBuffer(
        sizeof(uint32_t), MTL::ResourceStorageModeShared));
    if (!strong.buffer || !weak.buffer || !output.buffer || !changedBuffer)
        AF_ERROR("Could not allocate Metal Canny buffers", AF_ERR_NO_MEM);

    const size_t total = static_cast<size_t>(dims.elements());
    auto dispatch      = [&](const char* functionName, auto bindBuffers) {
        auto commandBuffer =
            NS::RetainPtr(metalCommandQueue()->commandBuffer());
        auto encoder =
            commandBuffer
                     ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                     : nullptr;
        if (!commandBuffer || !encoder)
            AF_ERROR("Could not create a Metal command encoder",
                          AF_ERR_RUNTIME);
        auto* pipeline = metalPipeline(functionName);
        encoder->setComputePipelineState(pipeline);
        bindBuffers(encoder.get());
        const auto width = std::min<NS::UInteger>(
            256, pipeline->maxTotalThreadsPerThreadgroup());
        encoder->dispatchThreads(MTL::Size(total, 1, 1),
                                      MTL::Size(width, 1, 1));
        encoder->endEncoding();
        submitCommandBuffer(commandBuffer.get());
        // The convergence loop reads changedBuffer on the CPU.
        syncCommandQueue();
        if (commandBuffer->status() == MTL::CommandBufferStatusError) {
            const std::string message = metalErrorDescription(
                commandBuffer->error(), "Metal Canny hysteresis failed");
            AF_ERROR(message.c_str(), AF_ERR_RUNTIME);
        }
    };

    dispatch("canny_hysteresis_init_char", [&](MTL::ComputeCommandEncoder* e) {
        e->setBuffer(strong.buffer, strong.offset, 0);
        e->setBuffer(output.buffer, output.offset, 1);
        e->setBytes(&params, sizeof(params), 2);
    });

    auto* changed = static_cast<uint32_t*>(changedBuffer->contents());
    do {
        *changed = 0;
        dispatch("canny_hysteresis_step_char",
                 [&](MTL::ComputeCommandEncoder* e) {
                     e->setBuffer(weak.buffer, weak.offset, 0);
                     e->setBuffer(output.buffer, output.offset, 1);
                     e->setBuffer(changedBuffer.get(), 0, 2);
                     e->setBytes(&params, sizeof(params), 3);
                 });
    } while (*changed != 0);

}

namespace {

const char* regionsFunctionName(const af_dtype type) noexcept {
    switch (type) {
        case f32: return "regions_float";
        case s32: return "regions_int";
        case u32: return "regions_uint";
        case s16: return "regions_short";
        case u16: return "regions_ushort";
        default: return nullptr;
    }
}

}  // namespace

bool supportsMetalRegions(const af_dtype type) noexcept {
    return regionsFunctionName(type) != nullptr;
}

void launchMetalRegions(BufferParam output, const size_t outputBytes,
                        const af::dim4& dims, const af::dim4& outputStrides,
                        BufferParam input, const size_t inputBytes,
                        const af::dim4& inputStrides,
                        const af_connectivity connectivity,
                        const af_dtype type) {
    RegionsParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]          = static_cast<uint64_t>(dims[i]);
        params.outputStrides[i] = static_cast<uint64_t>(outputStrides[i]);
        params.inputStrides[i]  = static_cast<uint64_t>(inputStrides[i]);
    }
    params.connectivity = static_cast<uint32_t>(connectivity);
    launchSingleInputKernel(output, outputBytes, dims, input, inputBytes,
                            &params, sizeof(params), regionsFunctionName(type),
                            "connected components");
}

void launchMetalAnisotropicDiffusion(BufferParam inout, const size_t bytes,
                                     const af::dim4& dims,
                                     const af::dim4& strides, const float dt,
                                     const float mct,
                                     const af_flux_function flux,
                                     const bool curvature) {
    DiffusionParams params{};
    for (int i = 0; i < 4; ++i) {
        params.dims[i]    = static_cast<uint64_t>(dims[i]);
        params.strides[i] = static_cast<uint64_t>(strides[i]);
    }
    params.dt        = dt;
    params.mct       = mct;
    params.flux      = static_cast<uint32_t>(flux);
    params.curvature = curvature;
    launchSingleInputKernel(inout, bytes, dims, inout, bytes, &params,
                            sizeof(params), "anisotropic_diffusion_float",
                            "anisotropic diffusion");
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
