/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * Distributed under the 3-clause BSD license.
 ********************************************************/

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <Metal.hpp>

#include <err_metal.hpp>
#include <metal_fft.hpp>
#include <platform.hpp>

#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>

namespace arrayfire {
namespace metal {
namespace {

MPSShape* graphShape(const af::dim4& dims) {
    return @[ @(dims[3]), @(dims[2]), @(dims[1]), @(dims[0]) ];
}

NSArray<NSNumber*>* graphAxes(const int rank) {
    NSMutableArray<NSNumber*>* axes =
        [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(rank)];
    // ArrayFire's dimension 0 is the fastest-varying dimension. MPSGraph's
    // real/Hermitian transforms reduce the last listed axis, so list the
    // higher ArrayFire dimensions first and dimension 0 last.
    for (int d = rank - 1; d >= 0; --d) {
        [axes addObject:@(3 - d)];
    }
    return axes;
}

id<MTLBuffer> bridgeBuffer(MTL::Buffer* buffer) {
    return (__bridge id<MTLBuffer>)(void*)buffer;
}

id<MTLCommandQueue> bridgeQueue() {
    return (__bridge id<MTLCommandQueue>)(void*)
        &arrayfire::metal::getCommandQueue();
}

void ensureBuffers(MTL::Buffer* output, MTL::Buffer* input) {
    if (!output || !input) {
        AF_ERROR("Could not allocate Metal FFT buffers", AF_ERR_NO_MEM);
    }
}

struct FFTGraphPlan {
    __strong MPSGraph* graph;
    __strong MPSGraphTensor* inputTensor;
    __strong MPSGraphTensor* outputTensor;
    __strong MPSShape* inputShape;
    __strong MPSShape* outputShape;
    MPSDataType inputType;
    MPSDataType outputType;
    std::mutex runMutex;

    FFTGraphPlan(MPSGraph* graph_, MPSGraphTensor* inputTensor_,
                 MPSGraphTensor* outputTensor_, MPSShape* inputShape_,
                 MPSShape* outputShape_, MPSDataType inputType_,
                 MPSDataType outputType_)
        : graph(graph_), inputTensor(inputTensor_), outputTensor(outputTensor_),
          inputShape(inputShape_), outputShape(outputShape_),
          inputType(inputType_), outputType(outputType_) {}
};

enum class FFTKind : unsigned char { C2CForward, C2CInverse, R2C, C2R };

struct FFTPlanCache {
    std::mutex mutex;
    size_t maxSize = 5;
    std::deque<std::pair<std::string, std::shared_ptr<FFTGraphPlan>>> plans;
};

FFTPlanCache& fftPlanCache() {
    static FFTPlanCache cache;
    return cache;
}

std::string fftPlanKey(FFTKind kind, const af::dim4& dims, int rank) {
    std::ostringstream key;
    key << static_cast<unsigned>(kind) << ':' << rank;
    for (int d = 0; d < 4; ++d) { key << ':' << dims[d]; }
    return key.str();
}

template<typename Factory>
std::shared_ptr<FFTGraphPlan> getFFTPlan(const std::string& key,
                                         Factory&& makePlan) {
    FFTPlanCache& cache = fftPlanCache();
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        for (const auto& entry : cache.plans) {
            if (entry.first == key) { return entry.second; }
        }
    }

    // Build outside the cache lock. Concurrent misses may build the same
    // template, so check again before inserting and use the first cached plan.
    std::shared_ptr<FFTGraphPlan> plan = makePlan();
    {
        std::lock_guard<std::mutex> lock(cache.mutex);
        for (const auto& entry : cache.plans) {
            if (entry.first == key) { return entry.second; }
        }
        if (cache.maxSize > 0) {
            if (cache.plans.size() >= cache.maxSize) { cache.plans.pop_back(); }
            cache.plans.emplace_front(key, plan);
        }
    }
    return plan;
}

void runGraph(const std::shared_ptr<FFTGraphPlan>& plan, MTL::Buffer* output,
              MTL::Buffer* input) {
    MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:bridgeBuffer(input)
                     shape:plan->inputShape
                  dataType:plan->inputType];
    MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:bridgeBuffer(output)
                     shape:plan->outputShape
                  dataType:plan->outputType];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        @{ plan->inputTensor : inputData };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{ plan->outputTensor : outputData };
    // The SDK does not promise that the same graph can execute concurrently.
    // Serialize submissions for a plan but do not hold the cache lock here.
    std::lock_guard<std::mutex> lock(plan->runMutex);
    [plan->graph runWithMTLCommandQueue:bridgeQueue()
                                  feeds:feeds
                       targetOperations:nil
                      resultsDictionary:results];
}

void checkFFTAvailability() {
    if (!@available(macOS 14.0, *)) {
        AF_ERROR("Metal FFT requires macOS 14 or newer", AF_ERR_NOT_SUPPORTED);
    }
}

}  // namespace

void setMetalFFTPlanCacheSize(size_t numPlans) {
    FFTPlanCache& cache = fftPlanCache();
    std::lock_guard<std::mutex> lock(cache.mutex);
    cache.maxSize = numPlans;
    while (cache.plans.size() > cache.maxSize) { cache.plans.pop_back(); }
}

void fftComplexFloat(MTL::Buffer* output, MTL::Buffer* input,
                     const af::dim4& dims, const int rank, const bool inverse) {
    ensureBuffers(output, input);
    checkFFTAvailability();

    @autoreleasepool {
        const FFTKind kind =
            inverse ? FFTKind::C2CInverse : FFTKind::C2CForward;
        auto plan = getFFTPlan(fftPlanKey(kind, dims, rank), [&] {
            MPSGraph* graph = [MPSGraph new];
            MPSShape* shape = graphShape(dims);
            MPSGraphTensor* inputTensor =
                [graph placeholderWithShape:shape
                                   dataType:MPSDataTypeComplexFloat32
                                       name:nil];
            MPSGraphFFTDescriptor* descriptor =
                [MPSGraphFFTDescriptor descriptor];
            descriptor.inverse = inverse;
            descriptor.scalingMode = MPSGraphFFTScalingModeNone;
            MPSGraphTensor* outputTensor =
                [graph fastFourierTransformWithTensor:inputTensor
                                                 axes:graphAxes(rank)
                                           descriptor:descriptor
                                                 name:nil];
            return std::make_shared<FFTGraphPlan>(
                graph, inputTensor, outputTensor, shape, shape,
                MPSDataTypeComplexFloat32, MPSDataTypeComplexFloat32);
        });
        runGraph(plan, output, input);
    }
}

void fftRealToComplexFloat(MTL::Buffer* output, MTL::Buffer* input,
                           const af::dim4& inputDims, const int rank) {
    ensureBuffers(output, input);
    checkFFTAvailability();

    @autoreleasepool {
        af::dim4 outputDims = inputDims;
        outputDims[0] = outputDims[0] / 2 + 1;
        auto plan = getFFTPlan(fftPlanKey(FFTKind::R2C, inputDims, rank), [&] {
            MPSGraph* graph = [MPSGraph new];
            MPSShape* inputShape = graphShape(inputDims);
            MPSShape* outputShape = graphShape(outputDims);
            MPSGraphTensor* inputTensor =
                [graph placeholderWithShape:inputShape
                                   dataType:MPSDataTypeFloat32
                                       name:nil];
            MPSGraphFFTDescriptor* descriptor =
                [MPSGraphFFTDescriptor descriptor];
            descriptor.inverse = NO;
            descriptor.scalingMode = MPSGraphFFTScalingModeNone;
            MPSGraphTensor* outputTensor =
                [graph realToHermiteanFFTWithTensor:inputTensor
                                               axes:graphAxes(rank)
                                         descriptor:descriptor
                                               name:nil];
            return std::make_shared<FFTGraphPlan>(
                graph, inputTensor, outputTensor, inputShape, outputShape,
                MPSDataTypeFloat32, MPSDataTypeComplexFloat32);
        });
        runGraph(plan, output, input);
    }
}

void fftComplexToRealFloat(MTL::Buffer* output, MTL::Buffer* input,
                           const af::dim4& outputDims, const int rank) {
    ensureBuffers(output, input);
    checkFFTAvailability();

    @autoreleasepool {
        af::dim4 inputDims = outputDims;
        inputDims[0] = outputDims[0] / 2 + 1;
        auto plan = getFFTPlan(fftPlanKey(FFTKind::C2R, outputDims, rank), [&] {
            MPSGraph* graph = [MPSGraph new];
            MPSShape* inputShape = graphShape(inputDims);
            MPSShape* outputShape = graphShape(outputDims);
            MPSGraphTensor* inputTensor =
                [graph placeholderWithShape:inputShape
                                   dataType:MPSDataTypeComplexFloat32
                                       name:nil];
            MPSGraphFFTDescriptor* descriptor =
                [MPSGraphFFTDescriptor descriptor];
            descriptor.inverse = YES;
            descriptor.scalingMode = MPSGraphFFTScalingModeNone;
            descriptor.roundToOddHermitean = (outputDims[0] % 2) != 0;
            MPSGraphTensor* outputTensor =
                [graph HermiteanToRealFFTWithTensor:inputTensor
                                              axes:graphAxes(rank)
                                        descriptor:descriptor
                                              name:nil];
            return std::make_shared<FFTGraphPlan>(
                graph, inputTensor, outputTensor, inputShape, outputShape,
                MPSDataTypeComplexFloat32, MPSDataTypeFloat32);
        });
        runGraph(plan, output, input);
    }
}

}  // namespace metal
}  // namespace arrayfire
