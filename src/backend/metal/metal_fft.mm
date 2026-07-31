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

void runGraph(MPSGraphTensor* inputTensor, MPSGraphTensor* outputTensor,
              MPSGraphTensorData* inputData, MPSGraphTensorData* outputData,
              MPSGraph* graph) {
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
        @{ inputTensor : inputData };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
        @{ outputTensor : outputData };
    [graph runWithMTLCommandQueue:bridgeQueue()
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

void fftComplexFloat(MTL::Buffer* output, MTL::Buffer* input,
                     const af::dim4& dims, const int rank, const bool inverse) {
    ensureBuffers(output, input);
    checkFFTAvailability();

    @autoreleasepool {
        MPSGraph* graph = [MPSGraph new];
        MPSShape* shape = graphShape(dims);
        MPSGraphTensor* inputTensor =
            [graph placeholderWithShape:shape
                               dataType:MPSDataTypeComplexFloat32
                                   name:nil];
        MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
        descriptor.inverse = inverse;
        descriptor.scalingMode = MPSGraphFFTScalingModeNone;
        MPSGraphTensor* outputTensor =
            [graph fastFourierTransformWithTensor:inputTensor
                                             axes:graphAxes(rank)
                                       descriptor:descriptor
                                             name:nil];

        MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bridgeBuffer(input)
                         shape:shape
                      dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bridgeBuffer(output)
                         shape:shape
                      dataType:MPSDataTypeComplexFloat32];
        runGraph(inputTensor, outputTensor, inputData, outputData, graph);
    }
}

void fftRealToComplexFloat(MTL::Buffer* output, MTL::Buffer* input,
                           const af::dim4& inputDims, const int rank) {
    ensureBuffers(output, input);
    checkFFTAvailability();

    @autoreleasepool {
        MPSGraph* graph = [MPSGraph new];
        MPSShape* inputShape = graphShape(inputDims);
        af::dim4 outputDims = inputDims;
        outputDims[0] = outputDims[0] / 2 + 1;
        MPSShape* outputShape = graphShape(outputDims);

        MPSGraphTensor* inputTensor =
            [graph placeholderWithShape:inputShape
                               dataType:MPSDataTypeFloat32
                                   name:nil];
        MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
        descriptor.inverse = NO;
        descriptor.scalingMode = MPSGraphFFTScalingModeNone;
        MPSGraphTensor* outputTensor =
            [graph realToHermiteanFFTWithTensor:inputTensor
                                           axes:graphAxes(rank)
                                     descriptor:descriptor
                                           name:nil];

        MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bridgeBuffer(input)
                         shape:inputShape
                      dataType:MPSDataTypeFloat32];
        MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bridgeBuffer(output)
                         shape:outputShape
                      dataType:MPSDataTypeComplexFloat32];
        runGraph(inputTensor, outputTensor, inputData, outputData, graph);
    }
}

void fftComplexToRealFloat(MTL::Buffer* output, MTL::Buffer* input,
                           const af::dim4& outputDims, const int rank) {
    ensureBuffers(output, input);
    checkFFTAvailability();

    @autoreleasepool {
        MPSGraph* graph = [MPSGraph new];
        af::dim4 inputDims = outputDims;
        inputDims[0] = outputDims[0] / 2 + 1;
        MPSShape* inputShape = graphShape(inputDims);
        MPSShape* outputShape = graphShape(outputDims);

        MPSGraphTensor* inputTensor =
            [graph placeholderWithShape:inputShape
                               dataType:MPSDataTypeComplexFloat32
                                   name:nil];
        MPSGraphFFTDescriptor* descriptor = [MPSGraphFFTDescriptor descriptor];
        descriptor.inverse = YES;
        descriptor.scalingMode = MPSGraphFFTScalingModeNone;
        descriptor.roundToOddHermitean = (outputDims[0] % 2) != 0;
        MPSGraphTensor* outputTensor =
            [graph HermiteanToRealFFTWithTensor:inputTensor
                                           axes:graphAxes(rank)
                                     descriptor:descriptor
                                           name:nil];

        MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bridgeBuffer(input)
                         shape:inputShape
                      dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bridgeBuffer(output)
                         shape:outputShape
                      dataType:MPSDataTypeFloat32];
        runGraph(inputTensor, outputTensor, inputData, outputData, graph);
    }
}

}  // namespace metal
}  // namespace arrayfire
