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

#include <cstdint>
#include <vector>

#include <err_metal.hpp>
#include <metal_inverse.hpp>
#include <platform.hpp>

namespace arrayfire {
namespace metal {
namespace {

MPSShape* graphShape(const af::dim4& dims) {
    // ArrayFire stores dimension 0 contiguously. MPSGraph uses the final
    // tensor axis as the contiguous axis, so reverse the four dimensions.
    return @[ @(dims[3]), @(dims[2]), @(dims[1]), @(dims[0]) ];
}

id<MTLBuffer> bridgeBuffer(MTL::Buffer* buffer) {
    return (__bridge id<MTLBuffer>)(void*)buffer;
}

id<MTLCommandQueue> bridgeQueue() {
    return (__bridge id<MTLCommandQueue>)(void*)
        &arrayfire::metal::getCommandQueue();
}

MPSDataType graphDataType(const af_dtype type) {
    switch (type) {
        case f32: return MPSDataTypeFloat32;
        case c32: return MPSDataTypeComplexFloat32;
        default:
            AF_ERROR("Metal inverse supports only single-precision values",
                     AF_ERR_NOT_SUPPORTED);
    }
}

}  // namespace

void inverseMatrix(MTL::Buffer* output, MTL::Buffer* input,
                   const af::dim4& dims, const af_dtype type) {
    if (!output || !input) {
        AF_ERROR("Could not allocate Metal inverse buffers", AF_ERR_NO_MEM);
    }
    if (dims[0] != dims[1]) {
        AF_ERROR("Metal matrix inverse requires a square matrix",
                 AF_ERR_ARG);
    }
    if (!@available(macOS 13.0, *)) {
        AF_ERROR("Metal matrix inverse requires macOS 13 or newer",
                 AF_ERR_NOT_SUPPORTED);
    }

    @autoreleasepool {
        MPSGraph* graph = [MPSGraph new];
        MPSShape* shape = graphShape(dims);
        const MPSDataType dataType = graphDataType(type);
        MPSGraphTensor* inputTensor =
            [graph placeholderWithShape:shape dataType:dataType name:nil];
        MPSGraphTensor* outputTensor = nil;

        if (type == c32) {
            // MPSGraph's matrix_inverse operation currently accepts real
            // tensors only.  A complex matrix A = R + iI can be represented
            // by the real block matrix [[R,-I],[I,R]]. Its inverse contains
            // the real and imaginary parts of A^-1 in the corresponding
            // blocks. This keeps the entire operation on the GPU and avoids
            // falling back to the host LAPACK implementation.
            if (!@available(macOS 14.0, *)) {
                AF_ERROR("Metal complex inverse requires macOS 14 or newer",
                         AF_ERR_NOT_SUPPORTED);
            }

            MPSGraphTensor* real =
                [graph realPartOfTensor:inputTensor name:nil];
            MPSGraphTensor* imaginary =
                [graph imaginaryPartOfTensor:inputTensor name:nil];
            MPSGraphTensor* negativeImaginary =
                [graph negativeWithTensor:imaginary name:nil];
            MPSGraphTensor* top =
                [graph concatTensors:@[ real, negativeImaginary ]
                          dimension:3
                               name:nil];
            MPSGraphTensor* bottom =
                [graph concatTensors:@[ imaginary, real ]
                          dimension:3
                               name:nil];
            MPSGraphTensor* block =
                [graph concatTensors:@[ top, bottom ] dimension:2 name:nil];
            MPSGraphTensor* blockInverse =
                [graph inverseOfTensor:block name:nil];

            // Refine the single-precision block inverse once.  This is a
            // Newton--Schulz correction, X <- X + X(I - AX), and materially
            // improves the residual for large, poorly-conditioned random
            // matrices without introducing a host synchronization.
            const NSUInteger blockSize =
                static_cast<NSUInteger>(2 * dims[0]);
            std::vector<uint32_t> indices(blockSize);
            for (NSUInteger i = 0; i < blockSize; ++i) {
                indices[i] = static_cast<uint32_t>(i);
            }
            NSData* indexData =
                [NSData dataWithBytes:indices.data()
                               length:indices.size() * sizeof(uint32_t)];
            MPSGraphTensor* indexTensor =
                [graph constantWithData:indexData
                                  shape:@[ @(blockSize) ]
                               dataType:MPSDataTypeInt32];
            MPSGraphTensor* identity =
                [graph oneHotWithIndicesTensor:indexTensor
                                         depth:blockSize
                                          axis:1
                                      dataType:MPSDataTypeFloat32
                                       onValue:1.0
                                      offValue:0.0
                                          name:nil];
            MPSGraphTensor* product =
                [graph matrixMultiplicationWithPrimaryTensor:block
                                               secondaryTensor:blockInverse
                                                          name:nil];
            MPSGraphTensor* residual =
                [graph subtractionWithPrimaryTensor:identity
                                    secondaryTensor:product
                                               name:nil];
            MPSGraphTensor* correction =
                [graph matrixMultiplicationWithPrimaryTensor:blockInverse
                                               secondaryTensor:residual
                                                          name:nil];
            blockInverse =
                [graph additionWithPrimaryTensor:blockInverse
                                 secondaryTensor:correction
                                            name:nil];

            const NSInteger rows = static_cast<NSInteger>(dims[1]);
            const NSInteger columns = static_cast<NSInteger>(dims[0]);
            MPSGraphTensor* inverseReal =
                [graph sliceTensor:blockInverse
                         dimension:2
                             start:0
                            length:rows
                              name:nil];
            inverseReal = [graph sliceTensor:inverseReal
                                   dimension:3
                                       start:0
                                      length:columns
                                        name:nil];
            MPSGraphTensor* inverseImaginary =
                [graph sliceTensor:blockInverse
                         dimension:2
                             start:rows
                            length:rows
                              name:nil];
            inverseImaginary = [graph sliceTensor:inverseImaginary
                                        dimension:3
                                            start:0
                                           length:columns
                                             name:nil];
            outputTensor =
                [graph complexTensorWithRealTensor:inverseReal
                                    imaginaryTensor:inverseImaginary
                                               name:nil];
        } else {
            outputTensor = [graph inverseOfTensor:inputTensor name:nil];
        }
        MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bridgeBuffer(input)
                         shape:shape
                         dataType:dataType];
        MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:bridgeBuffer(output)
                         shape:shape
                         dataType:dataType];
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
            @{inputTensor : inputData};
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
            @{outputTensor : outputData};
        [graph runWithMTLCommandQueue:bridgeQueue()
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:results];
    }
}

}  // namespace metal
}  // namespace arrayfire
