/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/traits.hpp>

#include <cstdint>

namespace arrayfire {
namespace metal {
namespace kernel {

struct CopyParams {
    uint64_t dims[4];
    int64_t outputStrides[4];
    int64_t inputStrides[4];
    int64_t outputOffset;
    int64_t inputOffset;
};

struct RangeParams {
    uint64_t dims[4];
    uint64_t strides[4];
    uint32_t sequenceDimension;
};

const char* copyFunctionName(const af_dtype type);

struct ResizeParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint32_t method;
};

struct PadParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    int64_t lower[4];
    uint32_t borderType;
};

struct HsvRgbParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
};

struct MomentsParams {
    uint64_t inputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t moment;
};

struct HistogramParams {
    uint64_t inputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t bins;
    float minValue;
    float maxValue;
    uint32_t linear;
};

struct MorphParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t maskDims[4];
    uint64_t maskStrides[4];
    uint32_t dilation;
    uint32_t volume;
};

struct NearestNeighbourParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t queryDims[4];
    uint64_t queryStrides[4];
    uint64_t trainDims[4];
    uint64_t trainStrides[4];
    uint32_t distanceDimension;
    uint32_t distanceType;
};

struct IirParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t coefficientStrides[4];
    uint64_t feedbackDims[4];
    uint64_t feedbackStrides[4];
    uint32_t feedbackBatched;
};

struct DotParams {
    uint64_t length;
    uint64_t lhsStride;
    uint64_t rhsStride;
    uint32_t conjugateLhs;
    uint32_t conjugateRhs;
};

struct GemmParams {
    uint64_t outputDims[4];
    int64_t outputStrides[4];
    uint64_t lhsDims[4];
    int64_t lhsStrides[4];
    uint64_t rhsDims[4];
    int64_t rhsStrides[4];
    uint32_t transposeLhs;
    uint32_t conjugateLhs;
    uint32_t transposeRhs;
    uint32_t conjugateRhs;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
};

struct BilateralParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    float spatialSigma;
    float chromaticSigma;
};

struct MeanshiftParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    float spatialSigma;
    float chromaticSigma;
    uint32_t iterations;
    uint32_t color;
};

struct MedfiltParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t windowLength;
    uint32_t windowWidth;
    uint32_t padding;
    uint32_t oneDimensional;
};
struct MatchTemplateParams {
    uint64_t dims[4], outputStrides[4], searchStrides[4], templateDims[4],
        templateStrides[4];
    uint32_t matchType;
};
struct RotateParams {
    uint64_t odims[4], ostrides[4], idims[4], istrides[4];
    float transform[6];
    uint32_t method;
};
struct LuParams {
    uint64_t odims[4], ostrides[4], idims[4], istrides[4];
    uint32_t lower;
};
struct LuFactorParams {
    uint64_t rows;
    uint64_t columns;
    uint64_t stride0;
    uint64_t stride1;
    uint32_t step;
};
struct QRFactorParams {
    uint64_t rows;
    uint64_t columns;
    uint64_t stride0;
    uint64_t stride1;
    uint32_t step;
};
struct QRGenerateParams {
    uint64_t rows;
    uint64_t columns;
    uint64_t outputStride0;
    uint64_t outputStride1;
    uint64_t packedStride0;
    uint64_t packedStride1;
    uint32_t step;
};
struct PivotParams {
    uint64_t outputCount, pivotCount;
};

// Parameters shared by the native Metal solve kernels.  The kernels operate
// directly on ArrayFire's strided buffers; no host-side view of the data is
// needed.  batchZ is kept separately so a linear batch id can be mapped back
// to the third and fourth dimensions without relying on contiguous storage.
struct SolveParams {
    uint64_t rows;
    uint64_t columns;
    uint64_t rhsColumns;
    uint64_t aStrides[4];
    uint64_t bStrides[4];
    uint64_t pivotStrides[4];
    uint32_t batch;
    uint32_t batchZ;
    uint32_t upper;
    uint32_t unit;
};

struct SolveGramParams {
    uint64_t rows;
    uint64_t columns;
    uint64_t rank;
    uint64_t rhsColumns;
    uint64_t aStrides[4];
    uint64_t bStrides[4];
    uint64_t gramStrides[4];
    uint64_t rhsStrides[4];
    uint32_t batch;
    uint32_t batchZ;
    uint32_t underdetermined;
};

struct SolveExpandParams {
    uint64_t rows;
    uint64_t columns;
    uint64_t rhsColumns;
    uint64_t aStrides[4];
    uint64_t yStrides[4];
    uint64_t outputStrides[4];
    uint32_t batch;
    uint32_t batchZ;
};

struct SparseMatmulParams {
    uint64_t sourceRows;
    uint64_t sourceColumns;
    uint64_t outputRows;
    uint64_t rhsColumns;
    uint64_t outputStrides[4];
    uint64_t rhsStrides[4];
    uint32_t operation;
};

struct TopKParams {
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t outputStrides[4];
    uint64_t k;
    uint32_t ascending;
};

const char* hsvRgbFunctionName(const af_dtype type, const bool hsvToRgb);

const char* momentsFunctionName(const af_dtype type);

const char* histogramFunctionName(const af_dtype type);

const char* morphFunctionName(const af_dtype type);

const char* nearestNeighbourFunctionName(const af_dtype inputType,
                                         const af_dtype outputType,
                                         const af_match_type distanceType);

const char* iirFunctionName(const af_dtype type);

const char* dotFunctionName(const af_dtype type);
const char* gemmFunctionName(const af_dtype type);

const char* bilateralFunctionName(const af_dtype inputType,
                                  const af_dtype outputType);

const char* meanshiftFunctionName(const af_dtype type);

const char* medfiltFunctionName(const af_dtype type);

const char* matchTemplateFunctionName(const af_dtype in, const af_dtype out);

const char* rotateFunctionName(const af_dtype t);

const char* luFunctionName(const af_dtype t);
const char* luFactorFunctionName(const af_dtype t);
const char* qrFunctionName(const af_dtype t);

const char* floodFillFunctionName(const af_dtype type);

const char* indexFunctionName(const af_dtype type);

const char* assignFunctionName(const af_dtype type);

const char* transformFunctionName(const af_dtype type);

const char* sortFunctionName(const af_dtype type);

const char* sortByKeyFunctionName(const af_dtype keyType,
                                  const af_dtype valueType);

const char* sortIndexFunctionName(const af_dtype type);

const char* scanFunctionName(const af_dtype inputType,
                             const af_dtype outputType);

const char* scanByKeyFunctionName(const af_dtype keyType,
                                  const af_dtype valueType);

const char* meanFunctionName(const af_dtype inputType,
                             const af_dtype outputType);

const char* meanWeightedFunctionName(const af_dtype valueType,
                                     const af_dtype weightType);

const char* reduceFunctionName(const af_dtype inputType,
                               const af_dtype outputType);
const char* reduceByKeyCompactFunctionName(const af_dtype keyType);
const char* reduceByKeyFunctionName(const af_dtype keyType,
                                    const af_dtype inputType,
                                    const af_dtype outputType);
const char* ireduceFunctionName(const af_dtype type);
const char* ireduceAllFunctionName(const af_dtype type);
const char* rreduceFunctionName(const af_dtype type);
const char* fastFunctionName(const af_dtype type);

struct ScanParams {
    uint64_t dims[4];
    int64_t outputStrides[4];
    int64_t inputStrides[4];
    uint32_t dimension;
    uint32_t operation;
    uint32_t inclusive;
};

struct ScanByKeyParams {
    uint64_t dims[4];
    int64_t outputStrides[4];
    int64_t keyStrides[4];
    int64_t inputStrides[4];
    uint32_t dimension;
    uint32_t operation;
    uint32_t inclusive;
};

struct MeanParams {
    uint64_t outputDims[4];
    int64_t outputStrides[4];
    uint64_t inputDims[4];
    int64_t inputStrides[4];
    int64_t weightStrides[4];
    uint32_t dimension;
    uint32_t reduceAll;
};

struct ApproxParams {
    uint64_t outputDims[4];
    int64_t outputStrides[4];
    uint64_t inputDims[4];
    int64_t inputStrides[4];
    uint64_t xDims[4];
    int64_t xStrides[4];
    int64_t yStrides[4];
    int64_t outputOffset;
    int64_t inputOffset;
    int64_t xOffset;
    int64_t yOffset;
    uint32_t xDimension;
    uint32_t yDimension;
    uint32_t method;
    float xBegin;
    float xStep;
    float yBegin;
    float yStep;
    float offGrid;
};

struct ReduceParams {
    uint64_t outputDims[4];
    int64_t outputStrides[4];
    uint64_t inputDims[4];
    int64_t inputStrides[4];
    uint32_t dimension;
    uint32_t operation;
    uint32_t reduceAll;
    uint32_t changeNan;
    float nanValue;
};

struct ReduceByKeyCompactParams {
    uint64_t length;
    int64_t inputStride;
};

struct ReduceByKeyParams {
    uint64_t outputDims[4];
    int64_t outputStrides[4];
    uint64_t inputDims[4];
    int64_t inputStrides[4];
    int64_t keyStride;
    uint32_t dimension;
    uint32_t operation;
    uint32_t nReduced;
    uint32_t changeNan;
    float nanValue;
};

struct IReduceParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t rlenDims[4];
    uint64_t rlenStrides[4];
    uint32_t dimension;
    uint32_t isMax;
};

struct IReduceAllParams {
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint32_t isMax;
};

struct ConvolveParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t signalDims[4];
    uint64_t signalStrides[4];
    uint64_t filterDims[4];
    uint64_t filterStrides[4];
    uint32_t rank;
    uint32_t expand;
    uint32_t batchKind;
};

struct SeparableConvolveParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t signalDims[4];
    uint64_t signalStrides[4];
    uint64_t filterDims[4];
    uint64_t filterStrides[4];
    uint32_t convDim;
    uint32_t expand;
};

struct ConvolveNNParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t signalDims[4];
    uint64_t signalStrides[4];
    uint64_t filterDims[4];
    uint64_t filterStrides[4];
    uint64_t gradientDims[4];
    uint64_t gradientStrides[4];
    uint64_t stride[2];
    int64_t padding[2];
    uint64_t dilation[2];
};

struct FastParams {
    uint64_t rows;
    uint64_t columns;
    float threshold;
    uint32_t arcLength;
    uint32_t nonmax;
    uint32_t maxFeatures;
    uint32_t edge;
};

struct HarrisParams {
    uint64_t elements;
    uint32_t rows;
    uint32_t columns;
    uint32_t border;
    float k;
};

struct HarrisNonMaxParams {
    uint32_t rows;
    uint32_t columns;
    uint32_t border;
    uint32_t maxCorners;
    float minResponse;
};

struct HarrisKeepParams {
    uint32_t corners;
};

struct SusanParams {
    uint32_t rows;
    uint32_t columns;
    uint32_t radius;
    uint32_t border;
    float differenceThreshold;
    float geometricThreshold;
};

struct SusanNonMaxParams {
    uint32_t rows;
    uint32_t columns;
    uint32_t border;
    uint32_t maxCorners;
};

struct SvdInitParams {
    uint32_t inputRows;
    uint32_t inputColumns;
    uint32_t workRows;
    uint32_t workColumns;
    uint32_t transpose;
};

struct SvdStageParams {
    uint32_t rows;
    uint32_t columns;
    uint32_t stage;
    uint32_t threads;
};

struct SvdFinalizeParams {
    uint32_t inputRows;
    uint32_t inputColumns;
    uint32_t workRows;
    uint32_t workColumns;
    uint32_t transpose;
};

struct RandomParams {
    uint64_t elements;
    uint64_t seed;
    uint64_t counter;
    uint32_t type;
};

struct MersenneParams {
    uint64_t elements;
    uint32_t mask;
    uint32_t reserved;
};

struct MersenneInitParams {
    uint64_t seed;
};

struct SparseParams {
    uint32_t rows;
    uint32_t columns;
    uint32_t outputStride;
    uint32_t nonzeros;
};

struct SparseDenseToCsrParams {
    uint32_t rows;
    uint32_t columns;
    uint32_t inputStride;
    uint32_t nonzeros;
};

struct SparseCooToCsrParams {
    uint32_t rows;
    uint32_t nonzeros;
};

struct SparseCsrArithParams {
    uint32_t rows;
    uint32_t lhsNonzeros;
    uint32_t rhsNonzeros;
    uint32_t operation;
};

struct SparseArithParams {
    uint32_t nonzeros;
    uint32_t rows;
    uint32_t columns;
    uint32_t rhsStride;
    uint32_t csr;
    uint32_t reverse;
    uint32_t operation;
};

struct FFTConvolveParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t signalDims[4];
    uint64_t signalStrides[4];
    uint64_t filterDims[4];
    uint64_t filterStrides[4];
    uint64_t offset;
    uint32_t kind;
};

struct FFTConvolvePackParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
};

struct FFTConvolvePadParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t offset;
};

struct FFTConvolveReorderParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t filterDims[4];
    uint64_t filterOffset;
    uint64_t signalHalfDim0;
    uint64_t fftScale;
    uint32_t kind;
};

struct OrbParams {
    uint32_t features;
    uint32_t rows;
    uint32_t columns;
    uint32_t patchSize;
};

struct OrbHarrisParams {
    uint32_t features;
    uint32_t rows;
    uint32_t columns;
    uint32_t blockSize;
    uint32_t patchSize;
    float kThreshold;
};

struct OrbExtractParams {
    uint32_t features;
    uint32_t rows;
    uint32_t columns;
    uint32_t patchSize;
    float scale;
};

struct SiftParams {
    uint64_t elements;
};

struct ArrayAddParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t leftDims[4];
    uint64_t leftStrides[4];
    uint64_t rightDims[4];
    uint64_t rightStrides[4];
};

struct ExampleFunctionParams {
    uint64_t dims[4];
    int64_t outputStrides[4];
    int64_t leftStrides[4];
    int64_t rightStrides[4];
};

const char* exampleFunctionName(af_dtype type);

struct SortByKeyParams {
    uint64_t dims[4];
    uint64_t keyStrides[4];
    uint64_t valueStrides[4];
    uint32_t ascending;
    uint32_t dimension;
};

struct SortParams {
    uint64_t dims[4];
    uint64_t strides[4];
    uint32_t ascending;
    uint32_t dimension;
};

struct TransformParams {
    uint64_t outputDims[4];
    int64_t outputStrides[4];
    uint64_t inputDims[4];
    int64_t inputStrides[4];
    uint64_t transformDims[4];
    int64_t transformStrides[4];
    int64_t outputOffset;
    int64_t inputOffset;
    int64_t transformOffset;
    uint32_t method;
    uint32_t inverse;
    uint32_t perspective;
};

struct AssignParams {
    uint64_t outputDims[4];
    int64_t destinationStrides[4];
    uint64_t rhsDims[4];
    int64_t rhsStrides[4];
    int64_t offsets[4];
    int64_t indexOffsets[4];
    int64_t indexStrides[4];
    int64_t outputOffset;
    int64_t rhsOffset;
    uint32_t isSequence[4];
};

struct IndexParams {
    uint64_t outputDims[4];
    int64_t outputStrides[4];
    uint64_t inputDims[4];
    int64_t inputStrides[4];
    int64_t offsets[4];
    int64_t steps[4];
    int64_t indexOffsets[4];
    int64_t indexStrides[4];
    int64_t outputOffset;
    int64_t inputOffset;
    uint32_t isSequence[4];
};

struct FloodFillParams {
    uint64_t imageDims[4];
    uint64_t imageStrides[4];
    uint64_t seedDims[4];
    uint64_t seedXStrides[4];
    uint64_t seedYStrides[4];
};

struct CannyParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t firstStrides[4];
    uint64_t secondStrides[4];
    uint64_t thirdStrides[4];
};

struct RegionsParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t connectivity;
};

struct DiffusionParams {
    uint64_t dims[4];
    uint64_t strides[4];
    float dt;
    float mct;
    uint32_t flux;
    uint32_t curvature;
};

struct IotaParams {
    uint64_t dims[4];
    uint64_t strides[4];
    uint64_t sourceDims[4];
};

struct IdentityParams {
    uint64_t dims[4];
    uint64_t strides[4];
};

struct TileParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
};

struct ShiftParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint64_t shifts[4];
};

struct ReorderParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint64_t reorderDims[4];
};

struct SelectParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t aDims[4];
    uint64_t aStrides[4];
    uint64_t bDims[4];
    uint64_t bStrides[4];
    uint64_t conditionDims[4];
    uint64_t conditionStrides[4];
    uint32_t flip;
};

struct JoinParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t outputOffset[4];
};

struct LookupParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint32_t dimension;
    uint32_t indexType;
};

struct DiagonalParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    int32_t diagonal;
};

struct DiffParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t dimension;
};

struct TriangleParams {
    uint64_t dims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t upper;
    uint32_t unitDiagonal;
};

struct TransposeParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputStrides[4];
    uint32_t conjugate;
};

struct TransposeInplaceParams {
    uint64_t dims[4];
    uint64_t strides[4];
    uint32_t conjugate;
};

struct UnwrapParams {
    uint64_t outputDims[4];
    uint64_t outputStrides[4];
    uint64_t inputDims[4];
    uint64_t inputStrides[4];
    uint64_t windowX;
    uint64_t windowY;
    uint64_t strideX;
    uint64_t strideY;
    int64_t paddingX;
    int64_t paddingY;
    uint64_t dilationX;
    uint64_t dilationY;
    uint32_t columnDimension;
};

using WrapParams = UnwrapParams;

struct GradientParams {
    uint64_t dims[4];
    uint64_t inputStrides[4];
    uint64_t gradient0Strides[4];
    uint64_t gradient1Strides[4];
};

struct SobelParams {
    uint64_t dims[4];
    uint64_t inputStrides[4];
    uint64_t derivative0Strides[4];
    uint64_t derivative1Strides[4];
};


const char* rangeFunctionName(const af_dtype type);

const char* resizeFunctionName(const af_dtype type);

const char* padFunctionName(const af_dtype type);

const char* iotaFunctionName(const af_dtype type);

const char* identityFunctionName(const af_dtype type);

const char* tileFunctionName(const af_dtype type);

const char* shiftFunctionName(const af_dtype type);

const char* reorderFunctionName(const af_dtype type);

const char* selectFunctionName(const af_dtype type, const bool scalar);

const char* joinFunctionName(const af_dtype type);

const char* lookupFunctionName(const af_dtype type);

int lookupIndexType(const af_dtype type);

const char* diagonalFunctionName(const af_dtype type, const bool create);

const char* diffFunctionName(const af_dtype type, const bool secondOrder);

const char* triangleFunctionName(const af_dtype type);

const char* transposeFunctionName(const af_dtype type, const bool inplace);

const char* unwrapFunctionName(const af_dtype type);

const char* wrapFunctionName(const af_dtype type);

const char* gradientFunctionName(const af_dtype type);

const char* sobelFunctionName(const af_dtype type);

const char* susanResponseFunctionName(const af_dtype type);

const char* susanNonMaxFunctionName(const af_dtype type);

const char* svdInitFunctionName(const af_dtype type);

const char* svdStageFunctionName(const af_dtype type);

const char* svdSortFunctionName(const af_dtype type);

const char* svdFinalizeFunctionName(const af_dtype type);

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
