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
#include <af/traits.hpp>

#include <cstddef>
#include <vector>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalRange(af_dtype type) noexcept;
bool supportsMetalCopy(af_dtype type) noexcept;
bool supportsMetalResize(af_dtype type) noexcept;
bool supportsMetalPadBorders(af_dtype type) noexcept;
bool supportsMetalIota(af_dtype type) noexcept;
bool supportsMetalIdentity(af_dtype type) noexcept;
bool supportsMetalTile(af_dtype type) noexcept;
bool supportsMetalShift(af_dtype type) noexcept;
bool supportsMetalReorder(af_dtype type) noexcept;
bool supportsMetalSelect(af_dtype type) noexcept;
bool supportsMetalJoin(af_dtype type) noexcept;
bool supportsMetalLookup(af_dtype inputType, af_dtype indexType) noexcept;
bool supportsMetalDiagonal(af_dtype type) noexcept;
bool supportsMetalDiff(af_dtype type) noexcept;
bool supportsMetalTriangle(af_dtype type) noexcept;
bool supportsMetalTranspose(af_dtype type) noexcept;
bool supportsMetalUnwrap(af_dtype type) noexcept;
bool supportsMetalWrap(af_dtype type) noexcept;
bool supportsMetalGradient(af_dtype type) noexcept;
bool supportsMetalSobel(af_dtype inputType) noexcept;
bool supportsMetalHsvRgb(af_dtype type) noexcept;
bool supportsMetalMoments(af_dtype type) noexcept;

void launchMetalRange(void* output, size_t bytes, const af::dim4& dims,
                      const af::dim4& strides, unsigned sequenceDimension,
                      af_dtype type);

void launchMetalCopy(void* output, size_t outputBytes, const af::dim4& dims,
                     const af::dim4& outputStrides, const void* input,
                     size_t inputBytes, const af::dim4& inputStrides,
                     af_dtype type);

void launchMetalResize(void* output, size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, const void* input,
                       size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, af_interp_type method,
                       af_dtype type);

void launchMetalPadBorders(void* output, size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, const void* input,
                           size_t inputBytes, const af::dim4& inputDims,
                           const af::dim4& inputStrides,
                           const af::dim4& lowerPadding,
                           af_border_type borderType, af_dtype type);

void launchMetalIota(void* output, size_t bytes, const af::dim4& dims,
                     const af::dim4& strides, const af::dim4& sourceDims,
                     af_dtype type);

void launchMetalIdentity(void* output, size_t bytes, const af::dim4& dims,
                         const af::dim4& strides, af_dtype type);

void launchMetalTile(void* output, size_t outputBytes,
                     const af::dim4& outputDims, const af::dim4& outputStrides,
                     const void* input, size_t inputBytes,
                     const af::dim4& inputDims, const af::dim4& inputStrides,
                     af_dtype type);

void launchMetalShift(void* output, size_t outputBytes,
                      const af::dim4& outputDims, const af::dim4& outputStrides,
                      const void* input, size_t inputBytes,
                      const af::dim4& inputStrides, const af::dim4& shifts,
                      af_dtype type);

void launchMetalReorder(void* output, size_t outputBytes,
                        const af::dim4& outputDims,
                        const af::dim4& outputStrides, const void* input,
                        size_t inputBytes, const af::dim4& inputStrides,
                        const af::dim4& reorderDims, af_dtype type);

void launchMetalSelect(void* output, size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, const void* condition,
                       size_t conditionBytes, const af::dim4& conditionDims,
                       const af::dim4& conditionStrides, const void* a,
                       size_t aBytes, const af::dim4& aDims,
                       const af::dim4& aStrides, const void* b, size_t bBytes,
                       const af::dim4& bDims, const af::dim4& bStrides,
                       const void* scalar, bool flip, af_dtype type);

void launchMetalJoinAppend(void* output, size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, const void* input,
                           size_t inputBytes, const af::dim4& inputDims,
                           const af::dim4& inputStrides,
                           const af::dim4& outputOffset, bool preserveOutput,
                           af_dtype type);

void launchMetalLookup(void* output, size_t outputBytes,
                       const af::dim4& outputDims,
                       const af::dim4& outputStrides, const void* input,
                       size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, const void* indices,
                       size_t indexBytes, unsigned dimension,
                       af_dtype inputType, af_dtype indexType);

void launchMetalDiagCreate(void* output, size_t outputBytes,
                           const af::dim4& outputDims,
                           const af::dim4& outputStrides, const void* input,
                           size_t inputBytes, const af::dim4& inputDims,
                           const af::dim4& inputStrides, int diagonal,
                           af_dtype type);

void launchMetalDiagExtract(void* output, size_t outputBytes,
                            const af::dim4& outputDims,
                            const af::dim4& outputStrides, const void* input,
                            size_t inputBytes, const af::dim4& inputDims,
                            const af::dim4& inputStrides, int diagonal,
                            af_dtype type);

void launchMetalDiff(void* output, size_t outputBytes,
                     const af::dim4& outputDims, const af::dim4& outputStrides,
                     const void* input, size_t inputBytes,
                     const af::dim4& inputStrides, unsigned dimension,
                     bool secondOrder, af_dtype type);

void launchMetalTriangle(void* output, size_t outputBytes,
                         const af::dim4& outputDims,
                         const af::dim4& outputStrides, const void* input,
                         size_t inputBytes, const af::dim4& inputStrides,
                         bool upper, bool unitDiagonal, af_dtype type);

void launchMetalTranspose(void* output, size_t outputBytes,
                          const af::dim4& outputDims,
                          const af::dim4& outputStrides, const void* input,
                          size_t inputBytes, const af::dim4& inputStrides,
                          bool conjugate, af_dtype type);

void launchMetalTransposeInplace(void* input, size_t inputBytes,
                                 const af::dim4& dims, const af::dim4& strides,
                                 bool conjugate, af_dtype type);

void launchMetalUnwrap(
    void* output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, const void* input, size_t inputBytes,
    const af::dim4& inputDims, const af::dim4& inputStrides, dim_t windowX,
    dim_t windowY, dim_t strideX, dim_t strideY, dim_t paddingX, dim_t paddingY,
    dim_t dilationX, dim_t dilationY, unsigned columnDimension, af_dtype type);

void launchMetalWrap(void* output, size_t outputBytes,
                     const af::dim4& outputDims, const af::dim4& outputStrides,
                     const void* input, size_t inputBytes,
                     const af::dim4& inputDims, const af::dim4& inputStrides,
                     dim_t windowX, dim_t windowY, dim_t strideX, dim_t strideY,
                     dim_t paddingX, dim_t paddingY, dim_t dilationX,
                     dim_t dilationY, unsigned columnDimension, af_dtype type);

void launchMetalGradient(void* gradient0, const af::dim4& gradient0Strides,
                         void* gradient1, const af::dim4& gradient1Strides,
                         size_t outputBytes, const void* input,
                         size_t inputBytes, const af::dim4& inputDims,
                         const af::dim4& inputStrides, af_dtype type);

void launchMetalSobel(void* derivative0, const af::dim4& derivative0Strides,
                      void* derivative1, const af::dim4& derivative1Strides,
                      size_t outputBytes, const void* input, size_t inputBytes,
                      const af::dim4& inputDims, const af::dim4& inputStrides,
                      af_dtype inputType);

void launchMetalHsvRgb(void* output, size_t outputBytes,
                       const af::dim4& outputStrides, const void* input,
                       size_t inputBytes, const af::dim4& inputDims,
                       const af::dim4& inputStrides, bool hsvToRgb,
                       af_dtype type);

void launchMetalMoments(void* output, size_t outputBytes,
                        const af::dim4& outputStrides, const void* input,
                        size_t inputBytes, const af::dim4& inputDims,
                        const af::dim4& inputStrides, af_moment_type moment,
                        af_dtype type);

template<typename T>
void rangeMetal(Param<T> output, const unsigned sequenceDimension) {
    launchMetalRange(output.get(),
                     static_cast<size_t>(output.dims().elements()) * sizeof(T),
                     output.dims(), output.strides(), sequenceDimension,
                     static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void copyMetal(Param<T> output, CParam<T> input) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i)
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    launchMetalCopy(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.strides(), static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void hsvRgbMetal(Param<T> output, CParam<T> input, const bool hsvToRgb) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i)
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    launchMetalHsvRgb(output.get(),
                      static_cast<size_t>(output.dims().elements()) * sizeof(T),
                      output.strides(), input.get(), inputElements * sizeof(T),
                      input.dims(), input.strides(), hsvToRgb,
                      static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void momentsMetal(Param<float> output, CParam<T> input,
                  const af_moment_type moment) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i)
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    launchMetalMoments(
        output.get(),
        static_cast<size_t>(output.dims().elements()) * sizeof(float),
        output.strides(), input.get(), inputElements * sizeof(T), input.dims(),
        input.strides(), moment,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void resizeMetal(Param<T> output, CParam<T> input,
                 const af_interp_type method) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i)
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    launchMetalResize(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.dims(), input.strides(), method,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void padBordersMetal(Param<T> output, CParam<T> input,
                     const af::dim4 lowerPadding, const af::dim4,
                     const af::borderType borderType) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i)
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    launchMetalPadBorders(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.dims(), input.strides(), lowerPadding,
        static_cast<af_border_type>(borderType),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void iotaMetal(Param<T> output, const af::dim4& sourceDims) {
    launchMetalIota(output.get(),
                    static_cast<size_t>(output.dims().elements()) * sizeof(T),
                    output.dims(), output.strides(), sourceDims,
                    static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void identityMetal(Param<T> output) {
    launchMetalIdentity(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void tileMetal(Param<T> output, CParam<T> input) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalTile(output.get(),
                    static_cast<size_t>(output.dims().elements()) * sizeof(T),
                    output.dims(), output.strides(), input.get(),
                    inputElements * sizeof(T), input.dims(), input.strides(),
                    static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void shiftMetal(Param<T> output, CParam<T> input, const af::dim4 shifts) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalShift(output.get(),
                     static_cast<size_t>(output.dims().elements()) * sizeof(T),
                     output.dims(), output.strides(), input.get(),
                     inputElements * sizeof(T), input.strides(), shifts,
                     static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void reorderMetal(Param<T> output, CParam<T> input, const af::dim4,
                  const af::dim4 reorderDims) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalReorder(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.strides(), reorderDims,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void selectMetal(Param<T> output, CParam<char> condition, CParam<T> a,
                 CParam<T> b) {
    const auto span = [](const auto& param) {
        size_t elements = 1;
        for (int i = 0; i < 4; ++i) {
            elements += static_cast<size_t>(param.dims(i) - 1) *
                        static_cast<size_t>(param.strides(i));
        }
        return elements;
    };
    launchMetalSelect(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), condition.get(),
        span(condition) * sizeof(char), condition.dims(), condition.strides(),
        a.get(), span(a) * sizeof(T), a.dims(), a.strides(), b.get(),
        span(b) * sizeof(T), b.dims(), b.strides(), nullptr, false,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T, bool flip>
void selectScalarMetal(Param<T> output, CParam<char> condition, CParam<T> a,
                       const T scalar) {
    const auto span = [](const auto& param) {
        size_t elements = 1;
        for (int i = 0; i < 4; ++i) {
            elements += static_cast<size_t>(param.dims(i) - 1) *
                        static_cast<size_t>(param.strides(i));
        }
        return elements;
    };
    launchMetalSelect(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), condition.get(),
        span(condition) * sizeof(char), condition.dims(), condition.strides(),
        a.get(), span(a) * sizeof(T), a.dims(), a.strides(), nullptr, 0,
        af::dim4(1), af::dim4(1), &scalar, flip,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void joinMetal(const int dim, Param<T> output,
               const std::vector<CParam<T>> inputs, const int inputCount) {
    af::dim4 outputOffset(0, 0, 0, 0);
    bool wroteOutput = false;
    for (int inputIndex = 0; inputIndex < inputCount; ++inputIndex) {
        const CParam<T>& input = inputs[inputIndex];
        if (input.dims().elements() == 0) { continue; }
        size_t inputElements = 1;
        for (int i = 0; i < 4; ++i) {
            inputElements += static_cast<size_t>(input.dims(i) - 1) *
                             static_cast<size_t>(input.strides(i));
        }
        launchMetalJoinAppend(
            output.get(),
            static_cast<size_t>(output.dims().elements()) * sizeof(T),
            output.dims(), output.strides(), input.get(),
            inputElements * sizeof(T), input.dims(), input.strides(),
            outputOffset, wroteOutput,
            static_cast<af_dtype>(af::dtype_traits<T>::af_type));
        outputOffset[dim] += input.dims(dim);
        wroteOutput = true;
    }
}

template<typename InT, typename IndexT>
void lookupMetal(Param<InT> output, CParam<InT> input, CParam<IndexT> indices,
                 const unsigned dimension) {
    size_t inputElements = 1;
    size_t indexElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
        indexElements += static_cast<size_t>(indices.dims(i) - 1) *
                         static_cast<size_t>(indices.strides(i));
    }
    launchMetalLookup(
        output.get(),
        static_cast<size_t>(output.dims().elements()) * sizeof(InT),
        output.dims(), output.strides(), input.get(),
        inputElements * sizeof(InT), input.dims(), input.strides(),
        indices.get(), indexElements * sizeof(IndexT), dimension,
        static_cast<af_dtype>(af::dtype_traits<InT>::af_type),
        static_cast<af_dtype>(af::dtype_traits<IndexT>::af_type));
}

template<typename T>
void diagCreateMetal(Param<T> output, CParam<T> input, const int diagonal) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalDiagCreate(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.dims(), input.strides(), diagonal,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void diagExtractMetal(Param<T> output, CParam<T> input, const int diagonal) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalDiagExtract(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.dims(), input.strides(), diagonal,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void diff1Metal(Param<T> output, CParam<T> input, const int dimension) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalDiff(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.strides(), static_cast<unsigned>(dimension), false,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void diff2Metal(Param<T> output, CParam<T> input, const int dimension) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalDiff(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.strides(), static_cast<unsigned>(dimension), true,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void triangleMetal(Param<T> output, CParam<T> input, const bool upper,
                   const bool unitDiagonal) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalTriangle(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.strides(), upper, unitDiagonal,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void transposeMetal(Param<T> output, CParam<T> input, const bool conjugate) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalTranspose(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.strides(), conjugate,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void transposeInplaceMetal(Param<T> input, const bool conjugate) {
    launchMetalTransposeInplace(
        input.get(), static_cast<size_t>(input.dims().elements()) * sizeof(T),
        input.dims(), input.strides(), conjugate,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void unwrapMetal(Param<T> output, CParam<T> input, const dim_t windowX,
                 const dim_t windowY, const dim_t strideX, const dim_t strideY,
                 const dim_t paddingX, const dim_t paddingY,
                 const dim_t dilationX, const dim_t dilationY,
                 const int columnDimension) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalUnwrap(
        output.get(), static_cast<size_t>(output.dims().elements()) * sizeof(T),
        output.dims(), output.strides(), input.get(), inputElements * sizeof(T),
        input.dims(), input.strides(), windowX, windowY, strideX, strideY,
        paddingX, paddingY, dilationX, dilationY,
        static_cast<unsigned>(columnDimension),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void wrapMetal(Param<T> output, CParam<T> input, const dim_t windowX,
               const dim_t windowY, const dim_t strideX, const dim_t strideY,
               const dim_t paddingX, const dim_t paddingY,
               const dim_t dilationX, const dim_t dilationY,
               const int columnDimension) {
    size_t inputElements  = 1;
    size_t outputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
        outputElements += static_cast<size_t>(output.dims(i) - 1) *
                          static_cast<size_t>(output.strides(i));
    }
    launchMetalWrap(output.get(), outputElements * sizeof(T), output.dims(),
                    output.strides(), input.get(), inputElements * sizeof(T),
                    input.dims(), input.strides(), windowX, windowY, strideX,
                    strideY, paddingX, paddingY, dilationX, dilationY,
                    static_cast<unsigned>(columnDimension),
                    static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename T>
void gradientMetal(Param<T> gradient0, Param<T> gradient1, CParam<T> input) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalGradient(
        gradient0.get(), gradient0.strides(), gradient1.get(),
        gradient1.strides(),
        static_cast<size_t>(gradient0.dims().elements()) * sizeof(T),
        input.get(), inputElements * sizeof(T), input.dims(), input.strides(),
        static_cast<af_dtype>(af::dtype_traits<T>::af_type));
}

template<typename Ti, typename To>
void sobelMetal(Param<To> derivative0, Param<To> derivative1,
                CParam<Ti> input) {
    size_t inputElements = 1;
    for (int i = 0; i < 4; ++i) {
        inputElements += static_cast<size_t>(input.dims(i) - 1) *
                         static_cast<size_t>(input.strides(i));
    }
    launchMetalSobel(
        derivative0.get(), derivative0.strides(), derivative1.get(),
        derivative1.strides(),
        static_cast<size_t>(derivative0.dims().elements()) * sizeof(To),
        input.get(), inputElements * sizeof(Ti), input.dims(), input.strides(),
        static_cast<af_dtype>(af::dtype_traits<Ti>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
