/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once
#include <Param.hpp>
#include <af/traits.hpp>

namespace arrayfire {
namespace metal {
namespace kernel {

bool supportsMetalNearestNeighbour(af_dtype inputType, af_dtype outputType,
                                   af_match_type distanceType) noexcept;

void launchMetalNearestNeighbour(
    BufferParam output, size_t outputBytes, const af::dim4& outputDims,
    const af::dim4& outputStrides, BufferParam query, size_t queryBytes,
    const af::dim4& queryDims, const af::dim4& queryStrides, BufferParam train,
    size_t trainBytes, const af::dim4& trainDims,
    const af::dim4& trainStrides, unsigned distanceDimension,
    af_match_type distanceType, af_dtype inputType, af_dtype outputType);

template<typename T, typename To>
void nearestNeighbourMetal(Param<To> output, CParam<T> query, CParam<T> train,
                           const unsigned distanceDimension,
                           const af_match_type distanceType) {
    size_t queryElements = 1, trainElements = 1;
    for (int i = 0; i < 4; ++i) {
        queryElements += static_cast<size_t>(query.dims(i) - 1) *
                         static_cast<size_t>(query.strides(i));
        trainElements += static_cast<size_t>(train.dims(i) - 1) *
                         static_cast<size_t>(train.strides(i));
    }
    launchMetalNearestNeighbour(
        output.bufferParam(),
        static_cast<size_t>(output.dims().elements()) * sizeof(To),
        output.dims(), output.strides(), query.bufferParam(),
        queryElements * sizeof(T), query.dims(), query.strides(),
        train.bufferParam(), trainElements * sizeof(T), train.dims(),
        train.strides(), distanceDimension, distanceType,
        static_cast<af_dtype>(af::dtype_traits<T>::af_type),
        static_cast<af_dtype>(af::dtype_traits<To>::af_type));
}

}  // namespace kernel
}  // namespace metal
}  // namespace arrayfire
