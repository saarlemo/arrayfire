/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <metal_stdlib>
using namespace metal;

template<typename T>
inline bool sortOrdered(T previous, T value, uint ascending) {
    return ascending != 0 ? previous <= value : previous >= value;
}
