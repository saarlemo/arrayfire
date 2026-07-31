/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <common/jit/ShiftNodeBase.hpp>
#include <jit/BufferNode.hpp>

namespace arrayfire {
namespace metal {
namespace jit {

using ShiftNode = common::ShiftNodeBase<BufferNode>;

}  // namespace jit
}  // namespace metal
}  // namespace arrayfire
