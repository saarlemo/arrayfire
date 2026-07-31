/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <utility>

namespace arrayfire {
namespace metal {

void syncCommandQueue(int device);

/// Per-device submission facade. Native work is encoded directly onto the
/// Metal command queue; host-side fallbacks are handled explicitly by their
/// callers.
class queue {
   public:
    explicit queue(const int device = 0) : device(device) {}

    template<typename F, typename... Args>
    void enqueueNative(const F func, Args &&...args) {
        func(std::forward<Args>(args)...);
    }

    void sync() { syncCommandQueue(device); }

    bool is_worker() const { return false; }

   private:
    const int device;
};
}  // namespace metal
}  // namespace arrayfire
