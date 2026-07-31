/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <kernel/KParam.hpp>
#include <af/dim4.hpp>

#include <cstddef>
#include <type_traits>

namespace MTL {
class Buffer;
}

namespace arrayfire {
namespace metal {

struct BufferParam {
    MTL::Buffer *buffer;
    size_t offset;
};

inline KParam makeKParam(const af::dim4 &dims, const af::dim4 &strides,
                         const dim_t offset) noexcept {
    return {{dims[0], dims[1], dims[2], dims[3]},
            {strides[0], strides[1], strides[2], strides[3]}, offset};
}

template<typename T>
class Param {
    template<typename>
    friend class Param;

   public:
    using value_type = T;

    Param() noexcept
        : m_buffer(nullptr), m_info{{0, 0, 0, 0}, {0, 0, 0, 0}, 0} {}

    Param(MTL::Buffer *buffer, const size_t offset, const af::dim4 &dims,
          const af::dim4 &strides) noexcept
        : m_buffer(buffer)
        , m_info(makeKParam(dims, strides, static_cast<dim_t>(offset))) {}

    template<
        typename U,
        typename = std::enable_if_t<std::is_convertible_v<U *, T *>>>
    Param(const Param<U> &other) noexcept
        : m_buffer(other.m_buffer), m_info(other.m_info) {}

    MTL::Buffer *getBuffer() const noexcept { return m_buffer; }

    dim_t getOffset() const noexcept { return m_info.offset; }

    BufferParam bufferParam() const noexcept {
        using Element = std::remove_const_t<T>;
        return {m_buffer,
                static_cast<size_t>(m_info.offset) * sizeof(Element)};
    }

    af::dim4 dims() const noexcept {
        return af::dim4(m_info.dims[0], m_info.dims[1], m_info.dims[2],
                        m_info.dims[3]);
    }

    af::dim4 strides() const noexcept {
        return af::dim4(m_info.strides[0], m_info.strides[1],
                        m_info.strides[2], m_info.strides[3]);
    }

    constexpr dim_t dims(const int index) const noexcept {
        return m_info.dims[index];
    }

    constexpr dim_t strides(const int index) const noexcept {
        return m_info.strides[index];
    }

    const KParam &info() const noexcept { return m_info; }

   private:
    MTL::Buffer *m_buffer;
    KParam m_info;
};

template<typename T>
using CParam = Param<const T>;

}  // namespace metal
}  // namespace arrayfire
