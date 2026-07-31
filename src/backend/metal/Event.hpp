/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <Metal.hpp>
#include <common/EventBase.hpp>
#include <queue.hpp>
#include <af/event.h>

#include <cstdint>
#include <type_traits>

namespace arrayfire {
namespace metal {

struct MetalEventData {
    MTL::SharedEvent *event{nullptr};
    std::uint64_t value{0};

    MetalEventData() noexcept = default;
    MetalEventData(int) noexcept {}
    MetalEventData(const MetalEventData &)            = delete;
    MetalEventData &operator=(const MetalEventData &) = delete;

    MetalEventData(MetalEventData &&other) noexcept
        : event(other.event), value(other.value) {
        other.event = nullptr;
        other.value = 0;
    }

    MetalEventData &operator=(MetalEventData &&other) noexcept {
        if (this != &other) {
            if (event) { event->release(); }
            event       = other.event;
            value       = other.value;
            other.event = nullptr;
            other.value = 0;
        }
        return *this;
    }

    MetalEventData &operator=(int) noexcept {
        if (event) { event->release(); }
        event = nullptr;
        value = 0;
        return *this;
    }

    explicit operator bool() const noexcept { return event != nullptr; }
};

class MetalEventPolicy {
   public:
    using EventType = MetalEventData;
    using QueueType = std::add_lvalue_reference<queue>::type;
    using ErrorType = int;

    static int createAndMarkEvent(MetalEventData *e) noexcept;

    static int markEvent(MetalEventData *e, metal::queue &stream) noexcept;

    static int waitForEvent(MetalEventData *e, metal::queue &stream) noexcept;

    static int syncForEvent(MetalEventData *e) noexcept;

    static int destroyEvent(MetalEventData *e) noexcept;
};

using Event = common::EventBase<MetalEventPolicy>;

/// \brief Creates a new event and marks it in the queue
Event makeEvent(metal::queue &queue);

af_event createEvent();

void markEventOnActiveQueue(af_event eventHandle);

void enqueueWaitOnActiveQueue(af_event eventHandle);

void block(af_event eventHandle);

af_event createAndMarkEvent();

}  // namespace metal
}  // namespace arrayfire
