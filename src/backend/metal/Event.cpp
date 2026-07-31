/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Event.hpp>

#include <common/err_common.hpp>
#include <events.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <af/event.h>
#include <limits>
#include <memory>

using std::make_unique;

namespace arrayfire {
namespace metal {

namespace {

void signalEvent(MTL::SharedEvent *event, const std::uint64_t value) {
    auto commandBuffer = NS::RetainPtr(getCommandQueue().commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    commandBuffer->encodeSignalEvent(event, value);
    submitCommandBuffer(commandBuffer.get());
    event->release();
}

void encodeWaitForEvent(MTL::SharedEvent *event, const std::uint64_t value) {
    auto commandBuffer = NS::RetainPtr(getCommandQueue().commandBuffer());
    if (!commandBuffer) {
        AF_ERROR("Could not create a Metal command buffer", AF_ERR_RUNTIME);
    }
    commandBuffer->encodeWait(event, value);
    submitCommandBuffer(commandBuffer.get());
    event->release();
}

}  // namespace

int MetalEventPolicy::createAndMarkEvent(MetalEventData *e) noexcept {
    if (!e) { return -1; }
    try {
        MTL::Device &device = getDevice();
        e->event            = device.newSharedEvent();
        e->value            = 0;
        return e->event ? 0 : -1;
    } catch (...) { return -1; }
}

int MetalEventPolicy::markEvent(MetalEventData *e,
                                metal::queue &stream) noexcept {
    if (!e || !e->event) { return -1; }
    MTL::SharedEvent *event = e->event;
    event->retain();
    try {
        const std::uint64_t value = ++e->value;
        stream.enqueueNative(signalEvent, event, value);
        return 0;
    } catch (...) {
        event->release();
        return -1;
    }
}

int MetalEventPolicy::waitForEvent(MetalEventData *e,
                                   metal::queue &stream) noexcept {
    if (!e || !e->event || e->value == 0) { return -1; }
    MTL::SharedEvent *event = e->event;
    event->retain();
    try {
        stream.enqueueNative(encodeWaitForEvent, event, e->value);
        return 0;
    } catch (...) {
        event->release();
        return -1;
    }
}

int MetalEventPolicy::syncForEvent(MetalEventData *e) noexcept {
    if (!e || !e->event || e->value == 0) { return -1; }
    return e->event->waitUntilSignaledValue(
               e->value, std::numeric_limits<std::uint64_t>::max())
               ? 0
               : -1;
}

int MetalEventPolicy::destroyEvent(MetalEventData *e) noexcept {
    if (e && e->event) {
        e->event->release();
        e->event = nullptr;
        e->value = 0;
    }
    return 0;
}

/// \brief Creates a new event and marks it in the queue
Event makeEvent(metal::queue& queue) {
    Event e;
    if (0 == e.create()) { e.mark(queue); }
    return e;
}

af_event createEvent() {
    auto e = make_unique<Event>();
    // Ensure that the default queue is initialized
    getQueue();
    if (e->create() != 0) {
        AF_ERROR("Could not create event", AF_ERR_RUNTIME);
    }
    Event& ref = *e.release();
    return getHandle(ref);
}

void markEventOnActiveQueue(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active queue
    if (event.mark(getQueue()) != 0) {
        AF_ERROR("Could not mark event on active queue", AF_ERR_RUNTIME);
    }
}

void enqueueWaitOnActiveQueue(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    // Use the currently-active queue
    if (event.enqueueWait(getQueue()) != 0) {
        AF_ERROR("Could not enqueue wait on active queue for event",
                 AF_ERR_RUNTIME);
    }
}

void block(af_event eventHandle) {
    Event& event = getEvent(eventHandle);
    if (event.block() != 0) {
        AF_ERROR("Could not block on active queue for event", AF_ERR_RUNTIME);
    }
}

af_event createAndMarkEvent() {
    af_event handle = createEvent();
    markEventOnActiveQueue(handle);
    return handle;
}

}  // namespace metal
}  // namespace arrayfire
