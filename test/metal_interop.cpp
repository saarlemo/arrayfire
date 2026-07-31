/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <gtest/gtest.h>
#include <testHelpers.hpp>
#include <af/metal.h>

#include <objc/message.h>
#include <objc/runtime.h>
#include <cstdint>
#include <vector>

namespace {

using NativeObject = void*;

struct NativeRange {
    size_t location;
    size_t length;
};

NativeObject sendObject(NativeObject object, const char* selector) {
    using Function = NativeObject (*)(NativeObject, SEL);
    return reinterpret_cast<Function>(objc_msgSend)(object,
                                                    sel_registerName(selector));
}

void sendVoid(NativeObject object, const char* selector) {
    using Function = void (*)(NativeObject, SEL);
    reinterpret_cast<Function>(objc_msgSend)(object,
                                             sel_registerName(selector));
}

NativeObject newSharedBuffer(af_mtl_device device, const void* data,
                             size_t bytes) {
    using Function =
        NativeObject (*)(NativeObject, SEL, const void*, size_t, size_t);
    return reinterpret_cast<Function>(objc_msgSend)(
        reinterpret_cast<NativeObject>(device),
        sel_registerName("newBufferWithBytes:length:options:"), data, bytes,
        size_t{0});
}

void fillBuffer(NativeObject encoder, af_mtl_buffer buffer, size_t bytes,
                uint8_t value) {
    using Function =
        void (*)(NativeObject, SEL, NativeObject, NativeRange, uint8_t);
    reinterpret_cast<Function>(objc_msgSend)(
        encoder, sel_registerName("fillBuffer:range:value:"),
        reinterpret_cast<NativeObject>(buffer), NativeRange{0, bytes}, value);
}

NativeObject newAutoreleasePool() {
    NativeObject poolClass =
        reinterpret_cast<NativeObject>(objc_getClass("NSAutoreleasePool"));
    return sendObject(sendObject(poolClass, "alloc"), "init");
}

void selectMetalBackend() {
#ifdef AF_UNIFIED
    ASSERT_SUCCESS(af_set_backend(AF_BACKEND_METAL));
#endif
}

}  // namespace

TEST(MetalInterop, NativeDeviceAndQueue) {
    selectMetalBackend();

    int active = -1;
    ASSERT_SUCCESS(af_get_device(&active));

    af_mtl_device device = nullptr;
    ASSERT_SUCCESS(afmtl_get_device(&device, active, false));
    ASSERT_NE(nullptr, device);
    ASSERT_EQ(device, afmtl::getDevice(active));

    af_mtl_command_queue queue = nullptr;
    ASSERT_SUCCESS(afmtl_get_command_queue(&queue, active, false));
    ASSERT_NE(nullptr, queue);
    ASSERT_EQ(queue, afmtl::getQueue(active));

    afmtl::setDevice(device);
    int selected = -1;
    ASSERT_SUCCESS(af_get_device(&selected));
    ASSERT_EQ(active, selected);

    MTL::Device* retainedDevice      = afmtl::getDevice(active, true);
    MTL::CommandQueue* retainedQueue = afmtl::getQueue(active, true);
    sendVoid(reinterpret_cast<NativeObject>(retainedQueue), "release");
    sendVoid(reinterpret_cast<NativeObject>(retainedDevice), "release");
}

TEST(MetalInterop, BufferRoundTripAndOffset) {
    selectMetalBackend();

    constexpr dim_t elements = 16;
    std::vector<float> values(elements);
    for (dim_t i = 0; i < elements; ++i) {
        values[static_cast<size_t>(i)] = static_cast<float>(i);
    }

    MTL::Device* device = afmtl::getDevice();
    MTL::Buffer* native = reinterpret_cast<MTL::Buffer*>(
        newSharedBuffer(device, values.data(), values.size() * sizeof(float)));
    ASSERT_NE(nullptr, native);

    af::array parent = afmtl::array(af::dim4(4, 4), native, f32);
    sendVoid(reinterpret_cast<NativeObject>(native), "release");

    afmtl::BufferView parentView = afmtl::getBuffer(parent);
    ASSERT_EQ(native, parentView.buffer);
    ASSERT_EQ(0u, parentView.offset);

    af::array slice             = parent.rows(1, 2);
    afmtl::BufferView sliceView = afmtl::getBuffer(slice);
    ASSERT_EQ(parentView.buffer, sliceView.buffer);
    ASSERT_EQ(sizeof(float), sliceView.offset);

    std::vector<float> actual(8);
    slice.host(actual.data());
    for (size_t i = 0; i < actual.size(); ++i) {
        const size_t column = i / 2;
        const size_t row    = i % 2 + 1;
        ASSERT_EQ(values[column * 4 + row], actual[i]);
    }
}

TEST(MetalInterop, DevicePointerIsNativeBuffer) {
    selectMetalBackend();

    af::array input              = af::constant(1, 8, f32);
    afmtl::BufferView bufferView = afmtl::getBuffer(input);

    void* devicePointer = nullptr;
    ASSERT_SUCCESS(af_get_device_ptr(&devicePointer, input.get()));
    ASSERT_EQ(bufferView.buffer, devicePointer);
    ASSERT_TRUE(input.isLocked());

    ASSERT_SUCCESS(af_unlock_array(input.get()));
    ASSERT_FALSE(input.isLocked());
}

TEST(MetalInterop, SubmittedWorkParticipatesInSync) {
    selectMetalBackend();

    constexpr size_t bytes = 32;
    std::vector<unsigned char> zeros(bytes, 0);
    MTL::Device* device = afmtl::getDevice();
    MTL::Buffer* native = reinterpret_cast<MTL::Buffer*>(
        newSharedBuffer(device, zeros.data(), bytes));
    ASSERT_NE(nullptr, native);

    af::array wrapped =
        afmtl::array(af::dim4(static_cast<dim_t>(bytes)), native, u8);
    sendVoid(reinterpret_cast<NativeObject>(native), "release");

    NativeObject pool                 = newAutoreleasePool();
    MTL::CommandBuffer* commandBuffer = reinterpret_cast<MTL::CommandBuffer*>(
        sendObject(reinterpret_cast<NativeObject>(afmtl::getQueue()),
                   "commandBuffer"));
    ASSERT_NE(nullptr, commandBuffer);
    NativeObject encoder = sendObject(
        reinterpret_cast<NativeObject>(commandBuffer), "blitCommandEncoder");
    ASSERT_NE(nullptr, encoder);
    fillBuffer(encoder, native, bytes, uint8_t{42});
    sendVoid(encoder, "endEncoding");

    afmtl::submit(commandBuffer);
    af::sync();

    std::vector<unsigned char> actual(bytes);
    wrapped.host(actual.data());
    for (const unsigned char value : actual) { ASSERT_EQ(42, value); }
    sendVoid(pool, "release");
}

TEST(MetalInterop, RejectsInvalidArguments) {
    selectMetalBackend();

    ASSERT_EQ(AF_ERR_ARG, afmtl_get_device(nullptr, -1, false));
    ASSERT_EQ(AF_ERR_ARG, afmtl_get_command_queue(nullptr, -1, false));
    ASSERT_EQ(AF_ERR_ARG, afmtl_set_device(nullptr));
    ASSERT_EQ(AF_ERR_ARG, afmtl_get_buffer(nullptr, nullptr, nullptr));
    ASSERT_EQ(AF_ERR_ARG, afmtl_release_buffer(nullptr));
    ASSERT_EQ(AF_ERR_ARG, afmtl_submit_command_buffer(nullptr, -1));
}

TEST(MetalInterop, RejectsUnsupportedDoubleArrays) {
    selectMetalBackend();

    if (af::isDoubleAvailable(af::getDevice())) {
        GTEST_SKIP() << "Metal device reports double-precision support";
    }

    try {
        af::array result = af::constant(1.0, 1, f64);
        (void)result;
        FAIL() << "Metal accepted an unsupported double-precision array";
    } catch (af::exception& err) {
        EXPECT_EQ(AF_ERR_NO_DBL, err.err());
    }
}

#ifdef AF_UNIFIED
TEST(MetalInterop, BufferViewReleaseSurvivesBackendSwitch) {
    selectMetalBackend();

    int available = 0;
    ASSERT_SUCCESS(af_get_available_backends(&available));
    if (!(available & AF_BACKEND_CPU)) {
        GTEST_SKIP() << "CPU backend is not available in this build";
    }

    af::array input        = af::constant(1, 8, f32);
    afmtl::BufferView view = afmtl::getBuffer(input);
    ASSERT_NE(nullptr, view.buffer);

    ASSERT_SUCCESS(af_set_backend(AF_BACKEND_CPU));
    view.reset();
    ASSERT_SUCCESS(af_set_backend(AF_BACKEND_METAL));
}
#endif
