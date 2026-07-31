/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <af/defines.h>

#include <stddef.h>

#ifdef __cplusplus
namespace MTL {
class Buffer;
class CommandBuffer;
class CommandQueue;
class Device;
} // namespace MTL

typedef MTL::Buffer *af_mtl_buffer;
typedef MTL::CommandBuffer *af_mtl_command_buffer;
typedef MTL::CommandQueue *af_mtl_command_queue;
typedef MTL::Device *af_mtl_device;
#else
typedef void *af_mtl_buffer;
typedef void *af_mtl_command_buffer;
typedef void *af_mtl_command_queue;
typedef void *af_mtl_device;
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
   Get a handle to an ArrayFire Metal device.

   \param[out] device native Metal device
   \param[in] id ArrayFire device id, or -1 for the active device
   \param[in] retain if true, retain the returned device for the caller
   \returns \ref af_err error code

   When \p retain is true, the caller owns a reference and must release it.

   \ingroup metal_mat
 */
AFAPI af_err afmtl_get_device(af_mtl_device *device, int id, const bool retain);

/**
   Get a handle to an ArrayFire Metal command queue.

   \param[out] queue native Metal command queue
   \param[in] id ArrayFire device id, or -1 for the active device
   \param[in] retain if true, retain the returned command queue for the caller
   \returns \ref af_err error code

   Custom command buffers intended to participate in ArrayFire synchronization
   should be passed to \ref afmtl_submit_command_buffer rather than committed
   directly. When \p retain is true, the caller owns a reference and must
   release it.

   \ingroup metal_mat
 */
AFAPI af_err afmtl_get_command_queue(af_mtl_command_queue *queue, int id,
                                     const bool retain);

/**
   Set ArrayFire's active device from a native Metal device.

   The device must refer to one of the devices exposed by ArrayFire.

   \param[in] device native Metal device
   \returns \ref af_err error code

   \ingroup metal_mat
 */
AFAPI af_err afmtl_set_device(af_mtl_device device);

/**
   Get the Metal buffer and byte offset backing an ArrayFire array.

   This function evaluates \p arr and acquires an interop reference to its
   allocation. Call \ref afmtl_release_buffer when native access is complete.
   Managed allocations remain locked against reuse, while externally supplied
   buffers are retained.

   \param[out] buffer native Metal buffer
   \param[out] offset byte offset of the first array element in \p buffer
   \param[in] arr ArrayFire array
   \returns \ref af_err error code

   \ingroup metal_mat
 */
AFAPI af_err afmtl_get_buffer(af_mtl_buffer *buffer, size_t *offset,
                              const af_array arr);

/**
   Release a buffer reference acquired by \ref afmtl_get_buffer.

   \param[in] buffer native Metal buffer
   \returns \ref af_err error code

   \ingroup metal_mat
 */
AFAPI af_err afmtl_release_buffer(af_mtl_buffer buffer);

/**
   Commit a custom Metal command buffer and register it with ArrayFire.

   The command buffer must have been created from the command queue returned
   for \p id and must not have been committed already. Registering it ensures
   that ArrayFire operations remain ordered with the custom work and that
   \ref af_sync waits for its completion.

   \param[in] command_buffer native Metal command buffer
   \param[in] id ArrayFire device id, or -1 for the active device
   \returns \ref af_err error code

   \ingroup metal_mat
 */
AFAPI af_err afmtl_submit_command_buffer(af_mtl_command_buffer command_buffer,
                                         int id);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <af/array.h>
#include <af/device.h>
#include <af/dim4.hpp>
#include <af/exception.h>

namespace afmtl {

class BufferView {
public:
  BufferView() noexcept : buffer(nullptr), offset(0) {}
  BufferView(MTL::Buffer *buffer, size_t offset) noexcept
      : buffer(buffer), offset(offset) {}
  BufferView(const BufferView &) = delete;
  BufferView &operator=(const BufferView &) = delete;
  BufferView(BufferView &&other) noexcept
      : buffer(other.buffer), offset(other.offset) {
    other.buffer = nullptr;
    other.offset = 0;
  }
  BufferView &operator=(BufferView &&other) noexcept {
    if (this != &other) {
      reset();
      buffer = other.buffer;
      offset = other.offset;
      other.buffer = nullptr;
      other.offset = 0;
    }
    return *this;
  }
  ~BufferView() { reset(); }

  void reset() noexcept {
    if (buffer) {
      afmtl_release_buffer(buffer);
    }
    buffer = nullptr;
    offset = 0;
  }

  MTL::Buffer *buffer;
  size_t offset;
};

/**
   Get an ArrayFire Metal device.

   \param[in] id ArrayFire device id, or -1 for the active device
   \param[in] retain if true, retain the returned device for the caller
   \returns native Metal device

   \ingroup metal_mat
 */
static inline MTL::Device *getDevice(int id = -1, bool retain = false) {
  MTL::Device *device = nullptr;
  af_err err = afmtl_get_device(&device, id, retain);
  if (err != AF_SUCCESS) {
    throw af::exception("Failed to get Metal device from ArrayFire");
  }
  return device;
}

/**
   Get an ArrayFire Metal command queue.

   \param[in] id ArrayFire device id, or -1 for the active device
   \param[in] retain if true, retain the returned queue for the caller
   \returns native Metal command queue

   \ingroup metal_mat
 */
static inline MTL::CommandQueue *getQueue(int id = -1, bool retain = false) {
  MTL::CommandQueue *queue = nullptr;
  af_err err = afmtl_get_command_queue(&queue, id, retain);
  if (err != AF_SUCCESS) {
    throw af::exception("Failed to get Metal command queue from ArrayFire");
  }
  return queue;
}

/**
   Set ArrayFire's active device from a native Metal device.

   \ingroup metal_mat
 */
static inline void setDevice(MTL::Device *device) {
  af_err err = afmtl_set_device(device);
  if (err != AF_SUCCESS) {
    throw af::exception("Failed to set ArrayFire Metal device");
  }
}

/**
   Get the buffer and byte offset backing an ArrayFire array.

   The returned move-only view keeps managed memory locked against reuse and
   retains external buffers. Its destructor releases that interop reference.

   \ingroup metal_mat
 */
static inline BufferView getBuffer(const af::array &arr) {
  MTL::Buffer *buffer = nullptr;
  size_t offset = 0;
  af_err err = afmtl_get_buffer(&buffer, &offset, arr.get());
  if (err != AF_SUCCESS) {
    throw af::exception("Failed to get Metal buffer from ArrayFire");
  }
  return BufferView(buffer, offset);
}

/**
   Create an ArrayFire array backed by a native Metal buffer.

   ArrayFire retains \p buffer. The input must be a contiguous allocation on
   the active ArrayFire Metal device.

   \ingroup metal_mat
 */
static inline af::array array(af::dim4 dims, MTL::Buffer *buffer,
                              af::dtype type) {
  af_array out = 0;
  const unsigned ndims = static_cast<unsigned>(dims.ndims());
  af_err err = af_device_array(&out, static_cast<void *>(buffer), ndims,
                               dims.get(), type);
  if (err != AF_SUCCESS) {
    throw af::exception("Failed to create array from Metal buffer");
  }
  return af::array(out);
}

/**
   Submit custom Metal work through ArrayFire's synchronization path.

   \ingroup metal_mat
 */
static inline void submit(MTL::CommandBuffer *commandBuffer, int id = -1) {
  af_err err = afmtl_submit_command_buffer(commandBuffer, id);
  if (err != AF_SUCCESS) {
    throw af::exception("Failed to submit Metal command buffer");
  }
}

} // namespace afmtl

#endif
