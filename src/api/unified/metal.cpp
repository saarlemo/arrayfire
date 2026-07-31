/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/backend.h>
#include <af/metal.h>

#include "symbol_manager.hpp"

af_err afmtl_get_device(af_mtl_device* device, int id, const bool retain) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_METAL) {
        CALL(afmtl_get_device, device, id, retain);
    }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afmtl_get_command_queue(af_mtl_command_queue* queue, int id,
                               const bool retain) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_METAL) {
        CALL(afmtl_get_command_queue, queue, id, retain);
    }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afmtl_set_device(af_mtl_device device) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_METAL) { CALL(afmtl_set_device, device); }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afmtl_get_buffer(af_mtl_buffer* buffer, size_t* offset,
                        const af_array arr) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_METAL) {
        CHECK_ARRAYS(arr);
        CALL(afmtl_get_buffer, buffer, offset, arr);
    }
    return AF_ERR_NOT_SUPPORTED;
}

af_err afmtl_release_buffer(af_mtl_buffer buffer) {
    using namespace arrayfire::unified;
    LibHandle handle = AFSymbolManager::getInstance().getHandle(
        backend_index(AF_BACKEND_METAL));
    if (!handle) { return AF_ERR_NOT_SUPPORTED; }

    using af_func    = std::add_pointer<decltype(afmtl_release_buffer)>::type;
    af_func function = reinterpret_cast<af_func>(
        arrayfire::common::getFunctionPointer(handle, __func__));
    if (!function) {
        AF_RETURN_ERROR(
            "requested symbol name could not be found in loaded library.",
            AF_ERR_LOAD_LIB);
    }
    return function(buffer);
}

af_err afmtl_submit_command_buffer(af_mtl_command_buffer command_buffer,
                                   int id) {
    af_backend backend;
    af_get_active_backend(&backend);
    if (backend == AF_BACKEND_METAL) {
        CALL(afmtl_submit_command_buffer, command_buffer, id);
    }
    return AF_ERR_NOT_SUPPORTED;
}
