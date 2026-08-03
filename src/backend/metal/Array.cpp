/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Metal.hpp>

#include <Array.hpp>

#include <Param.hpp>
#include <common/ArrayInfo.hpp>
#include <common/err_common.hpp>
#include <common/half.hpp>
#include <common/jit/NodeIterator.hpp>
#include <common/jit/ScalarNode.hpp>
#include <common/traits.hpp>
#include <copy.hpp>
#include <jit/BufferNode.hpp>
#include <kernel/KParam.hpp>
#include <memory.hpp>
#include <kernel/Array.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <traits.hpp>

#include <af/defines.h>
#include <af/dim4.hpp>
#include <af/seq.h>
#include <af/traits.hpp>

#include <nonstd/span.hpp>
#include <algorithm>  // IWYU pragma: keep
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <utility>

using af::dim4;
using arrayfire::common::half;
using arrayfire::common::Node;
using arrayfire::common::Node_map_t;
using arrayfire::common::Node_ptr;
using arrayfire::common::NodeIterator;
using arrayfire::metal::jit::BufferNode;

using nonstd::span;
using std::accumulate;
using std::adjacent_find;
using std::copy;
using std::find_if;
using std::is_standard_layout;
using std::make_shared;
using std::move;
using std::vector;

namespace arrayfire {
namespace metal {

namespace {

template<typename T>
void verifyTypeSupport() {}

template<>
void verifyTypeSupport<double>() {
    if (!isDoubleSupported(getActiveDeviceId())) {
        AF_ERROR("Double precision not supported", AF_ERR_NO_DBL);
    }
}

template<>
void verifyTypeSupport<cdouble>() {
    if (!isDoubleSupported(getActiveDeviceId())) {
        AF_ERROR("Double precision not supported", AF_ERR_NO_DBL);
    }
}

template<>
void verifyTypeSupport<common::half>() {
    if (!isHalfSupported(getActiveDeviceId())) {
        AF_ERROR("Half precision not supported", AF_ERR_NO_HALF);
    }
}

}  // namespace

std::shared_ptr<MTL::Buffer> managedBuffer(MTL::Buffer *buffer,
                                            const int device) {
    return std::shared_ptr<MTL::Buffer>(
        buffer, [device](MTL::Buffer *value) {
            const int previous = setDevice(device);
            memFree(value);
            if (previous >= 0) { setDevice(previous); }
        });
}

template<typename T>
shared_ptr<BufferNode> bufferNodePtr() {
    return std::make_shared<BufferNode>(
        static_cast<af::dtype>(dtype_traits<T>::af_type));
}

template<typename T>
Array<T>::Array(dim4 dims)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(managedBuffer(memAlloc<T>(dims.elements()).release(),
                         getActiveDeviceId()))
    , data_dims(dims)
    , node()
    , owner(true) {}

template<typename T>
Array<T>::Array(const dim4 &dims, T *const in_data, bool is_device,
                bool copy_device)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data()
    , data_dims(dims)
    , node()
    , owner(true) {
    static_assert(is_standard_layout<Array<T>>::value,
                  "Array<T> must be a standard layout type");
    static_assert(std::is_nothrow_move_assignable<Array<T>>::value,
                  "Array<T> is not move assignable");
    static_assert(std::is_nothrow_move_constructible<Array<T>>::value,
                  "Array<T> is not move constructible");
    static_assert(
        offsetof(Array<T>, info) == 0,
        "Array<T>::info must be the first member variable of Array<T>");
    if (is_device && !copy_device) {
        auto *buffer = reinterpret_cast<MTL::Buffer *>(in_data);
        retainBuffer(buffer);
        data = shared_ptr<MTL::Buffer>(
            buffer, [](MTL::Buffer *value) {
                releaseBuffer(value);
            });
    } else {
        data = managedBuffer(memAlloc<T>(dims.elements()).release(),
                             getActiveDeviceId());
        if (is_device) {
            getQueue().sync();
            if (!copyBuffer(
                    data.get(), 0, reinterpret_cast<MTL::Buffer *>(in_data), 0,
                    static_cast<size_t>(dims.elements()) * sizeof(T))) {
                AF_ERROR("Could not copy Metal buffer", AF_ERR_RUNTIME);
            }
        } else {
            // Ensure the memory being written to isnt used anywhere else.
            getQueue().sync();
            copy(in_data, in_data + dims.elements(), getHostPtr(false));
        }
    }
}

template<typename T>
Array<T>::Array(const af::dim4 &dims, Node_ptr n)
    : info(getActiveDeviceId(), dims, 0, calcStrides(dims),
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data()
    , data_dims(dims)
    , node(move(n))
    , owner(true) {}

template<typename T>
Array<T>::Array(const Array<T> &parent, const dim4 &dims, const dim_t &offset_,
                const dim4 &strides)
    : info(parent.getDevId(), dims, offset_, strides,
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data(parent.getData())
    , data_dims(parent.getDataDims())
    , node()
    , owner(false) {}

template<typename T>
Array<T>::Array(const dim4 &dims, const dim4 &strides, dim_t offset_,
                T *const in_data, bool is_device)
    : info(getActiveDeviceId(), dims, offset_, strides,
           static_cast<af_dtype>(dtype_traits<T>::af_type))
    , data()
    , data_dims(dims)
    , node()
    , owner(true) {
    if (is_device) {
        auto *buffer = reinterpret_cast<MTL::Buffer *>(in_data);
        retainBuffer(buffer);
        data = shared_ptr<MTL::Buffer>(
            buffer, [](MTL::Buffer *value) {
                releaseBuffer(value);
            });
    } else {
        data = managedBuffer(memAlloc<T>(info.total()).release(),
                             getActiveDeviceId());
        // Ensure the memory being written to isnt used anywhere else.
        getQueue().sync();
        copy(in_data, in_data + info.total(), getHostPtr(false));
    }
}

template<typename T>
void checkAndMigrate(Array<T> &arr) {
    const int sourceDevice      = arr.getDevId();
    const int destinationDevice = getActiveDeviceId();
    if (sourceDevice == destinationDevice) { return; }

    if (setDevice(sourceDevice) < 0) {
        AF_ERROR("Array references an unavailable Metal device", AF_ERR_DEVICE);
    }

    MTL::Buffer *staging = nullptr;
    try {
        MTL::Buffer *source = arr.device();
        const size_t bytes =
            static_cast<size_t>(arr.elements()) * sizeof(T);
        const void *sourceContents = source ? source->contents() : nullptr;

        if (bytes && !sourceContents) {
            staging =
                getDevice().newBuffer(bytes, MTL::ResourceStorageModeShared);
            if (!staging || !copyBuffer(staging, 0, source, 0, bytes)) {
                if (staging) {
                    staging->release();
                    staging = nullptr;
                }
                setDevice(destinationDevice);
                AF_ERROR("Could not stage a Metal array for migration",
                         AF_ERR_RUNTIME);
            }
            getQueue().sync();
            sourceContents = staging->contents();
        }

        setDevice(destinationDevice);
        auto migrated = memAlloc<T>(arr.elements());
        if (bytes) {
            void *destinationContents = migrated->contents();
            if (!destinationContents || !sourceContents) {
                if (staging) {
                    staging->release();
                    staging = nullptr;
                }
                AF_ERROR("Metal array migration requires shared staging memory",
                         AF_ERR_RUNTIME);
            }
            std::memcpy(destinationContents, sourceContents, bytes);
        }
        if (staging) {
            staging->release();
            staging = nullptr;
        }

        arr.data = managedBuffer(migrated.release(), destinationDevice);
        arr.data_dims = arr.dims();
        arr.owner     = true;
        arr.setId(destinationDevice);
    } catch (...) {
        if (staging) { staging->release(); }
        setDevice(destinationDevice);
        throw;
    }
}

template<typename T>
void Array<T>::eval() {
    evalMultiple<T>({this});
}

template<typename T>
void Array<T>::eval() const {
    const_cast<Array<T> *>(this)->eval();
}

template<typename T>
MTL::Buffer *Array<T>::device() {
    if (!isOwner() || getOffset() || data.use_count() > 1) {
        *this = copyArray<T>(*this);
    }
    return this->getBuffer();
}

template<typename T>
void evalMultiple(vector<Array<T> *> array_ptrs) {
    vector<Array<T> *> outputs;
    vector<common::Node_ptr> nodes;
    vector<Param<T>> params;
    if (getQueue().is_worker()) {
        AF_ERROR("Array not evaluated", AF_ERR_INTERNAL);
    }

    // Check if all the arrays have the same dimension
    auto it = adjacent_find(begin(array_ptrs), end(array_ptrs),
                            [](const Array<T> *l, const Array<T> *r) {
                                return l->dims() != r->dims();
                            });

    // If they are not the same. eval individually
    if (it != end(array_ptrs)) {
        for (auto ptr : array_ptrs) { ptr->eval(); }
        return;
    }

    for (Array<T> *array : array_ptrs) {
        if (array->isReady()) { continue; }

        array->setId(getActiveDeviceId());
        array->data =
            managedBuffer(memAlloc<T>(array->elements()).release(),
                          getActiveDeviceId());

        outputs.push_back(array);
        params.emplace_back(array->getBuffer(), array->getOffset(),
                            array->dims(), array->strides());
        nodes.push_back(array->node);
    }

    if (params.empty()) return;

    if (supportsNativeJit(nodes, params.size())) {
        getQueue().enqueueNative(kernel::evalMultipleMetal<T>, params, nodes);
    } else {
        // Multiple outputs consume one Metal buffer slot each. If their
        // combined graph exceeds Metal's binding-table limit, retain native
        // execution by submitting each otherwise-valid expression separately.
        for (size_t i = 0; i < nodes.size(); ++i) {
            if (!supportsNativeJit({nodes[i]}, 1)) {
                AF_ERROR("Expression type is not supported by Metal",
                         AF_ERR_TYPE);
            }
            getQueue().enqueueNative(kernel::evalMultipleMetal<T>,
                                     vector<Param<T>>{params[i]},
                                     vector<common::Node_ptr>{nodes[i]});
        }
    }

    for (Array<T> *array : outputs) { array->node.reset(); }
}

template<typename T>
Node_ptr Array<T>::getNode() {
    if (node) { return node; }

    std::shared_ptr<BufferNode> out = bufferNodePtr<T>();
    unsigned bytes = this->getDataDims().elements() * sizeof(T);
    KParam param{{dims()[0], dims()[1], dims()[2], dims()[3]},
                 {strides()[0], strides()[1], strides()[2], strides()[3]},
                 getOffset() * static_cast<dim_t>(sizeof(T))};
    out->setData(param, data, bytes, isLinear());
    return out;
}

template<typename T>
Node_ptr Array<T>::getNode() const {
    return const_cast<Array<T> *>(this)->getNode();
}

template<typename T>
Array<T> createHostDataArray(const dim4 &dims, const T *const data) {
    verifyTypeSupport<T>();
    return Array<T>(dims, const_cast<T *>(data), false);
}

template<typename T>
Array<T> createDeviceDataArray(const dim4 &dims, void *data, bool copy) {
    verifyTypeSupport<T>();
    bool is_device = true;
    return Array<T>(dims, static_cast<T *>(data), is_device, copy);
}

template<typename T>
Array<T> createValueArray(const dim4 &dims, const T &value) {
    verifyTypeSupport<T>();
    return createNodeArray<T>(dims,
                              make_shared<common::ScalarNode<T>>(value));
}

template<typename T>
Array<T> createEmptyArray(const dim4 &dims) {
    verifyTypeSupport<T>();
    return Array<T>(dims);
}

template<typename T>
kJITHeuristics passesJitHeuristics(span<Node *> root_nodes) {
    if (!evalFlag()) { return kJITHeuristics::Pass; }
    for (Node *n : root_nodes) {
        if (n->getHeight() > static_cast<int>(getMaxJitSize())) {
            return kJITHeuristics::TreeHeight;
        }
    }

    Node_map_t nodeMap;
    vector<Node *> nodes;
    vector<common::Node_ids> ids;
    for (Node *root : root_nodes) { root->getNodesMap(nodeMap, nodes, ids); }

    // Metal exposes 31 buffer binding slots. A single-output JIT kernel needs
    // one slot for the output and one for its constants in addition to every
    // unique input buffer.
    const size_t inputBuffers =
        count_if(nodes.begin(), nodes.end(), [](const Node *node) {
            return node->getNodeType() == common::kNodeType::Buffer ||
                   node->getNodeType() == common::kNodeType::Shift;
        });
    if (inputBuffers + 2 > 31) {
        return kJITHeuristics::KernelParameterSize;
    }

    size_t bytes = 0;
    if (getMemoryPressure() >= getMemoryPressureThreshold()) {
        bytes = accumulate(nodes.begin(), nodes.end(), size_t{0},
                           [](size_t previous, const Node *node) {
                               return previous + node->getBytes();
                           });
    }
    if (jitTreeExceedsMemoryPressure(bytes)) {
        return kJITHeuristics::MemoryPressure;
    }

    return kJITHeuristics::Pass;
}

template<typename T>
Array<T> createNodeArray(const dim4 &dims, Node_ptr node) {
    verifyTypeSupport<T>();
    Array<T> out(dims, node);
    return out;
}

template<typename T>
Array<T> createSubArray(const Array<T> &parent, const vector<af_seq> &index,
                        bool copy) {
    parent.eval();

    dim4 dDims          = parent.getDataDims();
    dim4 parent_strides = parent.strides();

    if (parent.isLinear() == false) {
        const Array<T> parentCopy = copyArray(parent);
        return createSubArray(parentCopy, index, copy);
    }

    const dim4 &pDims = parent.dims();
    dim4 dims         = toDims(index, pDims);
    dim4 strides      = toStride(index, dDims);

    // Find total offsets after indexing
    dim4 offsets = toOffset(index, pDims);
    dim_t offset = parent.getOffset();
    for (int i = 0; i < 4; i++) { offset += offsets[i] * parent_strides[i]; }

    Array<T> out = Array<T>(parent, dims, offset, strides);

    if (!copy) { return out; }

    if (strides[0] != 1 || strides[1] < 0 || strides[2] < 0 || strides[3] < 0) {
        out = copyArray(out);
    }

    return out;
}

template<typename T>
void destroyArray(Array<T> *A) {
    delete A;
}

template<typename T>
void writeHostDataArray(Array<T> &arr, const T *const data,
                        const size_t bytes) {
    if (!arr.isOwner()) { arr = copyArray<T>(arr); }
    arr.eval();
    // Ensure the memory being written to isnt used anywhere else.
    getQueue().sync();
    memcpy(arr.getHostPtr(), data, bytes);
}

template<typename T>
void writeDeviceDataArray(Array<T> &arr, const void *const data,
                          const size_t bytes) {
    if (!arr.isOwner()) { arr = copyArray<T>(arr); }
    arr.eval();
    getQueue().sync();
    if (!copyBuffer(
            arr.getBuffer(), static_cast<size_t>(arr.getOffset()) * sizeof(T),
            static_cast<MTL::Buffer *>(const_cast<void *>(data)), 0, bytes)) {
        AF_ERROR("Could not copy Metal buffer", AF_ERR_RUNTIME);
    }
}

template<typename T>
void Array<T>::setDataDims(const dim4 &new_dims) {
    data_dims = new_dims;
    modDims(new_dims);
}

#define INSTANTIATE(T)                                                        \
    template Array<T> createHostDataArray<T>(const dim4 &dims,                \
                                             const T *const data);            \
    template Array<T> createDeviceDataArray<T>(const dim4 &dims, void *data,  \
                                               bool copy);                    \
    template Array<T> createValueArray<T>(const dim4 &dims, const T &value);  \
    template Array<T> createEmptyArray<T>(const dim4 &dims);                  \
    template Array<T> createSubArray<T>(                                      \
        const Array<T> &parent, const vector<af_seq> &index, bool copy);      \
    template void destroyArray<T>(Array<T> * A);                              \
    template Array<T> createNodeArray<T>(const dim4 &dims, Node_ptr node);    \
    template void Array<T>::eval();                                           \
    template void Array<T>::eval() const;                                     \
    template MTL::Buffer *Array<T>::device();                                 \
    template Array<T>::Array(const af::dim4 &dims, T *const in_data,          \
                             bool is_device, bool copy_device);               \
    template Array<T>::Array(const af::dim4 &dims, const af::dim4 &strides,   \
                             dim_t offset, T *const in_data, bool is_device); \
    template Node_ptr Array<T>::getNode();                                    \
    template Node_ptr Array<T>::getNode() const;                              \
    template void writeHostDataArray<T>(Array<T> & arr, const T *const data,  \
                                        const size_t bytes);                  \
    template void writeDeviceDataArray<T>(                                    \
        Array<T> & arr, const void *const data, const size_t bytes);          \
    template void evalMultiple<T>(vector<Array<T> *> arrays);                 \
    template kJITHeuristics passesJitHeuristics<T>(span<Node *> n);           \
    template void Array<T>::setDataDims(const dim4 &new_dims);                \
    template void checkAndMigrate<T>(Array<T> &arr);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(schar)
INSTANTIATE(uchar)
INSTANTIATE(char)
INSTANTIATE(intl)
INSTANTIATE(uintl)
INSTANTIATE(short)
INSTANTIATE(ushort)
INSTANTIATE(half)

}  // namespace metal
}  // namespace arrayfire
