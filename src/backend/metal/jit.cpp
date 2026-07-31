/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Metal.hpp>

#include <common/TemplateArg.hpp>
#include <common/jit/ModdimNode.hpp>
#include <common/jit/NodeIterator.hpp>
#include <common/kernel_cache.hpp>
#include <err_metal.hpp>
#include <jit.hpp>
#include <jit/BufferNode.hpp>
#include <jit/ShiftNode.hpp>
#include <metal_kernel_headers/jit.hpp>
#include <platform.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace arrayfire {
namespace metal {
namespace {

using common::Node;
using common::Node_ids;
using common::Node_map_t;
using common::Node_ptr;

constexpr size_t kMetalMaxBufferArguments = 31;
constexpr size_t kSetBytesLimit           = 4096;

const char *metalTypeName(af::dtype type) {
    switch (type) {
        case f32: return "float";
        case c32: return "float2";
        case s32: return "int";
        case u32: return "uint";
        case s64: return "long";
        case u64: return "ulong";
        case s16: return "short";
        case u16: return "ushort";
        case b8:
        case s8: return "char";
        case u8: return "uchar";
        case f16: return "half";
        case f64:
        case c64: break;
    }
    return nullptr;
}

bool gatherNodes(const std::vector<Node_ptr> &outputs, Node_map_t &nodeMap,
                 std::vector<Node *> &nodes, std::vector<Node_ids> &ids,
                 std::vector<int> &outputIds) {
    for (const Node_ptr &output : outputs) {
        outputIds.push_back(output->getNodesMap(nodeMap, nodes, ids));
    }
    for (const Node *node : nodes)
        if (!metalTypeName(node->getType())) return false;
    return true;
}

std::vector<Node_ptr> cloneForModdims(const std::vector<Node *> &nodes,
                                      const std::vector<Node_ids> &ids) {
    std::vector<Node_ptr> clones;
    clones.reserve(nodes.size());
    for (Node *node : nodes) { clones.emplace_back(node->clone()); }

    for (const Node_ids &nodeIds : ids) {
        auto &children = clones[nodeIds.id]->m_children;
        for (int child = 0;
             child < Node::kMaxChildren && children[child] != nullptr;
             ++child) {
            children[child] = clones[nodeIds.child_ids[child]];
        }
    }

    for (Node_ptr &node : clones) {
        if (node->getOp() != af_moddims_t) continue;
        const auto *moddim = static_cast<common::ModdimNode *>(node.get());
        for (common::NodeIterator<> it(node.get()), end; it != end; ++it) {
            if (it->isBuffer()) { it->modDims(moddim->m_new_shape); }
        }
    }
    return clones;
}

size_t aligned16(size_t value) { return (value + 15) & ~size_t(15); }

void appendConstant(std::vector<std::uint8_t> &storage, const void *value,
                    size_t bytes) {
    const size_t offset = aligned16(storage.size());
    storage.resize(offset + bytes);
    std::memcpy(storage.data() + offset, value, bytes);
    storage.resize(aligned16(storage.size()), 0);
}

struct BufferBinding {
    int index;
    MTL::Buffer *buffer;
    size_t offset;
};

}  // namespace

bool supportsNativeJit(const std::vector<Node_ptr> &outputNodes,
                       size_t outputCount) {
    Node_map_t nodeMap;
    std::vector<Node *> nodes;
    std::vector<Node_ids> ids;
    std::vector<int> outputIds;
    if (!gatherNodes(outputNodes, nodeMap, nodes, ids, outputIds)) return false;

    const size_t inputCount = std::count_if(
        nodes.begin(), nodes.end(), [](const Node *node) {
            return node->getNodeType() == common::kNodeType::Buffer ||
                   node->getNodeType() == common::kNodeType::Shift;
        });
    return inputCount + outputCount + 1 <= kMetalMaxBufferArguments;
}

void evalNodesMetal(std::vector<JitOutput> outputs,
                    std::vector<Node_ptr> outputNodes) {
    if (outputs.empty()) return;

    const af::dim4 outputDims = outputs.front().dims;
    const dim_t elements      = outputDims.elements();
    if (elements == 0) return;

    Node_map_t nodeMap;
    std::vector<Node *> nodes;
    std::vector<Node_ids> ids;
    std::vector<int> outputIds;
    if (!gatherNodes(outputNodes, nodeMap, nodes, ids, outputIds)) {
        AF_ERROR("Unsupported type in Metal JIT", AF_ERR_TYPE);
    }

    std::vector<Node_ptr> clones;
    const bool hasModdims =
        std::any_of(nodes.begin(), nodes.end(), [](const Node *node) {
            return node->getOp() == af_moddims_t;
        });
    if (hasModdims) {
        clones = cloneForModdims(nodes, ids);
        for (size_t i = 0; i < nodes.size(); ++i) nodes[i] = clones[i].get();
    }

    std::vector<BufferBinding> inputBindings;
    std::vector<std::uint8_t> constants;
    std::unordered_map<int, int> bufferIndices;
    int logicalArgument = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const int nodeId = ids[i].id;
        logicalArgument = nodes[i]->setArgs(
            logicalArgument, false,
            [&inputBindings, &constants, &bufferIndices,
             nodeId](int id, const void *value, size_t bytes, bool isBuffer) {
                UNUSED(id);
                if (isBuffer) {
                    auto *buffer = *static_cast<MTL::Buffer *const *>(value);
                    const int bufferIndex =
                        static_cast<int>(inputBindings.size());
                    bufferIndices[nodeId] = bufferIndex;
                    inputBindings.push_back({bufferIndex, buffer, bytes});
                } else {
                    appendConstant(constants, value, bytes);
                }
            });
    }

    std::stringstream constantFields;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const Node *node = nodes[i];
        const int id     = ids[i].id;
        if (node->getNodeType() == common::kNodeType::Buffer ||
            node->getNodeType() == common::kNodeType::Shift) {
            constantFields << "alignas(16) long dims" << id << "[4];\n"
                           << "long strides" << id << "[4];\n";
            if (node->getNodeType() == common::kNodeType::Shift) {
                for (int dim = 0; dim < 4; ++dim) {
                    constantFields << "alignas(16) int shift" << id << "_"
                                   << dim << ";\n";
                }
            }
        } else if (node->isScalar()) {
            constantFields << "alignas(16) " << metalTypeName(node->getType())
                           << " scalar" << id << ";\n";
        }
    }

    std::int64_t hostOutputDims[4];
    for (int dim = 0; dim < 4; ++dim)
        hostOutputDims[dim] = static_cast<std::int64_t>(outputDims[dim]);
    appendConstant(constants, hostOutputDims, sizeof(hostOutputDims));
    constantFields << "alignas(16) long outputDims[4];\n";

    for (size_t output = 0; output < outputs.size(); ++output) {
        std::int64_t hostOutputStrides[4];
        for (int dim = 0; dim < 4; ++dim) {
            hostOutputStrides[dim] =
                static_cast<std::int64_t>(outputs[output].strides[dim]);
        }
        appendConstant(constants, hostOutputStrides,
                       sizeof(hostOutputStrides));
        constantFields << "alignas(16) long outputStrides" << output
                       << "[4];\n";
    }

    std::stringstream parameters;
    std::stringstream offsets;
    std::stringstream functions;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const int id = ids[i].id;
        if (nodes[i]->getNodeType() == common::kNodeType::Buffer ||
            nodes[i]->getNodeType() == common::kNodeType::Shift) {
            parameters << "device const "
                       << metalTypeName(nodes[i]->getType()) << "* in" << id
                       << " [[buffer(" << bufferIndices.at(id) << ")]],\n";
        }
        nodes[i]->genOffsets(offsets, ids[i].id, false);
        if (nodes[i]->getOp() == af_moddims_t) {
            functions << metalTypeName(nodes[i]->getType()) << " val"
                      << ids[i].id << " = val" << ids[i].child_ids[0]
                      << ";\n";
        } else if (nodes[i]->isScalar()) {
            functions << metalTypeName(nodes[i]->getType()) << " val"
                      << ids[i].id << " = constants.scalar" << ids[i].id
                      << ";\n";
        } else {
            nodes[i]->genFuncs(functions, ids[i]);
        }
    }

    for (size_t output = 0; output < outputs.size(); ++output) {
        parameters << "device " << metalTypeName(outputs[output].type)
                   << "* out" << output << " [[buffer("
                   << inputBindings.size() + output << ")]],\n";
    }
    const int constantsIndex =
        static_cast<int>(inputBindings.size() + outputs.size());
    parameters << "constant JitConstants& constants [[buffer("
               << constantsIndex << ")]],\n"
               << "uint gid [[thread_position_in_grid]])";

    std::stringstream source;
    source.write(reinterpret_cast<const char *>(kernel::jit_metal_src.ptr),
                 kernel::jit_metal_src.length - 1);
    source << "\nstruct alignas(16) JitConstants {\n"
           << constantFields.str() << "};\n"
           << "kernel void af_metal_jit(\n"
           << parameters.str() << " {\n"
           << "const long total = constants.outputDims[0] * "
              "constants.outputDims[1] * constants.outputDims[2] * "
              "constants.outputDims[3];\n"
           << "if (long(gid) >= total) return;\n"
           << "long coordinate = long(gid);\n"
           << "const long x = coordinate % constants.outputDims[0];\n"
           << "coordinate /= constants.outputDims[0];\n"
           << "const long y = coordinate % constants.outputDims[1];\n"
           << "coordinate /= constants.outputDims[1];\n"
           << "const long z = coordinate % constants.outputDims[2];\n"
           << "const long w = coordinate / constants.outputDims[2];\n"
           << offsets.str() << functions.str();

    for (size_t output = 0; output < outputs.size(); ++output) {
        source << "const long outputOffset" << output << " = "
               << "x * constants.outputStrides" << output << "[0] + "
               << "y * constants.outputStrides" << output << "[1] + "
               << "z * constants.outputStrides" << output << "[2] + "
               << "w * constants.outputStrides" << output << "[3];\n"
               << "out" << output << "[outputOffset" << output << "] = val"
               << outputIds[output] << ";\n";
    }
    source << "}\n";

    const std::string sourceText = source.str();
    const common::Source kernelSource{sourceText.data(), sourceText.size(), 0};
    MTL::ComputePipelineState *pipeline =
        common::getKernel("af_metal_jit", {{kernelSource}}, TemplateArgs(), {},
                          true)
            .get();
    auto commandBuffer =
        NS::RetainPtr(getCommandQueue().commandBuffer());
    auto encoder = commandBuffer
                       ? NS::RetainPtr(commandBuffer->computeCommandEncoder())
                       : nullptr;
    if (!commandBuffer || !encoder) {
        AF_ERROR("Could not create a Metal JIT command encoder",
                 AF_ERR_RUNTIME);
    }

    encoder->setComputePipelineState(pipeline);
    for (const BufferBinding &binding : inputBindings) {
        if (!binding.buffer)
            AF_ERROR("Metal JIT received a null input buffer", AF_ERR_NO_MEM);
        encoder->setBuffer(binding.buffer, binding.offset, binding.index);
    }
    for (size_t output = 0; output < outputs.size(); ++output) {
        if (!outputs[output].storage.buffer)
            AF_ERROR("Metal JIT received a null output buffer", AF_ERR_NO_MEM);
        encoder->setBuffer(outputs[output].storage.buffer,
                           outputs[output].storage.offset,
                           inputBindings.size() + output);
    }

    NS::SharedPtr<MTL::Buffer> constantsBuffer;
    if (constants.size() <= kSetBytesLimit) {
        encoder->setBytes(constants.data(), constants.size(), constantsIndex);
    } else {
        constantsBuffer = NS::TransferPtr(getDevice().newBuffer(
            constants.data(), constants.size(), MTL::ResourceStorageModeShared));
        if (!constantsBuffer)
            AF_ERROR("Could not allocate Metal JIT constants", AF_ERR_NO_MEM);
        encoder->setBuffer(constantsBuffer.get(), 0, constantsIndex);
    }

    const NS::UInteger width =
        std::min<NS::UInteger>(256,
                               pipeline->maxTotalThreadsPerThreadgroup());
    encoder->dispatchThreads(MTL::Size(static_cast<NS::UInteger>(elements), 1,
                                       1),
                             MTL::Size(width, 1, 1));
    encoder->endEncoding();
    submitCommandBuffer(commandBuffer.get());
}

}  // namespace metal
}  // namespace arrayfire
