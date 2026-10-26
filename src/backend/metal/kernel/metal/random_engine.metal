/*******************************************************
 * Copyright (c) 2026, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <metal_stdlib>
using namespace metal;

struct RandomParams {
    ulong elements;
    ulong seed;
    ulong counter;
    uint type;
};

struct MersenneParams {
    ulong elements;
    ulong elementsPerBlock;
    uint mask;
    uint blocks;
};

struct MersenneInitParams { ulong seed; };

kernel void random_mersenne_init(const device uint* table [[buffer(0)]],
                                 device uint* state [[buffer(1)]],
                                 constant MersenneInitParams& p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid >= 32) return;
    const uint stream = gid;
    const uint tableOffset = stream * 16;
    const uint stateOffset = stream * 351;
    const uint hiddenSeed = table[tableOffset + 4] ^
                           (table[tableOffset + 8] << 16);
    uint repeated = hiddenSeed;
    repeated += repeated >> 16;
    repeated += repeated >> 8;
    repeated &= 0xff;
    repeated |= repeated << 8;
    repeated |= repeated << 16;
    state[stateOffset] = uint(p.seed);
    state[stateOffset + 1] = hiddenSeed ^
               (1812433253u * (state[stateOffset] ^
                                (state[stateOffset] >> 30)) + 1u);
    for (uint i = 2; i < 351; ++i) {
        state[stateOffset + i] = repeated;
        state[stateOffset + i] ^=
            1812433253u * (state[stateOffset + i - 1] ^
                           (state[stateOffset + i - 1] >> 30)) + i;
    }
}

inline float randomFloat01(const uint value) {
    return fma(float(value), 2.3283064365386963e-10f,
               1.1641532182693481e-10f);
}

inline float randomFloatUniform(const uint value) {
    return 1.0f - randomFloat01(value);
}

inline float randomFloatUniform(thread const uint* values, const uint index) {
    return randomFloatUniform(values[index]);
}

inline half randomHalfUniform(thread const uint* values, const uint index) {
    const uint word = values[index >> 1];
    const float value = float((word >> (16u * (index & 1u))) & 0xffffu);
    return half(1.0f - fma(value, 1.0f / 65536.0f, 0.5f / 65536.0f));
}

inline char randomCharUniform(thread const uint* values, const uint index) {
    return char((values[index >> 2] >> (8u * (index & 3u))) & 1u);
}

inline char randomSCharUniform(thread const uint* values, const uint index) {
    return char((values[index >> 2] >> (8u * (index & 3u))) & 0xffu);
}

inline uchar randomUCharUniform(thread const uint* values, const uint index) {
    return uchar((values[index >> 2] >> (8u * (index & 3u))) & 0xffu);
}

inline short randomShortUniform(thread const uint* values, const uint index) {
    return short((values[index >> 1] >> (16u * (index & 1u))) & 0xffffu);
}

inline ushort randomUShortUniform(thread const uint* values,
                                  const uint index) {
    return ushort((values[index >> 1] >> (16u * (index & 1u))) & 0xffffu);
}

inline int randomIntUniform(thread const uint* values, const uint index) {
    return int(values[index]);
}

inline uint randomUIntUniform(thread const uint* values, const uint index) {
    return values[index];
}

inline long randomLongUniform(thread const uint* values, const uint index) {
    return long((ulong(values[index * 2]) << 32) | ulong(values[index * 2 + 1]));
}

inline ulong randomULongUniform(thread const uint* values, const uint index) {
    return (ulong(values[index * 2]) << 32) | ulong(values[index * 2 + 1]);
}

inline long randomMersenneLongUniform(thread const uint* values,
                                      const uint index) {
    return long((ulong(values[index * 2 + 1]) << 32) |
                ulong(values[index * 2]));
}

inline ulong randomMersenneULongUniform(thread const uint* values,
                                        const uint index) {
    return (ulong(values[index * 2 + 1]) << 32) | ulong(values[index * 2]);
}

inline void philoxRound(thread uint* key, thread uint* counter) {
    const ulong product0 = ulong(0xD2511F53u) * ulong(counter[0]);
    const ulong product1 = ulong(0xCD9E8D57u) * ulong(counter[2]);
    const uint hi0 = uint(product0 >> 32);
    const uint hi1 = uint(product1 >> 32);
    const uint lo0 = uint(product0);
    const uint lo1 = uint(product1);
    counter[0] = hi1 ^ counter[1] ^ key[0];
    counter[1] = lo1;
    counter[2] = hi0 ^ counter[3] ^ key[1];
    counter[3] = lo0;
}

inline void philox(thread uint* key, thread uint* counter) {
    for (uint round = 0; round < 10; ++round) {
        philoxRound(key, counter);
        if (round != 9) {
            key[0] += 0x9E3779B9u;
            key[1] += 0xBB67AE85u;
        }
    }
}

constant uint threefryRotations[8] = {13, 15, 26, 6, 17, 29, 16, 24};

inline uint rotateLeft(uint value, uint amount) {
    return (value << amount) | (value >> (32 - amount));
}

inline void threefry(thread uint* key, thread uint* counter) {
    uint ks[3] = {key[0], key[1], 0x1BD11BDAu ^ key[0] ^ key[1]};
    uint x0 = counter[0] + ks[0];
    uint x1 = counter[1] + ks[1];
    for (uint round = 0; round < 16; ++round) {
        x0 += x1;
        x1 = rotateLeft(x1, threefryRotations[round & 7]);
        x1 ^= x0;
        if ((round & 3) == 3) {
            const uint injection = (round + 1) / 4;
            x0 += ks[injection % 3];
            x1 += ks[(injection + 1) % 3] + injection;
        }
    }
    counter[0] = x0;
    counter[1] = x1;
}

#define DEFINE_RANDOM_UNIFORM(NAME, TYPE, PHILOX_RESET, THREEFRY_RESET, FN) \
kernel void NAME(device TYPE* output [[buffer(0)]],                        \
                 constant RandomParams& p [[buffer(1)]],                 \
                 uint gid [[thread_position_in_grid]]) {                  \
    if (ulong(gid) >= p.elements) return;                                  \
    const uint reset = p.type == 100 ? PHILOX_RESET : THREEFRY_RESET;       \
    const ulong block = ulong(gid) / ulong(reset);                          \
    const uint index = gid % reset;                                         \
    uint key[2] = {uint(p.seed), uint(p.seed >> 32)};                        \
    uint values[4] = {0, 0, 0, 0};                                          \
    if (p.type == 100) {                                                    \
        uint counter[4] = {uint(p.counter + block),                       \
                           uint((p.counter + block) >> 32), 0, 0};          \
        philox(key, counter);                                               \
        values[0] = counter[0]; values[1] = counter[1];                     \
        values[2] = counter[2]; values[3] = counter[3];                     \
    } else {                                                                \
        uint counter[2] = {uint(p.counter + block),                         \
                           uint((p.counter + block) >> 32)};                \
        threefry(key, counter);                                             \
        values[0] = counter[0]; values[1] = counter[1];                     \
    }                                                                       \
    output[gid] = FN(values, index);                                        \
}

DEFINE_RANDOM_UNIFORM(random_uniform_float, float, 4, 2,
                      randomFloatUniform)
DEFINE_RANDOM_UNIFORM(random_uniform_int, int, 4, 2, randomIntUniform)
DEFINE_RANDOM_UNIFORM(random_uniform_uint, uint, 4, 2, randomUIntUniform)
DEFINE_RANDOM_UNIFORM(random_uniform_long, long, 2, 1, randomLongUniform)
DEFINE_RANDOM_UNIFORM(random_uniform_ulong, ulong, 2, 1, randomULongUniform)
DEFINE_RANDOM_UNIFORM(random_uniform_char, char, 16, 8, randomCharUniform)
DEFINE_RANDOM_UNIFORM(random_uniform_schar, char, 16, 8, randomSCharUniform)
DEFINE_RANDOM_UNIFORM(random_uniform_uchar, uchar, 16, 8, randomUCharUniform)
DEFINE_RANDOM_UNIFORM(random_uniform_short, short, 8, 4, randomShortUniform)
DEFINE_RANDOM_UNIFORM(random_uniform_ushort, ushort, 8, 4,
                      randomUShortUniform)
DEFINE_RANDOM_UNIFORM(random_uniform_half, half, 8, 4, randomHalfUniform)

inline float randomFloatNegative11(const uint value) {
    return fma(float(value), 1.0f / 2147483648.0f,
               0.5f / 2147483648.0f);
}

inline float randomNormalFloat(thread const uint* values, const uint index) {
    const uint pair = index >> 1;
    const float r1 = randomFloatNegative11(values[pair * 2]);
    const float r2 = randomFloat01(values[pair * 2 + 1]);
    const float radius = sqrt(-2.0f * log(r2));
    const float theta = 3.14159265358979323846f * r1;
    return index & 1u ? radius * cos(theta) : radius * sin(theta);
}

inline float randomHalfNegative11(const uint word, const uint index) {
    const float value = float((word >> (16u * (index & 1u))) & 0xffffu);
    return fma(value, 1.0f / 32768.0f, 0.5f / 32768.0f);
}

inline half randomNormalHalf(thread const uint* values, const uint index) {
    const uint pair = index >> 1;
    const float r1 = randomHalfNegative11(values[pair], 0);
    const float r2 = float((values[pair] >> 16) & 0xffffu) / 65536.0f +
                     0.5f / 65536.0f;
    const float radius = sqrt(-2.0f * log(r2));
    const float theta = 3.14159265358979323846f * r1;
    return half(index & 1u ? radius * cos(theta) : radius * sin(theta));
}

#define DEFINE_RANDOM_NORMAL(NAME, TYPE, PHILOX_RESET, THREEFRY_RESET, FN)  \
kernel void NAME(device TYPE* output [[buffer(0)]],                        \
                 constant RandomParams& p [[buffer(1)]],                 \
                 uint gid [[thread_position_in_grid]]) {                  \
    if (ulong(gid) >= p.elements) return;                                  \
    const uint reset = p.type == 100 ? PHILOX_RESET : THREEFRY_RESET;       \
    const ulong block = ulong(gid) / ulong(reset);                          \
    const uint index = gid % reset;                                         \
    uint key[2] = {uint(p.seed), uint(p.seed >> 32)};                        \
    uint values[4] = {0, 0, 0, 0};                                          \
    if (p.type == 100) {                                                    \
        uint counter[4] = {uint(p.counter + block),                       \
                           uint((p.counter + block) >> 32), 0, 0};          \
        philox(key, counter);                                               \
        values[0] = counter[0]; values[1] = counter[1];                     \
        values[2] = counter[2]; values[3] = counter[3];                     \
    } else {                                                                \
        const ulong first = p.counter + block * 2;                          \
        uint counter[2] = {uint(first), uint(first >> 32)};                  \
        threefry(key, counter);                                             \
        values[0] = counter[0]; values[1] = counter[1];                     \
        counter[0] = uint(first + 1); counter[1] = uint((first + 1) >> 32);  \
        threefry(key, counter);                                               \
        values[2] = counter[0]; values[3] = counter[1];                     \
    }                                                                       \
    output[gid] = FN(values, index);                                        \
}

DEFINE_RANDOM_NORMAL(random_normal_float, float, 4, 4, randomNormalFloat)
DEFINE_RANDOM_NORMAL(random_normal_half, half, 8, 8, randomNormalHalf)

inline uint mersenneRecursion(threadgroup const uint* table, const uint mask,
                              const uint sh1, const uint sh2, const uint x1,
                              const uint x2, uint y) {
    uint x = (x1 & mask) ^ x2;
    x ^= x << sh1;
    y = x ^ (y >> sh2);
    return y ^ table[y & 0x0fu];
}

inline uint mersenneTemper(threadgroup const uint* table, const uint value,
                           uint t) {
    t ^= t >> 16;
    t ^= t >> 8;
    return value ^ table[t & 0x0fu];
}

#define DEFINE_MERSENNE(NAME, TYPE, FN)                                     \
kernel void NAME(device TYPE* output [[buffer(0)]],                        \
                 device uint* state [[buffer(1)]],                         \
                 const device uint* pos [[buffer(2)]],                     \
                 const device uint* sh1 [[buffer(3)]],                     \
                 const device uint* sh2 [[buffer(4)]],                     \
                 const device uint* recursionTable [[buffer(5)]],          \
                 const device uint* temperTable [[buffer(6)]],             \
                 constant MersenneParams& p [[buffer(7)]],                \
                 uint lane [[thread_index_in_threadgroup]],                \
                 uint block [[threadgroup_position_in_grid]]) {            \
    if (block >= p.blocks) return;                                         \
    threadgroup uint localState[768];                                       \
    threadgroup uint localRecursion[16];                                    \
    threadgroup uint localTemper[16];                                       \
    if (lane < 256)                                                          \
        localState[417 + lane] = state[block * 351 + lane];                 \
    if (lane < 95)                                                           \
        localState[673 + lane] = state[block * 351 + 256 + lane];           \
    if (lane < 16) {                                                        \
        localRecursion[lane] = recursionTable[block * 16 + lane];           \
        localTemper[lane] = temperTable[block * 16 + lane];                 \
    }                                                                       \
    threadgroup_barrier(mem_flags::mem_threadgroup);                        \
    const uint posValue = pos[block];                                       \
    const uint sh1Value = sh1[block];                                       \
    const uint sh2Value = sh2[block];                                       \
    const ulong start = ulong(block) * p.elementsPerBlock;                   \
    if (start >= p.elements) return;                                         \
    const ulong end = min(start + p.elementsPerBlock, p.elements);           \
    const uint valuesPerLane = 16u / uint(sizeof(TYPE));                    \
    const ulong elementsPerIteration = ulong(256u) * valuesPerLane;          \
    const ulong iterations = (end - start + elementsPerIteration - 1) /      \
                             elementsPerIteration;                          \
    uint words[4];                                                          \
    uint offsetX1 = 417u + lane;                                            \
    uint offsetX2 = 417u + lane + 1u;                                       \
    uint offsetY = 417u + lane + posValue;                                  \
    uint offsetT = 417u + lane + posValue - 1u;                             \
    uint offsetO = lane;                                                    \
    for (ulong iteration = 0; iteration < iterations; ++iteration) {        \
        for (uint word = 0; word < 4; ++word) {                             \
            const uint value = mersenneRecursion(                           \
                localRecursion, p.mask, sh1Value, sh2Value,                 \
                localState[offsetX1 % 768u],                                \
                localState[offsetX2 % 768u],                                \
                localState[offsetY % 768u]);                                \
            localState[offsetO % 768u] = value;                             \
            words[word] = mersenneTemper(                                   \
                localTemper, value, localState[offsetT % 768u]);             \
            offsetX1 = (offsetX1 + 256u) % 768u;                            \
            offsetX2 = (offsetX2 + 256u) % 768u;                            \
            offsetY = (offsetY + 256u) % 768u;                              \
            offsetT = (offsetT + 256u) % 768u;                              \
            offsetO = (offsetO + 256u) % 768u;                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                 \
        }                                                                   \
        const ulong iterationStart =                                        \
            start + iteration * elementsPerIteration + lane;                 \
        for (uint j = 0; j < valuesPerLane; ++j) {                           \
            const ulong outputIndex = iterationStart + ulong(j) * 256u;      \
            if (outputIndex < p.elements)                                    \
                output[outputIndex] = FN(words, j);                          \
        }                                                                   \
    }                                                                       \
    if (lane < 256)                                                         \
        state[block * 351 + lane] = localState[417 + lane];                 \
    if (lane < 95)                                                          \
        state[block * 351 + 256 + lane] = localState[673 + lane];           \
}

DEFINE_MERSENNE(random_mersenne_uniform_float, float,
                randomFloatUniform)
DEFINE_MERSENNE(random_mersenne_uniform_int, int, randomIntUniform)
DEFINE_MERSENNE(random_mersenne_uniform_uint, uint, randomUIntUniform)
DEFINE_MERSENNE(random_mersenne_uniform_long, long,
                randomMersenneLongUniform)
DEFINE_MERSENNE(random_mersenne_uniform_ulong, ulong,
                randomMersenneULongUniform)
DEFINE_MERSENNE(random_mersenne_uniform_char, char, randomCharUniform)
DEFINE_MERSENNE(random_mersenne_uniform_schar, char,
                randomSCharUniform)
DEFINE_MERSENNE(random_mersenne_uniform_uchar, uchar,
                randomUCharUniform)
DEFINE_MERSENNE(random_mersenne_uniform_short, short,
                randomShortUniform)
DEFINE_MERSENNE(random_mersenne_uniform_ushort, ushort,
                randomUShortUniform)
DEFINE_MERSENNE(random_mersenne_uniform_half, half,
                randomHalfUniform)

DEFINE_MERSENNE(random_mersenne_normal_float, float,
                randomNormalFloat)
DEFINE_MERSENNE(random_mersenne_normal_half, half, randomNormalHalf)
