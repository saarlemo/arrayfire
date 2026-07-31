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
    uint mask;
    uint reserved;
};

struct MersenneInitParams { ulong seed; };

kernel void random_mersenne_init(const device uint* table [[buffer(0)]],
                                 device uint* state [[buffer(1)]],
                                 constant MersenneInitParams& p [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    const uint hiddenSeed = table[4] ^ (table[8] << 16);
    uint repeated = hiddenSeed;
    repeated += repeated >> 16;
    repeated += repeated >> 8;
    repeated &= 0xff;
    repeated |= repeated << 8;
    repeated |= repeated << 16;
    state[0] = uint(p.seed);
    state[1] = hiddenSeed ^
               (1812433253u * (state[0] ^ (state[0] >> 30)) + 1u);
    for (uint i = 2; i < 351; ++i) {
        state[i] = repeated;
        state[i] ^= 1812433253u * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
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

inline uint mersenneRecursion(const device uint* table, const uint mask,
                              const uint sh1, const uint sh2, const uint x1,
                              const uint x2, uint y) {
    uint x = (x1 & mask) ^ x2;
    x ^= x << sh1;
    y = x ^ (y >> sh2);
    return y ^ table[y & 0x0fu];
}

inline uint mersenneTemper(const device uint* table, const uint value,
                           uint t) {
    t ^= t >> 16;
    t ^= t >> 8;
    return value ^ table[t & 0x0fu];
}

inline void mersenneWords(thread uint* out, thread uint* state,
                          const uint iteration, const uint pos,
                          const uint sh1, const uint sh2, const uint mask,
                          const device uint* recursionTable,
                          const device uint* temperTable) {
    int index = int(iteration % 768u);
    int offsetX1 = (768 - 351 + index) % 768;
    int offsetX2 = (768 - 351 + index + 1) % 768;
    int offsetY = (768 - 351 + index + int(pos)) % 768;
    int offsetT = (768 - 351 + index + int(pos) - 1) % 768;
    for (uint i = 0; i < 4; ++i) {
        state[index] = mersenneRecursion(recursionTable, mask, sh1, sh2,
                                         state[offsetX1], state[offsetX2],
                                         state[offsetY]);
        out[i] = mersenneTemper(temperTable, state[index], state[offsetT]);
        offsetX1 = (offsetX1 + 1) % 768;
        offsetX2 = (offsetX2 + 1) % 768;
        offsetY = (offsetY + 1) % 768;
        offsetT = (offsetT + 1) % 768;
        index = (index + 1) % 768;
    }
}

#define DEFINE_MERSENNE_UNIFORM(NAME, TYPE, RESET, FN)                     \
kernel void NAME(device TYPE* output [[buffer(0)]],                        \
                 device uint* state [[buffer(1)]],                        \
                 const device uint* pos [[buffer(2)]],                     \
                 const device uint* sh1 [[buffer(3)]],                     \
                 const device uint* sh2 [[buffer(4)]],                     \
                 const device uint* recursionTable [[buffer(5)]],          \
                 const device uint* temperTable [[buffer(6)]],              \
                 constant MersenneParams& p [[buffer(7)]],                  \
                 uint gid [[thread_position_in_grid]]) {                  \
    if (gid != 0) return;                                                   \
    uint localState[768];                                                   \
    for (uint i = 0; i < 351; ++i) localState[417 + i] = state[i];          \
    const uint posValue = pos[0];                                           \
    const uint sh1Value = sh1[0];                                           \
    const uint sh2Value = sh2[0];                                           \
    for (uint base = 0; base < p.elements; base += RESET) {                \
        uint words[4];                                                      \
        mersenneWords(words, localState, base, posValue, sh1Value,           \
                      sh2Value, p.mask, recursionTable, temperTable);       \
        for (uint j = 0; j < RESET && base + j < p.elements; ++j)            \
            output[base + j] = FN(words, j);                                \
    }                                                                       \
    for (uint i = 0; i < 351; ++i) state[i] = localState[417 + i];           \
}

DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_float, float, 4,
                        randomFloatUniform)
DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_int, int, 4, randomIntUniform)
DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_uint, uint, 4,
                        randomUIntUniform)
DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_long, long, 2,
                        randomLongUniform)
DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_ulong, ulong, 2,
                        randomULongUniform)
DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_char, char, 16,
                        randomCharUniform)
DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_schar, char, 16,
                        randomSCharUniform)
DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_uchar, uchar, 16,
                        randomUCharUniform)
DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_short, short, 8,
                        randomShortUniform)
DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_ushort, ushort, 8,
                        randomUShortUniform)
DEFINE_MERSENNE_UNIFORM(random_mersenne_uniform_half, half, 8,
                        randomHalfUniform)

#define DEFINE_MERSENNE_NORMAL(NAME, TYPE, RESET, FN)                       \
kernel void NAME(device TYPE* output [[buffer(0)]],                        \
                 device uint* state [[buffer(1)]],                        \
                 const device uint* pos [[buffer(2)]],                     \
                 const device uint* sh1 [[buffer(3)]],                     \
                 const device uint* sh2 [[buffer(4)]],                     \
                 const device uint* recursionTable [[buffer(5)]],          \
                 const device uint* temperTable [[buffer(6)]],              \
                 constant MersenneParams& p [[buffer(7)]],                  \
                 uint gid [[thread_position_in_grid]]) {                  \
    if (gid != 0) return;                                                   \
    uint localState[768];                                                   \
    for (uint i = 0; i < 351; ++i) localState[417 + i] = state[i];          \
    const uint posValue = pos[0];                                           \
    const uint sh1Value = sh1[0];                                           \
    const uint sh2Value = sh2[0];                                           \
    for (uint base = 0; base < p.elements; base += RESET) {                \
        uint words[4];                                                      \
        mersenneWords(words, localState, base, posValue, sh1Value,           \
                      sh2Value, p.mask, recursionTable, temperTable);       \
        for (uint j = 0; j < RESET && base + j < p.elements; ++j)            \
            output[base + j] = FN(words, j);                                \
    }                                                                       \
    for (uint i = 0; i < 351; ++i) state[i] = localState[417 + i];           \
}

DEFINE_MERSENNE_NORMAL(random_mersenne_normal_float, float, 4,
                       randomNormalFloat)
DEFINE_MERSENNE_NORMAL(random_mersenne_normal_half, half, 8,
                       randomNormalHalf)
