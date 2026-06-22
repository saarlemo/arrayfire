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

inline float randomFloat(uint value) {
    return 1.0f - fma(float(value), 2.3283064365386963e-10f,
                      1.1641532182693481e-10f);
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

kernel void random_uniform_float(device float* output [[buffer(0)]],
    constant RandomParams& p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.elements) return;
    uint key[2] = {uint(p.seed), uint(p.seed >> 32)};
    if (p.type == 100) {
        const ulong iteration = ulong(gid / 1024) * 1024;
        const uint within = gid % 1024;
        const uint valueIndex = within / 256;
        const ulong firstWrite = iteration + ulong(within % 256);
        const uint lowCounter = uint(p.counter);
        const uint highCounter = uint(p.counter >> 32);
        uint counter[4] = {lowCounter + uint(firstWrite), 0, 0, 0};
        counter[1] = highCounter + uint(counter[0] < lowCounter);
        counter[2] = uint(counter[1] < highCounter);
        philox(key, counter);
        output[gid] = randomFloat(counter[valueIndex]);
    } else {
        const ulong blockCounter = p.counter + ulong(gid / 2);
        uint counter[2] = {uint(blockCounter), uint(blockCounter >> 32)};
        threefry(key, counter);
        output[gid] = randomFloat(counter[gid & 1]);
    }
}
