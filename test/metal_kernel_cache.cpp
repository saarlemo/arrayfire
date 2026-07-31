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

#include <dirent.h>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

namespace {

class TemporaryCacheDirectory {
   public:
    TemporaryCacheDirectory() {
        char path[]  = "/private/tmp/arrayfire-metal-cache.XXXXXX";
        char* result = mkdtemp(path);
        if (result) { path_ = result; }

        const char* previous = std::getenv("AF_JIT_KERNEL_CACHE_DIRECTORY");
        if (previous) {
            hadPrevious_ = true;
            previous_    = previous;
        }
        if (!path_.empty()) {
            setenv("AF_JIT_KERNEL_CACHE_DIRECTORY", path_.c_str(), 1);
        }
    }

    ~TemporaryCacheDirectory() {
        if (hadPrevious_) {
            setenv("AF_JIT_KERNEL_CACHE_DIRECTORY", previous_.c_str(), 1);
        } else {
            unsetenv("AF_JIT_KERNEL_CACHE_DIRECTORY");
        }

        DIR* directory = opendir(path_.c_str());
        if (directory) {
            while (dirent* entry = readdir(directory)) {
                const std::string name(entry->d_name);
                if (name == "." || name == "..") { continue; }
                unlink((path_ + "/" + name).c_str());
            }
            closedir(directory);
        }
        if (!path_.empty()) { rmdir(path_.c_str()); }
    }

    bool valid() const { return !path_.empty(); }

    std::vector<std::string> archives() const {
        std::vector<std::string> result;
        DIR* directory = opendir(path_.c_str());
        if (!directory) { return result; }
        while (dirent* entry = readdir(directory)) {
            const std::string name(entry->d_name);
            const std::string suffix = ".binary.metallib";
            if (name.size() >= suffix.size() &&
                name.compare(name.size() - suffix.size(), suffix.size(),
                             suffix) == 0) {
                result.push_back(name);
            }
        }
        closedir(directory);
        return result;
    }

   private:
    std::string path_;
    std::string previous_;
    bool hadPrevious_ = false;
};

}  // namespace

TEST(MetalKernelCache, PersistsStaticAndJitPipelines) {
    TemporaryCacheDirectory cache;
    ASSERT_TRUE(cache.valid());

    af::array sequence = af::range(af::dim4(64), 0, f32);
    std::vector<float> sequenceHost(64);
    sequence.host(sequenceHost.data());
    ASSERT_EQ(cache.archives().size(), 1u);

    af::array output = af::sin(sequence * 2.0f + 1.0f);
    std::vector<float> outputHost(64);
    output.host(outputHost.data());
    ASSERT_EQ(cache.archives().size(), 2u);
}
