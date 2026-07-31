/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <convolve.hpp>
#include <copy.hpp>
#include <fast.hpp>
#include <kernel/orb.hpp>
#include <memory.hpp>
#include <platform.hpp>
#include <queue.hpp>
#include <resize.hpp>
#include <sort_index.hpp>
#include <af/dim4.hpp>

#include <cmath>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

using af::dim4;
using std::ceil;
using std::floor;
using std::min;
using std::move;
using std::pow;
using std::round;
using std::sqrt;
using std::vector;

namespace arrayfire {
namespace metal {

template<typename T, typename convAccT>
unsigned orb(Array<float>& x, Array<float>& y, Array<float>& score,
             Array<float>& ori, Array<float>& size, Array<uint>& desc,
             const Array<T>& image, const float fast_thr,
             const unsigned max_feat, const float scl_fctr,
             const unsigned levels, const bool blur_img) {
    image.eval();
    getQueue().sync();

    float patch_size = REF_PAT_SIZE;

    const dim4& idims   = image.dims();
    float min_side      = min(idims[0], idims[1]);
    unsigned max_levels = 0;
    float scl_sum       = 0.f;

    for (unsigned i = 0; i < levels; i++) {
        min_side /= scl_fctr;

        // Minimum image side for a descriptor to be computed
        if (min_side < patch_size || max_levels == levels) { break; }

        max_levels++;
        scl_sum += 1.f / pow(scl_fctr, static_cast<float>(i));
    }

    vector<buffer_ptr> h_x_pyr(max_levels);
    vector<buffer_ptr> h_y_pyr(max_levels);
    vector<buffer_ptr> h_score_pyr(max_levels);
    vector<buffer_ptr> h_ori_pyr(max_levels);
    vector<buffer_ptr> h_size_pyr(max_levels);
    vector<buffer_ptr> h_desc_pyr(max_levels);

    vector<unsigned> feat_pyr(max_levels);
    unsigned total_feat = 0;

    // Compute number of features to keep for each level
    vector<unsigned> lvl_best(max_levels);
    unsigned feat_sum = 0;
    for (unsigned i = 0; i < max_levels - 1; i++) {
        auto lvl_scl = pow(scl_fctr, static_cast<float>(i));
        lvl_best[i]  = ceil((static_cast<float>(max_feat) / scl_sum) / lvl_scl);
        feat_sum += lvl_best[i];
    }
    lvl_best[max_levels - 1] = max_feat - feat_sum;

    // Maintain a reference to previous level image
    Array<T> prev_img = createEmptyArray<T>(dim4());
    dim4 prev_ldims;

    dim4 gauss_dims(9);
    Array<T> gauss_filter = createEmptyArray<T>(dim4());

    auto ref_pattern = memAlloc<int>(REF_PAT_LENGTH);
    memcpy(bufferData<int>(ref_pattern.get()), kernel::ref_pat,
           REF_PAT_LENGTH * sizeof(int));

    for (unsigned i = 0; i < max_levels; i++) {
        dim4 ldims;
        const auto lvl_scl = pow(scl_fctr, static_cast<float>(i));
        Array<T> lvl_img   = createEmptyArray<T>(dim4());

        if (i == 0) {
            // First level is used in its original size
            lvl_img = image;
            ldims   = image.dims();

            prev_img   = image;
            prev_ldims = image.dims();
        } else {
            // Resize previous level image to current level dimensions
            ldims[0] = round(idims[0] / lvl_scl);
            ldims[1] = round(idims[1] / lvl_scl);

            lvl_img =
                resize<T>(prev_img, ldims[0], ldims[1], AF_INTERP_BILINEAR);

            prev_img   = lvl_img;
            prev_ldims = lvl_img.dims();
        }
        prev_img.eval();
        lvl_img.eval();
        getQueue().sync();

        Array<float> x_feat     = createEmptyArray<float>(dim4());
        Array<float> y_feat     = createEmptyArray<float>(dim4());
        Array<float> score_feat = createEmptyArray<float>(dim4());

        // Round feature size to nearest odd integer
        float size = 2.f * floor(static_cast<float>(patch_size) / 2.f) + 1.f;

        // Avoid keeping features that might be too wide and might not fit on
        // the image, sqrt(2.f) is the radius when angle is 45 degrees and
        // represents widest case possible
        unsigned edge = ceil(size * sqrt(2.f) / 2.f);

        unsigned lvl_feat = fast(x_feat, y_feat, score_feat, lvl_img, fast_thr,
                                 9, 1, 0.15f, edge);

        if (lvl_feat == 0) { continue; }

        auto h_x_harris     = memAlloc<float>(lvl_feat);
        auto h_y_harris     = memAlloc<float>(lvl_feat);
        Array<float> score_harris =
            createEmptyArray<float>(dim4(lvl_feat));

        // Calculate Harris responses
        // Good block_size >= 7 (must be an odd number)
        unsigned usable_feat = 0;
        if constexpr (std::is_same<T, float>::value) {
            kernel::orbHarrisMetal(
                {h_x_harris.get(), 0}, {h_y_harris.get(), 0},
                score_harris.bufferParam(), x_feat.bufferParam(),
                y_feat.bufferParam(), lvl_feat, &usable_feat, lvl_img, 7,
                0.04f, static_cast<unsigned>(patch_size));
        } else {
            AF_ERROR("Double input is not supported by the Metal ORB kernels",
                     AF_ERR_NOT_SUPPORTED);
        }

        if (usable_feat == 0) { continue; }

        // Sort features according to Harris responses
        score_harris.resetDims(dim4(usable_feat));
        Array<float> harris_sorted = createEmptyArray<float>(af::dim4());
        Array<unsigned> harris_idx = createEmptyArray<unsigned>(af::dim4());

        sort_index<float>(harris_sorted, harris_idx, score_harris, 0, false);
        getQueue().sync();

        usable_feat = min(usable_feat, lvl_best[i]);

        if (usable_feat == 0) { continue; }

        auto h_x_lvl     = memAlloc<float>(usable_feat);
        auto h_y_lvl     = memAlloc<float>(usable_feat);
        auto h_score_lvl = memAlloc<float>(usable_feat);

        // Keep only features with higher Harris responses
        kernel::orbKeepMetal(
            {h_x_lvl.get(), 0}, {h_y_lvl.get(), 0}, {h_score_lvl.get(), 0},
            {h_x_harris.get(), 0}, {h_y_harris.get(), 0},
            harris_sorted.bufferParam(), harris_idx.bufferParam(), usable_feat);

        auto h_ori_lvl  = memAlloc<float>(usable_feat);
        auto h_size_lvl = memAlloc<float>(usable_feat);

        // Compute orientation of features
        if constexpr (std::is_same<T, float>::value) {
            kernel::orbCentroidMetal(
                {h_x_lvl.get(), 0}, {h_y_lvl.get(), 0},
                {h_ori_lvl.get(), 0}, usable_feat, lvl_img, patch_size);
        } else {
            AF_ERROR("Double input is not supported by the Metal ORB kernels",
                     AF_ERR_NOT_SUPPORTED);
        }

        Array<T> lvl_filt = createEmptyArray<T>(dim4());

        if (blur_img) {
            // Calculate a separable Gaussian kernel, if one is not already
            // stored
            if (gauss_filter.isEmpty()) {
                gauss_filter = createEmptyArray<T>(gauss_dims);
                gaussian1D(gauss_filter.getHostPtr(), gauss_dims[0], 2.f);
            }

            // Filter level image with Gaussian kernel to reduce noise
            // sensitivity
            lvl_filt = convolve2<T, convAccT>(lvl_img, gauss_filter,
                                              gauss_filter, false);
        }
        lvl_filt.eval();
        getQueue().sync();

        // Compute ORB descriptors
        auto h_desc_lvl = memAlloc<unsigned>(usable_feat * 8);
        memset(bufferData<unsigned>(h_desc_lvl.get()), 0,
               usable_feat * 8 * sizeof(unsigned));
        if constexpr (std::is_same<T, float>::value) {
            const auto image_for_descriptor = blur_img ? lvl_filt : lvl_img;
            kernel::orbExtractMetal(
                {h_desc_lvl.get(), 0}, {h_x_lvl.get(), 0},
                {h_y_lvl.get(), 0}, {h_ori_lvl.get(), 0},
                {h_size_lvl.get(), 0}, {ref_pattern.get(), 0}, usable_feat,
                image_for_descriptor, static_cast<float>(lvl_scl),
                static_cast<unsigned>(patch_size));
        } else {
            AF_ERROR("Double input is not supported by the Metal ORB kernels",
                     AF_ERR_NOT_SUPPORTED);
        }

        // Store results to pyramids
        total_feat += usable_feat;
        feat_pyr[i]    = usable_feat;
        h_x_pyr[i]     = move(h_x_lvl);
        h_y_pyr[i]     = move(h_y_lvl);
        h_score_pyr[i] = move(h_score_lvl);
        h_ori_pyr[i]   = move(h_ori_lvl);
        h_size_pyr[i]  = move(h_size_lvl);
        h_desc_pyr[i]  = move(h_desc_lvl);
    }

    if (total_feat > 0) {
        // Allocate feature Arrays
        const af::dim4 total_feat_dims(total_feat);
        const af::dim4 desc_dims(8, total_feat);

        x     = createEmptyArray<float>(total_feat_dims);
        y     = createEmptyArray<float>(total_feat_dims);
        score = createEmptyArray<float>(total_feat_dims);
        ori   = createEmptyArray<float>(total_feat_dims);
        size  = createEmptyArray<float>(total_feat_dims);
        desc  = createEmptyArray<uint>(desc_dims);

        unsigned offset = 0;
        for (unsigned i = 0; i < max_levels; i++) {
            if (feat_pyr[i] == 0) { continue; }

            const size_t featureOffset = offset * sizeof(float);
            const size_t featureBytes = feat_pyr[i] * sizeof(float);
            const bool copied =
                copyBuffer(x.getBuffer(), featureOffset, h_x_pyr[i].get(), 0,
                           featureBytes) &&
                copyBuffer(y.getBuffer(), featureOffset, h_y_pyr[i].get(), 0,
                           featureBytes) &&
                copyBuffer(score.getBuffer(), featureOffset,
                           h_score_pyr[i].get(), 0, featureBytes) &&
                copyBuffer(ori.getBuffer(), featureOffset, h_ori_pyr[i].get(),
                           0, featureBytes) &&
                copyBuffer(size.getBuffer(), featureOffset,
                           h_size_pyr[i].get(), 0, featureBytes) &&
                copyBuffer(desc.getBuffer(), offset * 8 * sizeof(unsigned),
                           h_desc_pyr[i].get(), 0,
                           feat_pyr[i] * 8 * sizeof(unsigned));
            if (!copied) {
                AF_ERROR("Could not copy Metal ORB buffers", AF_ERR_RUNTIME);
            }
            offset += feat_pyr[i];
        }
    }

    return total_feat;
}

#define INSTANTIATE(T, convAccT)                                              \
    template unsigned orb<T, convAccT>(                                       \
        Array<float> & x, Array<float> & y, Array<float> & score,             \
        Array<float> & ori, Array<float> & size, Array<uint> & desc,          \
        const Array<T>& image, const float fast_thr, const unsigned max_feat, \
        const float scl_fctr, const unsigned levels, const bool blur_img);

INSTANTIATE(float, float)
INSTANTIATE(double, double)

}  // namespace metal
}  // namespace arrayfire
