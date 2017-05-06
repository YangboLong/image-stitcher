/// Copyright (C)
///
/// This program is free software; you can redistribute it and/or
/// modify it under the terms of the GNU General Public License
/// as published by the Free Software Foundation; either version 2
/// of the License, or (at your option) any later version.
///
/// This program is distributed in the hope that it will be useful,
/// but WITHOUT ANY WARRANTY; without even the implied warranty of
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// GNU General Public License for more details.
///
/// You should have received a copy of the GNU General Public License
/// along with this program; if not, write to the Free Software
/// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
///
/// @File    descriptor.cpp
/// @Brief   Form a descriptor at each feature point and select the top ones.
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-04-16

#include "descriptor.h"
#include <numeric>

/// --------------------------------------------------------------------------
/// @Brief   Form descriptors around feature points and compute the mean value
///          and standard deviation for them.
///
/// @Param   pts, corner points
/// @Param   img, grayscale image
/// --------------------------------------------------------------------------
Descriptor::Descriptor(std::vector<CornerPoint> &pts, cv::Mat &img) {
    // initialize descriptor patch parameters
    patch_size_ = 13;
    half_patch_size_ = patch_size_ / 2;

    // form a descriptor by vectorizing an image patch
    vectorize(pts, img);

    // compute mean value and standard deviation for each descriptor
    compute_mean_std();
}

/// --------------------------------------------------------------------------
/// @Brief   Extract fixed size image patches (13x13) around every feature
///          point, form a descriptor simply by vectorizing the image pixel
///          value in raster scan order.
///
/// @Param   pts, coordinates of feature points
/// @Param   img, grayscale image
/// --------------------------------------------------------------------------
void Descriptor::vectorize(std::vector<CornerPoint> &pts, cv::Mat &img) {
    int rows = pts.size();
    for (int i = 0; i < rows; i++) {
        int row = pts[i].point.x, col = pts[i].point.y;
        std::vector<float> tmp;
        for (int r = row - half_patch_size_; r <= row + half_patch_size_; r++) {
            for (int c = col - half_patch_size_; c <= col + half_patch_size_; c++) {
                if (r < 0 || r >= img.rows || c < 0 || c >= img.cols) {
                    tmp.push_back(0);
                } else {
                    tmp.push_back(img.at<float>(r, c));
                }
            }
        }
        // each row represents the verctorized pixels of a patch
        descriptors_.push_back(tmp);
    }
}

/// --------------------------------------------------------------------------
/// @Brief   Compute mean value and standard deviation for each descriptor.
/// --------------------------------------------------------------------------
void Descriptor::compute_mean_std() {
    for (size_t i = 0; i < descriptors_.size(); i++) {
        // compute mean value for one descriptor
        std::vector<float> v = descriptors_[i];
        float sum = std::accumulate(v.begin(), v.end(), 0.0);
        float mean = sum / v.size();

        // compute standard deviation for one descriptor
        std::vector<float> diff(v.size());
        std::transform(v.begin(), v.end(), diff.begin(),
                [mean](float x) { return x - mean; });
        float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        float stdev = std::sqrt(sq_sum / v.size());

        // save n values for all descriptors
        mean_.push_back(mean);
        std_.push_back(stdev);
    }
}

