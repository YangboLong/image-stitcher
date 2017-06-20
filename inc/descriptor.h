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
/// @File    descriptor.h
/// @Brief   Form a descriptor at each feature point and select the top ones.
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-04-16

#pragma once

#include "harris.h"

class Descriptor {
public:
    Descriptor(std::vector<CornerPoint> &pts, cv::Mat &img);
    std::vector<std::vector<float>> get_descriptors() { return descriptors_; }
    std::vector<float> get_mean() { return mean_; }
    std::vector<float> get_std() { return std_; }

private:
    int patch_size_;
    int half_patch_size_;
    std::vector<std::vector<float>> descriptors_; // nx169, n is number of corner points
    std::vector<float> mean_; // size n, mean values for descriptors
    std::vector<float> std_; //  size n, standard deviations for descriptors

    void vectorize(std::vector<CornerPoint> &pts, cv::Mat &img);
    void compute_mean_std();
};

