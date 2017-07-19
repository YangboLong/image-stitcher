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
/// @File    harris.h
/// @Brief   Interface of the Harris corner detection algorithm
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-04-01

#pragma once

#include <opencv2/opencv.hpp>

struct CornerPoint {
    float response;
    cv::Point point;
};

struct Derivatives {
    cv::Mat ix2;
    cv::Mat iy2;
    cv::Mat ixy; // Ix*Iy
};

struct CompareResponse {
    bool operator() (CornerPoint const &left, CornerPoint const &right) {
        return left.response > right.response;
    }
};

class Harris {
public:
    Harris(cv::Mat &img);
    std::vector<CornerPoint> nonmax_suppression(float percentage);
    cv::Mat mark_in_image(cv::Mat &img, std::vector<CornerPoint> &pts,
                          int dim, cv::Vec3b col);

private:
    int win_size_; // gaussian window size
    float alpha_; // sensitivity factor within [0.04, 0.06]
    int patch_size_; // patch size for nonmaximum suppression
    cv::Mat harris_response_;

    cv::Mat gaussian_filter(cv::Mat &img);
    Derivatives compute_derivatives(cv::Mat &img);
    Derivatives second_moment_matrix(Derivatives &derivs);
    cv::Mat compute_harris_response(Derivatives &matrix);
};
