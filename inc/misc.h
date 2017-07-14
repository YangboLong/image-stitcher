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
/// @File    misc.h
/// @Brief   Class with miscellaneous functions
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-04-05

#pragma once

#include <opencv2/opencv.hpp>

class Misc {
public:
    static cv::Mat rgb2gray(cv::Mat &img);
    static void display_image(cv::Mat &img, int i);
    static void print_mat(cv::Mat &img);
    static void print_point(std::vector<cv::Point> &p);
    static void swap_coordinates(std::vector<cv::Point> &pts);
    static cv::Mat stitch_images(cv::Mat &img1, cv::Mat &img2,
            std::array<std::array<double, 3>, 3> &aff_mat);
};
