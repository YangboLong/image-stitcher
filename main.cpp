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
/// @File    main.cpp
/// @Brief   Entry point of the whole project
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-04-01

#include <stdio.h>
#include "harris.h"
#include "misc.h"

int main(int argc, char **argv) {
    cv::Mat img_src, gray;

    if (argc <= 1) {
        std::cout << "No image provided. Usage: ./harris [path/to/image]" << std::endl;
        return 0;
    } else {
        img_src = cv::imread(argv[1]);
    }

    // convert color image to grayscale
    cvtColor(img_src, gray, CV_BGR2GRAY);
    gray.convertTo(gray, CV_32F);
    // gray = Misc::rgb2gray(img_src);

    // run harris algorithm
    Harris harris(gray);

    // local nonmaximum suppression
    float percentage = 0.001;
    std::vector<CornerPoint> pts = harris.nonmax_suppression(percentage);

    // mark top corner points in image
    int marker_size = 5;
    cv::Mat img_dst = harris.mark_in_image(img_src, pts, marker_size);
    // display result
    Misc::display_image(img_dst);

    return 0;
}
