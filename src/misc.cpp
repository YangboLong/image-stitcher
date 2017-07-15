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
/// @File    misc.cpp
/// @Brief
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-04-05

#include "misc.h"
#include <string>

cv::Mat Misc::rgb2gray(cv::Mat &img) {
    cv::Mat gray(img.rows, img.cols, CV_32F);
    for (int c = 0; c < img.cols; c++) {
        for (int r = 0; r < img.rows; r++) {
            gray.at<float>(r, c) =
                0.114 * img.at<cv::Vec3b>(r, c)[0] +
                0.587 * img.at<cv::Vec3b>(r, c)[1] +
                0.299 * img.at<cv::Vec3b>(r, c)[2];
        }
    }

    return gray;
}

void Misc::display_image(cv::Mat &img, int i) {
    std::string str = "image" + std::to_string(i);
    cv::namedWindow(str, CV_WINDOW_AUTOSIZE);
    cv::imshow(str, img);
}

void Misc::print_mat(cv::Mat &img) {
    int row_count = 0, elem_count = 0;
    for (int r = 0; r < img.rows; r++) {
        row_count++;
        std::cout << "row " << row_count << ": ";
        for (int c = 0; c < img.cols; c++) {
            std::cout << img.at<float>(r, c) << " ";
            elem_count++;
        }
        std::cout << std::endl;
    }

    std::cout << row_count << " rows " << elem_count / row_count
        << " cols " << elem_count << " elements" << std::endl;
}

void Misc::print_point(std::vector<cv::Point> &vp) {
    for (size_t i = 0; i < vp.size(); i++) {
        cv::Point p = vp[i];
        std::cout << p.x << ", " << p.y << "; ";
    }
    std::cout << std::endl;
}

void Misc::swap_coordinates(std::vector<cv::Point> &pts) {
    for (auto &p : pts) {
        auto tmp = p.x;
        p.x = p.y;
        p.y = tmp;
    }
}

cv::Mat Misc::stitch_images(cv::Mat &img1, cv::Mat &img2,
        std::array<std::array<double, 3>, 3> &aff_mat) {
    // construct an empty pano image to hold the stitching result
    int height = std::max(img1.rows, img2.rows), width = img1.cols + img2.cols;
    cv::Mat pano(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    // affine matrix for passing in warpAffine
    cv::Mat warp_mat(2, 3, CV_64FC1);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            warp_mat.at<double>(i, j) = aff_mat[i][j];
        }
    }

    // fill src1 partially on the top left cornor with img1
    cv::Mat src1 = cv::Mat::zeros(height, width, img1.type());
    img1.copyTo(src1.rowRange(0, img1.rows).colRange(0, img1.cols));
    // prepare src2 for img2's affine transformation
    cv::Mat src2 = cv::Mat::zeros(height, width, img2.type());

    // apply the affine transform to img2
    cv::warpAffine(img2, src2, warp_mat, src2.size());

    // blend two images
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int ch = 0; ch < 3; ch++) {
                if (src1.at<cv::Vec3b>(i, j)[ch] == 0) {
                    pano.at<cv::Vec3b>(i, j)[ch] = src2.at<cv::Vec3b>(i, j)[ch];
                } else if (src2.at<cv::Vec3b>(i, j)[ch] == 0) {
                    pano.at<cv::Vec3b>(i, j)[ch] = src1.at<cv::Vec3b>(i, j)[ch];
                } else { // overlapping zone
                    pano.at<cv::Vec3b>(i, j)[ch] = (src1.at<cv::Vec3b>(i, j)[ch]
                            + src2.at<cv::Vec3b>(i, j)[ch]) / 2;
                }
            }
        }
    }

    return pano;
}

