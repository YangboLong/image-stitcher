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
    cv::Mat pano(height, width, CV_8UC3, cv::Scalar(0, 0,0));

    // pixel coordinates in img2, homogeneous points
    int height2 = img2.rows, width2 = img2.cols, pts = height2 * width2;
    std::vector<std::vector<int>> img2_pts(pts, std::vector<int>(3));
    int idx = 0;
    for (int i = 0; i < width2; i++) {
        for (int j = 0; j < height2; j++) {
            img2_pts[idx][0] = i;
            img2_pts[idx][1] = j;
            img2_pts[idx][2] = 1;
            idx++;
        }
    }

    // apply affine transformation to pixel coordinates in img2
    std::vector<std::vector<double>> new_img2_pts(3, std::vector<double>(pts));
    std::vector<std::vector<int>> img2_pts_t(3, std::vector<int>(pts));
    for (size_t i = 0; i < pts; i++) {
        for (size_t j = 0; j < 3; j++) {
            img2_pts_t[j][i] = img2_pts[i][j]; // img2_pts transpose
        }
    }
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < pts; j++) {
            for (int k = 0; k < 3; k++) {
                new_img2_pts[i][j] += aff_mat[i][k] * img2_pts_t[k][j];
            }
        }
    }

    // fill the top left of the pano image with img1
    for (int i = 0; i < img1.rows; i++) {
        for (int j = 0; j < img1.cols; j++) {
            pano.at<cv::Vec3b>(i, j) = img1.at<cv::Vec3b>(i, j);
        }
    }

    // blend pixels in the overlapping zone and stitch images
    for (int i = 0; i < pts; i++) {
        int r, c; // rows and cols in the pano image
        double x = new_img2_pts[1][i], y = new_img2_pts[0][i];
        if (x >= 0 && y >= 0 && x < height && y < width) {
            r = (int)std::round(x);
            c = (int)std::round(y);
            // blend pixels in three channels separately
            for (int ch = 0; ch < 3; ch++) {
                if (c < img1.cols) { // overlapping zone
                    pano.at<cv::Vec3b>(r, c)[ch] = (pano.at<cv::Vec3b>(r, c)[ch] +
                            img2.at<cv::Vec3b>(img2_pts[i][1], img2_pts[i][0])[ch]) / 2;
                } else { // non-overlapping portion in img2
                    pano.at<cv::Vec3b>(r, c)[ch] =
                        img2.at<cv::Vec3b>(img2_pts[i][1], img2_pts[i][0])[ch];
                }
            }
        }
    }

    return pano;
}

