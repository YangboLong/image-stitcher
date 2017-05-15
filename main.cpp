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

#include <iostream>
#include <algorithm>
#include "harris.h"
#include "descriptor.h"
#include "misc.h"
#include <omp.h>

/// --------------------------------------------------------------------------
/// @Brief   Compute correlation between every descriptor in one image and
///          every descriptor in the other image.
///
/// @Param   des1, descriptors in image 1
/// @Param   des2, descriptors in image 2
///
/// @Return  Correlation matrix
/// --------------------------------------------------------------------------
std::vector<std::vector<float> > correlate(Descriptor &des1, Descriptor &des2) {
    // set up multithreaded computing
    int num_threads = std::max(1, omp_get_max_threads());
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    // rows and columns of the descriptors_ matrix
    size_t des1_rows = des1.get_descriptors().size();
    size_t des1_cols = des1.get_descriptors()[0].size();
    size_t des2_rows = des2.get_descriptors().size();
    size_t des2_cols = des2.get_descriptors()[0].size();
    std::vector<std::vector<float> > c1(des1_rows, std::vector<float>(des1_cols));
    std::vector<std::vector<float> > c2(des2_rows, std::vector<float>(des2_cols));

#pragma omp parallel for
    for (size_t i = 0; i < des1_rows; i++) {
        for (size_t j = 0; j < des1_cols; j++) {
            c1[i][j] = (des1.get_descriptors()[i][j] - des1.get_mean()[i])
                / des1.get_std()[i]; // nx169
        }
    }
#pragma omp parallel for
    for (size_t i = 0; i < des2_rows; i++) {
        for (size_t j = 0; j < des2_cols; j++) {
            c2[i][j] = (des2.get_descriptors()[i][j] - des2.get_mean()[i])
                / des2.get_std()[i]; // mx169
        }
    }

    std::vector<std::vector<float> > c2t(des2_cols, std::vector<float>(des2_rows));
#pragma omp parallel for
    for (size_t i = 0; i < des2_rows; i++) {
        for (size_t j = 0; j < des2_cols; j++) {
            c2t[j][i] = c2[i][j]; // c2 transpose, 169xm
        }
    }

    // correlation matrix, nxm
    std::vector<std::vector<float> > corr(des1_rows, std::vector<float>(des2_rows));
#pragma omp parallel for
    for (size_t i = 0; i < des1_rows; i++) {
        for (size_t j = 0; j < des2_rows; j++) {
            for (size_t k = 0; k < des1_cols; k++) { // des1_cols = des2_cols, 169
                corr[i][j] += c1[i][k] * c2t[k][j];
            }
        }
    }
    return corr;
}

/// --------------------------------------------------------------------------
/// @Brief   Compute the absolute values for the corr elements
///
/// @Param   corr, correlation matrix
/// --------------------------------------------------------------------------
void absolute(std::vector<std::vector<float> > &corr) {
    size_t rows = corr.size(), cols = corr[0].size();
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (corr[i][j] < 0) {
                corr[i][j] *= -1;
            }
        }
    }
}

/// --------------------------------------------------------------------------
/// @Brief   Normalize the correlation matrix
///
/// @Param   corr, correlation matrix
/// --------------------------------------------------------------------------
void normalize(std::vector<std::vector<float> > &corr) {
    std::vector<std::vector<float> > res;
    for (size_t i = 0; i < corr.size(); i++) {
        float norm = sqrt(std::inner_product(corr[i].begin(), corr[i].end(),
                    corr[i].begin(), 0.0));
        for (size_t j = 0; j < corr[i].size(); j++) {
            corr[i][j] /= norm;
        }
    }
}

/// --------------------------------------------------------------------------
/// @Brief   Select only the top few descriptor pairs
///
/// @Param   corr, correlation matrix
/// @Param   count, count of descriptor pairs to select
///
/// @Return  Coordinates of the selected feature points
/// --------------------------------------------------------------------------
std::vector<std::array<int, 2> >
select_top(std::vector<std::vector<float> > &corr, int count) {
    std::vector<std::array<int, 2> > res;
    int rows = corr.size(), cols = corr[0].size();
    float max_elem = 0.0;
    for (int i = 0; i < count; i++) {
        std::array<int, 2> coordinates; // coordinates of max element in corr
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (max_elem < corr[r][c]) {
                    max_elem = corr[r][c];
                    coordinates.at(0) = r;
                    coordinates.at(1) = c;
                }
            }
        }
        // clear elements in the same row and column as max_elem
        for (int c = 0; c < cols; c++) {
            corr[coordinates.at(0)][c] = 0;
        }
        for (int r = 0; r < cols; r++) {
            corr[r][coordinates.at(1)] = 0;
        }
        // save a top pair
        res.push_back(coordinates);

        // look for next max element
        max_elem = 0.0;
    }
    return res;
}

/// --------------------------------------------------------------------------
/// @Brief   Entry point
///
/// @Param   argc, three arguments including binary and two input images
/// @Param   argv, arguments
///
/// @Return  return value
/// --------------------------------------------------------------------------
int main(int argc, char **argv) {
    cv::Mat img_src1, img_src2, gray1, gray2, g1, g2;

    if (argc <= 2) {
        std::cout << "Usage: ./stitcher [path/to/image1] [path/to/image2]" << std::endl;
        return -1;
    } else {
        img_src1 = cv::imread(argv[1]);
        img_src2 = cv::imread(argv[2]);
    }

    // convert color image to grayscale
    cvtColor(img_src1, gray1, CV_BGR2GRAY);
    gray1.convertTo(g1, CV_32F);
    cvtColor(img_src2, gray2, CV_BGR2GRAY);
    gray2.convertTo(g2, CV_32F);

    // run harris algorithm
    Harris harris1(g1);
    Harris harris2(g2);
    // local nonmaximum suppression
    float percentage = 0.001;
    std::vector<CornerPoint> pts1 = harris1.nonmax_suppression(percentage);
    std::vector<CornerPoint> pts2 = harris2.nonmax_suppression(percentage);
    // mark top corner points in image
    int marker_size = 5;
    cv::Mat img_dst1 = harris1.mark_in_image(img_src1, pts1, marker_size);
    cv::Mat img_dst2 = harris2.mark_in_image(img_src2, pts2, marker_size);
    // display result
    Misc::display_image(img_dst1, 1);
    Misc::display_image(img_dst2, 2);

    // form descriptors
    Descriptor des1(pts1, g1), des2(pts2, g2);
    // compute the correlation between every descriptor pair
    std::vector<std::vector<float> > corr = correlate(des1, des2);
    // compute the absolute values for the corr elements
    absolute(corr);
    // normalize the correlation matrix
    normalize(corr);

    // select only the top max_count descriptor pairs
    int max_count = std::min<size_t>(100, std::min(pts1.size(), pts2.size()));
    std::vector<std::array<int, 2> > des_pairs = select_top(corr, max_count);
    // selected points coordinates in image
    std::vector<cv::Point> sel_pts1, sel_pts2;
    for (size_t i = 0; i < des_pairs.size(); i++) {
        sel_pts1.push_back(pts1[des_pairs[i][0]].point);
        sel_pts2.push_back(pts2[des_pairs[i][1]].point);
    }
    Misc::print_point(sel_pts1);
    Misc::print_point(sel_pts2);

    cv::waitKey(0);
    return 0;
}

