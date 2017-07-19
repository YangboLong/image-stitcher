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

#include "ransac.h"
#include "fund_mat_model.h"
#include "affine_model.h"

/// --------------------------------------------------------------------------
/// @Brief   Compute correlation between every descriptor in one image and
///          every descriptor in the other image.
///
/// @Param   des1, descriptors in image 1
/// @Param   des2, descriptors in image 2
///
/// @Return  Correlation matrix
/// --------------------------------------------------------------------------
std::vector<std::vector<float>> correlate(Descriptor &des1, Descriptor &des2) {
    // set up multithreaded computing
    int num_threads = std::max(1, omp_get_max_threads());
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    // rows and columns of the descriptors_ matrix
    size_t des1_rows = des1.get_descriptors().size();
    size_t des1_cols = des1.get_descriptors()[0].size();
    size_t des2_rows = des2.get_descriptors().size();
    size_t des2_cols = des2.get_descriptors()[0].size();
    std::vector<std::vector<float>> c1(des1_rows, std::vector<float>(des1_cols));
    std::vector<std::vector<float>> c2(des2_rows, std::vector<float>(des2_cols));

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

    std::vector<std::vector<float>> c2t(des2_cols, std::vector<float>(des2_rows));
#pragma omp parallel for
    for (size_t i = 0; i < des2_rows; i++) {
        for (size_t j = 0; j < des2_cols; j++) {
            c2t[j][i] = c2[i][j]; // c2 transpose, 169xm
        }
    }

    // correlation matrix, nxm
    std::vector<std::vector<float>> corr(des1_rows, std::vector<float>(des2_rows));
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
void absolute(std::vector<std::vector<float>> &corr) {
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
void normalize(std::vector<std::vector<float>> &corr) {
    std::vector<std::vector<float>> res;
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
std::vector<std::array<int, 2>>
select_top(std::vector<std::vector<float>> &corr, int count) {
    std::vector<std::array<int, 2>> res;
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
    std::vector<CornerPoint> cpts1 = harris1.nonmax_suppression(percentage);
    std::vector<CornerPoint> cpts2 = harris2.nonmax_suppression(percentage);
    // mark top corner points in images
    int marker_size = 5;
    cv::Mat img_dst1 = harris1.mark_in_image(img_src1, cpts1,
            marker_size, cv::Vec3b(0, 0, 255));
    cv::Mat img_dst2 = harris2.mark_in_image(img_src2, cpts2,
            marker_size, cv::Vec3b(0, 0, 255));
    // display result
    Misc::display_image(img_dst1, 1);
    Misc::display_image(img_dst2, 2);

    // form descriptors
    Descriptor des1(cpts1, g1), des2(cpts2, g2);
    // compute the correlation between every descriptor pair
    std::vector<std::vector<float>> corr = correlate(des1, des2);
    // compute the absolute values for the corr elements
    absolute(corr);
    // normalize the correlation matrix
    normalize(corr);

    // select only the top max_count descriptor pairs
    int max_count = std::min<size_t>(100, std::min(cpts1.size(), cpts2.size()));
    std::vector<std::array<int, 2>> des_pairs = select_top(corr, max_count);
    // selected points coordinates in image
    std::vector<cv::Point> sel_pts1, sel_pts2;
    for (size_t i = 0; i < des_pairs.size(); i++) {
        sel_pts1.push_back(cpts1[des_pairs[i][0]].point);
        sel_pts2.push_back(cpts2[des_pairs[i][1]].point);
    }
    // print point coordinates
    // Misc::print_point(sel_pts1);
    // Misc::print_point(sel_pts2);

    std::vector<std::shared_ptr<AbstrParam>> sel_pts1_homo, sel_pts2_homo;
    for (int i = 0; i < max_count; i++) {
        // note that x and y coordinates are swapped to match the pixel frame
        std::shared_ptr<AbstrParam> cand_pt1
            = std::make_shared<PointHomo>(sel_pts1[i].y, sel_pts1[i].x);
        std::shared_ptr<AbstrParam> cand_pt2
            = std::make_shared<PointHomo>(sel_pts2[i].y, sel_pts2[i].x);
        sel_pts1_homo.push_back(cand_pt1);
        sel_pts2_homo.push_back(cand_pt2);
    }

    // estimate the fundamental matrix
    RANSAC<FundMatModel, 8> estimator;
    estimator.initialize(5000);
    int start = cv::getTickCount();
    estimator.estimate(sel_pts1_homo, sel_pts2_homo);
    int end = cv::getTickCount();
    std::cout << "RANSAC took " << (float)(end - start) / cv::getTickFrequency()
              << " s to estimate the fundamental matrix." << std::endl;
    auto best_inliers = estimator.get_best_inliers();

    // print the fundamental matrix
    auto fund_matrix = estimator.get_best_matrix();
    std::cout << "Fundamental matrix:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << fund_matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // matched corner points
    std::array<std::vector<CornerPoint>, 2> mpts;
    for (int i = 0; i < 2; i++) {
        for (auto &inlier : best_inliers[i]) {
            CornerPoint cp;
            auto pt = std::dynamic_pointer_cast<PointHomo>(inlier);
            // swap coordinates to get back to the image frame
            cp.point.x = pt->point_homo_[1];
            cp.point.y = pt->point_homo_[0];
            mpts[i].push_back(cp);
        }
    }
    // mark matched corner points in images
    cv::Mat img_mcp1 = harris1.mark_in_image(img_dst1, mpts[0],
            marker_size, cv::Vec3b(0, 255, 0));
    cv::Mat img_mcp2 = harris2.mark_in_image(img_dst2, mpts[1],
            marker_size, cv::Vec3b(0, 255, 0));
    // display result
    Misc::display_image(img_mcp1, 1);
    Misc::display_image(img_mcp2, 2);

    // estimate the affine transformation
    RANSAC<AffineModel, 3> affine_estimator;
    affine_estimator.initialize(5000);
    start = cv::getTickCount();
    affine_estimator.estimate(best_inliers[0], best_inliers[1]);
    end = cv::getTickCount();
    std::cout << "RANSAC took " << (float)(end - start) / cv::getTickFrequency()
              << " s to estimate the affine transformation." << std::endl;
    best_inliers = affine_estimator.get_best_inliers();

    // affine transformed corner points
    std::array<std::vector<CornerPoint>, 2> aff_pts;
    for (int i = 0; i < 2; i++) {
        for (auto &inlier : best_inliers[i]) {
            CornerPoint cp;
            auto pt = std::dynamic_pointer_cast<PointHomo>(inlier);
            // swap coordinates to get back to the image frame
            cp.point.x = pt->point_homo_[1];
            cp.point.y = pt->point_homo_[0];
            aff_pts[i].push_back(cp);
        }
    }
    // mark matched corner points in images
    cv::Mat img_aff1 = harris1.mark_in_image(img_mcp1, aff_pts[0],
            marker_size, cv::Vec3b(255, 0, 0));
    cv::Mat img_aff2 = harris2.mark_in_image(img_mcp2, aff_pts[1],
            marker_size, cv::Vec3b(255, 0, 0));
    // display result
    Misc::display_image(img_aff1, 1);
    Misc::display_image(img_aff2, 2);
    cv::imwrite("img1_matched.png", img_aff1);
    cv::imwrite("img2_matched.png", img_aff2);

    // stitch two images together and display the panorama
    auto affine_matrix = affine_estimator.get_best_matrix();
    std::cout << "Affine transformation:" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << affine_matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
    cv::Mat pano = Misc::stitch_images(img_src1, img_src2, affine_matrix);
    Misc::display_image(pano, 3);
    cv::imwrite("pano.png", pano);

    cv::waitKey(0);
    return 0;
}

