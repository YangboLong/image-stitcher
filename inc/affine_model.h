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
/// @File    affine_model.h
/// @Brief   Affine transformation model derived from the abstract model
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-07-03

#pragma once

#include "abstract_model.h"
#include "fund_mat_model.h" // class PointHomo
#include <opencv2/opencv.hpp>

/// --------------------------------------------------------------------------
/// @Brief  Affine transformation model with 3 randomly selected points as its
///         minimun parameters.
/// --------------------------------------------------------------------------
class AffineModel : public AbstractModel<3> {
protected:
    const size_t num_pts_; // number of matched feature points
    std::vector<std::shared_ptr<PointHomo>> pts1_, pts2_; // all matched points
    cv::Point2f tri1_[3], tri2_[3]; // three point pairs in the two images
    std::vector<std::shared_ptr<PointHomo>> ldiff1_, ldiff2_; // least diff points
    std::array<std::vector<std::shared_ptr<AbstrParam>>, 2> ldiff_;
    cv::Mat warp_mat_; // warp matrix, 2x3
    std::array<std::array<double, 3>, 3> affine_mat_; // affine matrix, 3x3
    std::vector<double> affine_diff_; // affine difference
    double error_; // error of affine transformation

public:
    /// --------------------------------------------------------------------------
    /// @Brief   Initialize class AffineModel.
    ///
    /// @Param   inputs1, 3 random matched feature points in image 1
    /// @Param   inputs2, 3 random matched feature points in image 2
    /// @Param   pts1, all matched feature points in image 1
    /// @Param   pts2, all matched feature points in image 2
    /// --------------------------------------------------------------------------
    AffineModel(std::vector<std::shared_ptr<AbstrParam>> inputs1,
                std::vector<std::shared_ptr<AbstrParam>> inputs2,
                std::vector<std::shared_ptr<AbstrParam>> pts1,
                std::vector<std::shared_ptr<AbstrParam>> pts2)
        : num_pts_(pts1.size()),
          warp_mat_(2, 3, CV_64F),
          error_(0.0) {
        // checking input parameters
        if (num_pts_ != pts2.size()) {
            throw std::runtime_error("AffineModel: number of points wrong.");
        }

        for (size_t i = 0; i < num_pts_; i++) {
            auto pt1 = std::dynamic_pointer_cast<PointHomo>(pts1[i]);
            auto pt2 = std::dynamic_pointer_cast<PointHomo>(pts2[i]);
            if (pt1 == nullptr || pt2 == nullptr) {
                throw std::runtime_error("AffineModel: inputs type wrong.");
            } else {
                pts1_.push_back(pt1);
                pts2_.push_back(pt2);
            }
        }
        // initialize class members
        initialize(inputs1, inputs2);
    }

    virtual void
    initialize(std::vector<std::shared_ptr<AbstrParam>> inputs1,
               std::vector<std::shared_ptr<AbstrParam>> inputs2) override {
        // input parameters: 3 homogeneous point pairs
        if (inputs1.size() != 3 || inputs2.size() != 3) {
            throw std::runtime_error("AffineModel: number of inputs wrong.");
        }

        // check for AbstrParam types
        for (int i = 0; i < 3; i++) {
            auto pt1 = std::dynamic_pointer_cast<PointHomo>(inputs1[i]);
            auto pt2 = std::dynamic_pointer_cast<PointHomo>(inputs2[i]);
            if (pt1 == nullptr || pt2 == nullptr) {
                throw std::runtime_error("AffineModel: input types mismatch.");
            } else {
                tri1_[i] = cv::Point2f(pt1->point_homo_[0], pt1->point_homo_[1]);
                tri2_[i] = cv::Point2f(pt2->point_homo_[0], pt2->point_homo_[1]);
            }
        }

        // compute the affine transformation matrix
        warp_mat_ = cv::getAffineTransform(tri2_, tri1_);
        // extend the 2x3 warp matrix to be 3x3 affine matrix
        extend_warp_mat();
    }

    /// --------------------------------------------------------------------------
    /// @Brief   Evaluate 3 points and compute transformation error.
    ///
    /// @Param   inputs, input parameters to be evaluated
    /// @Param   thres, threshold for finding inliers
    ///
    /// @Return  the error of the computed affine transformation using 3 pair of
    ///          points, and the 3 pair of points
    /// --------------------------------------------------------------------------
    virtual std::tuple<double,
                       std::array<std::vector<std::shared_ptr<AbstrParam>>, 2>,
                       std::array<std::array<double, 3>, 3>>
    evaluate() override {
        // compute the affine difference of correspoinding points
        comp_affine_diff();

        // sort affine difference in ascending order
        std::vector<int> sorted_idx;
        for (auto &i : sort_idx(affine_diff_)) {
            sorted_idx.push_back(i);
        }

        // prepare points for a second affine transformation
        // and save the points with least affine difference
        for (int i = 0; i < 3; i++) {
            tri1_[i] = cv::Point2f(pts1_[sorted_idx[i]]->point_homo_[0],
                                   pts1_[sorted_idx[i]]->point_homo_[1]);
            tri2_[i] = cv::Point2f(pts2_[sorted_idx[i]]->point_homo_[0],
                                   pts2_[sorted_idx[i]]->point_homo_[1]);
            ldiff1_.push_back(pts1_[sorted_idx[i]]);
            ldiff2_.push_back(pts2_[sorted_idx[i]]);
        }

        // compute the second affine transformation
        warp_mat_ = cv::getAffineTransform(tri2_, tri1_);
        // extend the 2x3 warp matrix to be 3x3 affine matrix
        extend_warp_mat();

        // clear affine_diff_ before computing it again
        affine_diff_.clear();
        // compute the affine difference again
        comp_affine_diff();
        for (auto &d : affine_diff_) { error_ += d; }

        for (size_t i = 0; i < ldiff1_.size(); i++) {
            ldiff_[0].push_back(ldiff1_[i]);
            ldiff_[1].push_back(ldiff2_[i]);
        }

        return std::make_tuple(error_, ldiff_, affine_mat_);
    }

    void comp_affine_diff() {
        std::vector<std::vector<int>> pts1t(3, std::vector<int>(num_pts_));
        std::vector<std::vector<int>> pts2t(3, std::vector<int>(num_pts_));
        std::vector<std::vector<int>> tmp1(3, std::vector<int>(num_pts_));
        std::vector<std::vector<int>> tmp2(3, std::vector<int>(num_pts_));

        for (size_t i = 0; i < num_pts_; i++) {
            for (size_t j = 0; j < 3; j++) {
                pts1t[j][i] = pts1_[i]->point_homo_[j]; // pts1_ transpose
                pts2t[j][i] = pts2_[i]->point_homo_[j]; // pts2_ transpose
            }
        }
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < num_pts_; j++) {
                for (size_t k = 0; k < 3; k++) {
                    tmp2[i][j] += affine_mat_[i][k] * pts2t[k][j];
                }
                tmp1[i][j] = pow(pts1t[i][j] - tmp2[i][j], 2);
            }
        }
        for (size_t i = 0; i < num_pts_; i++) {
            affine_diff_.push_back(sqrt(tmp1[0][i] + tmp1[1][i] + tmp1[2][i]));
        }
    }

    template <typename Type>
    std::vector<size_t> sort_idx(const std::vector<Type> &v) {
        // initialize original index locations
        std::vector<size_t> idx(v.size());
        iota(idx.begin(), idx.end(), 0);

        // sort indices based on comparing values in v
        sort(idx.begin(), idx.end(),
             [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

        return idx;
    }

    void extend_warp_mat() {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                affine_mat_[i][j] = warp_mat_.at<double>(i, j);
            }
        }
        affine_mat_[2][0] = affine_mat_[2][1] = 0;
        affine_mat_[2][2] = 1;
    }
};

