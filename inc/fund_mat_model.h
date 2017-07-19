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
/// @File    fund_mat_model.h
/// @Brief   Fundamental matrix model derived from the abstract model
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-05-23

#pragma once

#include "abstract_model.h"
#include "eight_point.h"
#include <cmath> // fabs

/// --------------------------------------------------------------------------
/// @Brief  Homogenerous points in 3D
/// --------------------------------------------------------------------------
class PointHomo : public AbstrParam {
public:
    PointHomo(int x, int y) {
        point_homo_[0] = x;
        point_homo_[1] = y;
        point_homo_[2] = 1;
    };

    std::array<int, 3> point_homo_;
};

/// --------------------------------------------------------------------------
/// @Brief  Fundamental matrix model with 8 randomly selected points as its
///         minimun parameters.
/// --------------------------------------------------------------------------
class FundMatModel : public AbstractModel<8> {
protected:
    const size_t num_pts_; // number of feature points
    std::vector<std::shared_ptr<PointHomo>> pts1_, pts2_; // all feature points
    std::vector<std::vector<double>> rand_pts1_, rand_pts2_; // random points
    std::vector<std::shared_ptr<PointHomo>> inliers1_, inliers2_; // all inliers
    std::array<std::vector<std::shared_ptr<AbstrParam>>, 2> inliers_;
    double error_, thres_ie_; // fund_mat_'s error and inlier error threshold
    size_t thres_inliers_; // threshold of number of inliers
    std::vector<size_t> inline_idx_; // indices of inliers
    bool roughly_matched_; // if the inputs have passed the rough matching?
    std::array<std::array<double, 3>, 3> fund_mat_; // fundamental matrix, 3x3

public:
    /// --------------------------------------------------------------------------
    /// @Brief   Initialize class FundMatModel.
    ///
    /// @Param   inputs1, 8 random feature points in image 1
    /// @Param   inputs2, 8 random feature points in image 2
    /// @Param   pts1, all feature points in image 1
    /// @Param   pts2, all feature points in image 2
    /// --------------------------------------------------------------------------
    FundMatModel(std::vector<std::shared_ptr<AbstrParam>> inputs1,
                 std::vector<std::shared_ptr<AbstrParam>> inputs2,
                 std::vector<std::shared_ptr<AbstrParam>> pts1,
                 std::vector<std::shared_ptr<AbstrParam>> pts2)
        : num_pts_(pts1.size()) {
        // checking input parameters
        if (num_pts_ != pts2.size()) {
            throw std::runtime_error("FundMatModel: number of points wrong.");
        }

        for (size_t i = 0; i < num_pts_; i++) {
            auto pt1 = std::dynamic_pointer_cast<PointHomo>(pts1[i]);
            auto pt2 = std::dynamic_pointer_cast<PointHomo>(pts2[i]);
            if (pt1 == nullptr || pt2 == nullptr) {
                throw std::runtime_error("FundMatModel: inputs type wrong.");
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
        // input parameters: 8 homogeneous point pairs
        if (inputs1.size() != 8 || inputs2.size() != 8) {
            throw std::runtime_error("FundMatModel: number of inputs wrong.");
        }

        // check for AbstrParam types
        std::vector<double> vec1_x, vec1_y, vec2_x, vec2_y;
        for (int i = 0; i < 8; i++) {
            auto pt1 = std::dynamic_pointer_cast<PointHomo>(inputs1[i]);
            auto pt2 = std::dynamic_pointer_cast<PointHomo>(inputs2[i]);
            if (pt1 == nullptr || pt2 == nullptr) {
                throw std::runtime_error("FundMatModel: input types mismatch.");
            } else {
                vec1_x.push_back(pt1->point_homo_[0]);
                vec1_y.push_back(pt1->point_homo_[1]);
                vec2_x.push_back(pt2->point_homo_[0]);
                vec2_y.push_back(pt2->point_homo_[1]);
            }
        }
        rand_pts1_.push_back(vec1_x);
        rand_pts1_.push_back(vec1_y);
        rand_pts2_.push_back(vec2_x);
        rand_pts2_.push_back(vec2_y);

        error_ = 0.0;
        thres_ie_ = 0.0001;
        thres_inliers_ = 16;
        roughly_matched_ = false;
    }

    /// --------------------------------------------------------------------------
    /// @Brief   Evaluate 8 points and compute matching error.
    ///
    /// @Param   inputs, input parameters to be evaluated
    /// @Param   thres, threshold for finding inliers
    ///
    /// @Return  the error of the computed fundamental matrix using 8 pair of
    ///          points, and the 8 pair of points
    /// --------------------------------------------------------------------------
    virtual std::tuple<double,
                       std::array<std::vector<std::shared_ptr<AbstrParam>>, 2>,
                       std::array<std::array<double, 3>, 3>>
    evaluate() override {
        // first step, rough match using all feature point pairs
        rough_match();
        // second step, fine match using inliers only
        if (roughly_matched_) {
            fine_match();
            for (size_t i = 0; i < inliers1_.size(); i++) {
                inliers_[0].push_back(inliers1_[i]);
                inliers_[1].push_back(inliers2_[i]);
            }
        } else {
            error_ = std::numeric_limits<double>::max();
            // error_ = 1.0;
        }
        return std::make_tuple(error_, inliers_, fund_mat_);
    }

    /// --------------------------------------------------------------------------
    /// @Brief   Initialize Eigen::MatrixXd from 2D std::vector.
    ///
    /// @Param   pts1, the first Eigen matrix
    /// @Param   pts2, the second Eigen matrix
    /// --------------------------------------------------------------------------
    void init_eigen_mat(Eigen::MatrixXd &pts1, Eigen::MatrixXd &pts2) {
        for (int i = 0; i < 2; i++) {
            pts1.row(i) =
                Eigen::VectorXd::Map(&rand_pts1_[i][0], rand_pts1_[i].size());
            pts2.row(i) =
                Eigen::VectorXd::Map(&rand_pts2_[i][0], rand_pts2_[i].size());
        }
    }

    void rough_match() {
        EightPoint eight_point;
        Eigen::MatrixXd pts1(2, rand_pts1_[0].size()), pts2(2, rand_pts2_[0].size());
        // initialize Eigen::MatrixXd from 2D std::vector
        init_eigen_mat(pts1, pts2);

        // compute fundamental matrix using the normalized 8-point algorithm
        eight_point.comp_fund_mat(pts1, pts2);
        fund_mat_ = eight_point.get_fund_mat();

        // calculate the fundamental matrix's error
        std::vector<double> err = comp_mat_err(pts1_, pts2_);

        // find the indices of inliers
        for (size_t i = 0; i < err.size(); i++) {
            if (err[i] < thres_ie_) {
                inline_idx_.push_back(i);
            }
        }

        // check if there are enough inliers
        if (inline_idx_.size() > thres_inliers_) {
            // sort error in ascending order
            // std::vector<int> sorted_idx;
            // for (auto &i : sort_idx(err)) {
            //     sorted_idx.push_back(i);
            // }
            // inline_idx_.clear();
            // for (size_t i = 0; i < 8; i++) {
            //     inline_idx_.push_back(sorted_idx[i]);
            // }
            // std::sort(inline_idx_.begin(), inline_idx_.end());

            // indicate that they are roughly matched
            roughly_matched_ = true;
            // reset vectors to be ready for fine match
            rand_pts1_.clear();
            rand_pts2_.clear();
            // collect only the inliers for fine matching later
            std::vector<double> vec1_x, vec1_y, vec2_x, vec2_y;
            size_t j = 0;
            for (size_t i = 0; i < pts1_.size(); i++) {
                if (i == inline_idx_[j]) {
                    inliers1_.push_back(pts1_[i]);
                    inliers2_.push_back(pts2_[i]);
                    j++;

                    vec1_x.push_back(pts1_[i]->point_homo_[0]);
                    vec1_y.push_back(pts1_[i]->point_homo_[1]);
                    vec2_x.push_back(pts2_[i]->point_homo_[0]);
                    vec2_y.push_back(pts2_[i]->point_homo_[1]);
                }
            }
            rand_pts1_.push_back(vec1_x);
            rand_pts1_.push_back(vec1_y);
            rand_pts2_.push_back(vec2_x);
            rand_pts2_.push_back(vec2_y);
        }
    }

    void fine_match() {
        EightPoint eight_point;
        Eigen::MatrixXd pts1(2, rand_pts1_[0].size()), pts2(2, rand_pts2_[0].size());
        // initialize Eigen::MatrixXd from 2D std::vector
        init_eigen_mat(pts1, pts2);

        // compute fundamental matrix again using the inliers only
        eight_point.comp_fund_mat(pts1, pts2);
        fund_mat_ = eight_point.get_fund_mat();

        // calculate the fundamental matrix's error
        std::vector<double> abs_err = comp_mat_err(inliers1_, inliers2_);
        for (size_t i = 0; i < abs_err.size(); i++) {
            error_ += abs_err[i];
        }
    }

    std::vector<double>
    comp_mat_err(std::vector<std::shared_ptr<PointHomo>> &pts1,
                 std::vector<std::shared_ptr<PointHomo>> &pts2) {
        size_t num_pts = pts1.size();
        std::vector<std::array<double, 3>> tmp(num_pts, {{0, 0, 0}});
        std::vector<std::vector<double>> err(num_pts, std::vector<double>(num_pts));
        std::vector<double> err_diag(num_pts);
        for (size_t i = 0; i < num_pts; i++) {
            for (size_t j = 0; j < 3; j++) {
                for (size_t k = 0; k < 3; k++) {
                    tmp[i][j] += pts1[i]->point_homo_[k] * fund_mat_[k][j];
                }
            }
        }
        std::vector<std::vector<int>> pts2t(3, std::vector<int>(num_pts));
        for (size_t i = 0; i < num_pts; i++) {
            for (size_t j = 0; j < 3; j++) {
                pts2t[j][i] = pts2[i]->point_homo_[j]; // pts2 transpose
            }
        }
        for (size_t i = 0; i < num_pts; i++) {
            for (size_t j = 0; j < num_pts; j++) {
                for (size_t k = 0; k < 3; k++) {
                    err[i][j] += tmp[i][k] * pts2t[k][j];
                }
                if (i == j) {
                    if (!roughly_matched_) {
                        err_diag[i] = err[i][j] * err[i][j];
                    } else {
                        err_diag[i] = fabs(err[i][j]);
                    }
                }
            }
        }

        return err_diag;
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
};

