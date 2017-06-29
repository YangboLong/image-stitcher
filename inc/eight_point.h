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
/// @File    eight_point.h
/// @Brief   Eight-point algorithm
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-06-19

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

class EightPoint {
public:
    Eigen::Affine2d normalize(Eigen::Ref<const Eigen::MatrixXd> pts);
    void comp_fund_mat(Eigen::Ref<const Eigen::MatrixXd> pts1,
                       Eigen::Ref<const Eigen::MatrixXd> pts2);
    Eigen::Matrix3d fund_mat_from_pts(Eigen::Ref<const Eigen::MatrixXd> pts1,
                                      Eigen::Ref<const Eigen::MatrixXd> pts2);
    std::array<std::array<double, 3>, 3> get_fund_mat();

private:
    Eigen::Matrix3d fund_mat_;
};

