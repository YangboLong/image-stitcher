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
/// @File    eight_point.cpp
/// @Brief   Implementation of the eight-point algorithm
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-06-19

#include "eight_point.h"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

/// --------------------------------------------------------------------------
/// @Brief  Precondition the matrices by performing a normalization of each
///         point set by:
///         • translating center of mass to the origin
///         • scaling so that average distance of points from origin is sqrt(2)
///         • do this normalization to each point set independently
///
/// @Param  pts
///
/// @Return Normalized point coordinates
/// --------------------------------------------------------------------------
Eigen::Affine2d EightPoint::normalize(Eigen::Ref<const Eigen::MatrixXd> pts) {
    Eigen::Vector2d mean = pts.rowwise().mean();
    Eigen::Vector2d stddev =
        (pts.colwise() - mean).array().square().rowwise().mean().sqrt();

    Eigen::Affine2d t;
    t = Eigen::Scaling(1.0 / stddev.norm()) *  Eigen::Translation2d(-mean);
    return t;
}

/// --------------------------------------------------------------------------
/// @Brief   Compute the fundamental matrix from 8 pair of feature points.
///
/// @Param   pts1, point coordinates in image 1
/// @Param   pts2, point coordinates in image 2
///
/// @Return  Fundamental matrix
/// --------------------------------------------------------------------------
void EightPoint::comp_fund_mat(Eigen::Ref<const Eigen::MatrixXd> pts1,
                               Eigen::Ref<const Eigen::MatrixXd> pts2) {
    Eigen::Transform<double, 2, Eigen::Affine> t0 = normalize(pts1);
    Eigen::Transform<double, 2, Eigen::Affine> t1 = normalize(pts2);

    Eigen::Matrix<double, 2, Eigen::Dynamic> na =
        (t0.matrix() * pts1.colwise().homogeneous()).colwise().hnormalized();
    Eigen::Matrix<double, 2, Eigen::Dynamic> nb =
        (t1.matrix() * pts2.colwise().homogeneous()).colwise().hnormalized();

    Eigen::Matrix3d Fn = fund_mat_from_pts(na, nb);
    fund_mat_ = (t0.matrix().transpose() * Fn * t1.matrix());
}

/// --------------------------------------------------------------------------
/// @Brief   Helper function for calculating the fundamental matrix
///
/// @Param   pts1, point coordinates in image 1
/// @Param   pts2, point coordinates in image 2
///
/// @Return  Fundamental matrix
/// --------------------------------------------------------------------------
Eigen::Matrix3d
EightPoint::fund_mat_from_pts(Eigen::Ref<const Eigen::MatrixXd> pts1,
                              Eigen::Ref<const Eigen::MatrixXd> pts2) {
    eigen_assert(pts1.cols() == pts2.cols());
    eigen_assert(pts1.rows() == pts2.rows());
    eigen_assert(pts1.cols() >= 8);

    // Setup system of equations Ax = 0. There will be one row in A for
    // each correspondence.
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(pts1.cols(), 9);

    for (Eigen::DenseIndex i = 0; i < pts1.cols(); ++i) {
        const auto &ca = pts1.col(i);
        const auto &cb = pts2.col(i);

        auto r = A.row(i);

        r(0) = cb.x() * ca.x();     // F11
        r(1) = cb.x() * ca.y();     // F21
        r(2) = cb.x();              // F31
        r(3) = cb.y() * ca.x();     // F12
        r(4) = cb.y() * ca.y();     // F22
        r(5) = cb.y();              // F32
        r(6) = ca.x();              // F13
        r(7) = ca.y();              // F23
        r(8) = 1.0;                 // F33
    }

    // Seek for a least squares solution such that |Ax| = 1. Given by the unit
    // eigenvector of A'A associated with the smallest eigenvalue.
    Eigen::SelfAdjointEigenSolver< Eigen::Matrix<double, Eigen::Dynamic, 9> > e;
    e.compute((A.transpose() * A));
    eigen_assert(e.info() == Eigen::Success);

    // Sorted ascending by eigenvalue.
    Eigen::Matrix<double, 1, 9> f = e.eigenvectors().col(0);

    Eigen::Matrix3d F;
    F << f(0), f(3), f(6),
         f(1), f(4), f(7),
         f(2), f(5), f(8);

    // Enforce singularity constraint such that rank(F) = 2, which is the
    // closest singular matrix to F under Frobenius norm.
    Eigen::JacobiSVD<Eigen::Matrix3d>
        svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::DiagonalMatrix<double, 3>
        dPrime(svd.singularValues()(0), svd.singularValues()(1), 0.0);
    Eigen::Matrix3d FPrime =
        svd.matrixU() * dPrime * svd.matrixV().transpose();

    return FPrime;
}

/// --------------------------------------------------------------------------
/// @Brief   Convert Matrix3d and return a 2d array.
///
/// @Return  Fundamental matrix
/// --------------------------------------------------------------------------
std::array<std::array<double, 3>, 3> EightPoint::get_fund_mat() {
    std::array<std::array<double, 3>, 3> mat;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = fund_mat_(i, j);
        }
    }
    return mat;
}

