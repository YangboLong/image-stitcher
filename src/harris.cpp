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
/// @File    harris.cpp
/// @Brief   Implementation of the Harris corner detection algorithm
/// @Author  Yangbo Long <yangbo.long.mav@gmail.com>
/// @Version 0.1.0
/// @Date    2017-04-01

#include "harris.h"
#include "misc.h"

/// --------------------------------------------------------------------------
/// @Brief   1. Compute Gaussian derivatives at each pixel;
///          2. Compute second moment matrix in a Gaussian window around each pixel;
///          3. Compute corner response function R.
///
/// @Param   img, grayscale image
/// --------------------------------------------------------------------------
Harris::Harris(cv::Mat &img) {
    // gaussian window size
    win_size_ = 5;

    // sensitivity factor
    alpha_ = 0.06;

    // patch size for nonmaximum suppression
    patch_size_ = 3;

    // compute Gaussian derivatives (sobel filter) at each pixel
    Derivatives derivs = compute_derivatives(img);

    // apply Gaussian window to derivatives
    Derivatives matrix = second_moment_matrix(derivs);

    // compute Harris response
    harris_response_ = compute_harris_response(matrix);
}

/// --------------------------------------------------------------------------
/// @Brief   1. Threshold the Harris response R;
///          2. Find local maxima of response function (nonmaximum suppression).
///
/// @Param   percentage, only detect a percentage of all the pixels
///
/// @Return  The detected corner points
/// --------------------------------------------------------------------------
std::vector<CornerPoint>
Harris::nonmax_suppression(float percentage) {
    // use a mark matrix to mark local maximas
    cv::Mat mark_matrix(harris_response_.rows, harris_response_.cols, CV_32F,
            cv::Scalar::all(0));

    // std::cout << harris_response_.rows << std::endl << harris_response_.cols << std::endl;
    // create a vector for all corner points
    std::vector<CornerPoint> points;
    for (int r = 0; r < harris_response_.rows; r++) {
        for (int c = 0; c < harris_response_.cols; c++) {
            cv::Point p(r, c);

            CornerPoint cp;
            cp.response = harris_response_.at<float>(r, c);
            cp.point = p;

            points.push_back(cp);
        }
    }
    // Misc::print_mat(harris_response_);
    // std::cout << points.size() << " points" << std::endl;

    // sort corner points by response
    sort(points.begin(), points.end(), CompareResponse());

    size_t i = 0;
    std::vector<CornerPoint> top_points;

    // get top points by percentage
    size_t num_points = harris_response_.rows * harris_response_.cols * percentage;
    while (top_points.size() < num_points) {
        if (i == points.size()) break;

        // check if point marked in mark_matrix
        if(mark_matrix.at<int>(points[i].point.x, points[i].point.y) == 0) {
            for (int r = -patch_size_; r <= patch_size_; r++) {
                for (int c = -patch_size_; c <= patch_size_; c++) {
                    int sx = points[i].point.x + c;
                    int sy = points[i].point.y + r;

                    // bound checking
                    if(sx > mark_matrix.rows - 1)   sx = mark_matrix.rows - 1;
                    if(sx < 0)  sx = 0;
                    if(sy > mark_matrix.cols - 1)   sy = mark_matrix.cols - 1;
                    if(sy < 0)  sy = 0;

                    // mark a corner point and its neighbors within the suppression box
                    mark_matrix.at<int>(sx, sy) = 1;
                }
            }

            // convert back to original image coordinate system:
            // 1 is the half window size of sobel filter and
            // win_size_ is the half gaussian filter size
            points[i].point.x += 1 + win_size_;
            points[i].point.y += 1 + win_size_;
            top_points.push_back(points[i]);
        }

        i++;
    }

    return top_points;
}

/// --------------------------------------------------------------------------
/// @Brief   Compute the gradient of image intensity I(x,y) around (x,y) by
///          approximating the partial derivative with finite differences:
///          Ix = [f(x+1, y) - f(x,y)] / 1, which is equvalent to correlating
///          with a kernel [-1 1]. Sobel filter is another approximation and
///          is used for the implementation here.
///
/// @Param   img, grayscale image
///
/// @Return  Intensity gradient Ix, Iy, and Ix*Iy
/// --------------------------------------------------------------------------
Derivatives Harris::compute_derivatives(cv::Mat &img) {
    // vertical direction
    cv::Mat vertical(img.rows - 2, img.cols, CV_32F);
    for(int r = 1; r < img.rows - 1; r++) {
        for(int c = 0; c < img.cols; c++) {

            float a1 = img.at<float>(r - 1, c);
            float a2 = img.at<float>(r, c);
            float a3 = img.at<float>(r + 1, c);

            // vertical.at<float>(r - 1, c) = a1 + 2 * a2 + a3;
            vertical.at<float>(r - 1, c) = a1 + a2 + a3;
        }
    }

    // horizontal direction
    cv::Mat horizontal(img.rows, img.cols - 2, CV_32F);
    for(int r = 0; r < img.rows; r++) {
        for(int c = 1; c < img.cols - 1; c++) {

            float a1 = img.at<float>(r, c - 1);
            float a2 = img.at<float>(r, c);
            float a3 = img.at<float>(r, c + 1);

            // horizontal.at<float>(r, c - 1) = a1 + 2 * a2 + a3;
            horizontal.at<float>(r, c - 1) = a1 + a2 + a3;
        }
    }

    // apply Sobel filter to compute intensity gradients
    cv::Mat ix2(img.rows - 2, img.cols - 2, CV_32F);
    cv::Mat iy2(img.rows - 2, img.cols - 2, CV_32F);
    cv::Mat ixy(img.rows - 2, img.cols - 2, CV_32F);

    for(int r = 0; r < img.rows - 2; r++) {
        for(int c = 0; c < img.cols - 2; c++) {
            // intensity derivatives Ix, Iy, and Ix*Iy
            ix2.at<float>(r, c) = -horizontal.at<float>(r, c) +
                                 horizontal.at<float>(r + 2, c);
            iy2.at<float>(r, c) = -vertical.at<float>(r, c) +
                                  vertical.at<float>(r, c + 2);
            ixy.at<float>(r, c) = ix2.at<float>(r, c) * iy2.at<float>(r, c);
            // Ix*Ix, Iy*Iy
            ix2.at<float>(r, c) *= ix2.at<float>(r, c);
            iy2.at<float>(r, c) *= iy2.at<float>(r, c);
        }
    }

    Derivatives derivs;
    derivs.ix2 = ix2;
    derivs.iy2 = iy2;
    derivs.ixy = ixy;

    return derivs;
}

/// --------------------------------------------------------------------------
/// @Brief   Gaussian filter with standard deviation sigma = 1
///
/// @Param   img
///
/// @Return  Image filtered with Gaussian filter
/// --------------------------------------------------------------------------
cv::Mat Harris::gaussian_filter(cv::Mat &img) {
    // decompose gaussian filter to reduce time complexity
    cv::Mat interm(img.rows - win_size_ * 2, img.cols - win_size_ * 2, CV_32F);
    for(int r = win_size_; r < img.rows - win_size_; r++) {
        for(int c = win_size_; c < img.cols - win_size_; c++) {
            float res = 0;

            for(int x = -win_size_; x <= win_size_; x++) {
                float m = 1 / sqrt(2 * M_PI) * exp(-0.5 * x * x); // sigma = 1
                res += m * img.at<float>(r + x, c - win_size_);
            }

            interm.at<float>(r - win_size_, c - win_size_) = res;
        }
    }

    cv::Mat gauss(img.rows - win_size_ * 2, img.cols - win_size_ * 2, CV_32F);
    for(int r = win_size_; r < img.rows - win_size_; r++) {
        for(int c = win_size_; c < img.cols - win_size_; c++) {
            float res = 0;

            for(int y = -win_size_; y <= win_size_; y++) {
                float m = 1 / sqrt(2 * M_PI) * exp(-0.5 * y * y); // sigma = 1
                res += m * interm.at<float>(r - win_size_, c + y);
            }

            gauss.at<float>(r - win_size_, c - win_size_) = res;
        }
    }

    return gauss;
}

/// --------------------------------------------------------------------------
/// @Brief   Aapply Gaussian window function around each pixel to compute the
///          second moment matrix components Ix, Iy, and Ix*Iy.
///
/// @Param   derivs
/// @Param   filterRange
///
/// @Return  The second moment matrix
/// --------------------------------------------------------------------------
Derivatives Harris::second_moment_matrix(Derivatives &derivs) {
    Derivatives matrix;

    matrix.ix2 = gaussian_filter(derivs.ix2);
    matrix.iy2 = gaussian_filter(derivs.iy2);
    matrix.ixy = gaussian_filter(derivs.ixy);

    return matrix;
}

/// --------------------------------------------------------------------------
/// @Brief   Compute the corner response function R
///
/// @Param   matrix
///
/// @Return  Harris response
/// --------------------------------------------------------------------------
cv::Mat Harris::compute_harris_response(Derivatives &matrix) {
    cv::Mat response(matrix.ix2.rows, matrix.ix2.cols, CV_32F);

    for(int r = 0; r < matrix.ix2.rows; r++) {
        for(int c = 0; c < matrix.ix2.cols; c++) {
            float a11, a12, a21, a22;

            a11 = matrix.ix2.at<float>(r, c);
            a12 = matrix.ixy.at<float>(r, c);
            a21 = matrix.ixy.at<float>(r, c);
            a22 = matrix.iy2.at<float>(r, c);

            float det = a11 * a22 - a12 * a21;
            float trace = a11 + a22;

            response.at<float>(r, c) = fabs(det - alpha_ * trace * trace);
        }
    }

    return response;
}

/// --------------------------------------------------------------------------
/// @Brief   Mark corner points in image
///
/// @Param   img, the source image
/// @Param   pts, detected corner points
/// @Param   dim, dimension of the plus mark
/// @Param   col, color of the plus mark
///
/// @Return  An image with corner points marked
/// --------------------------------------------------------------------------
cv::Mat Harris::mark_in_image(cv::Mat &img, std::vector<CornerPoint> &pts,
                              int dim, cv::Vec3b col) {
    cv::Mat ret;
    img.copyTo(ret);

    for(auto &pt : pts) {
        cv::Point center = pt.point;
        for(int i = -dim; i <= dim; i++) {
            ret.at<cv::Vec3b>(cv::Point(center.y, center.x + i)) = col;
            ret.at<cv::Vec3b>(cv::Point(center.y + i, center.x)) = col;
        }
    }

    return ret;
}

