/**
 * @function stitch.cpp
 * @brief Test code for stitching two images using function addWeighted
 * @author Yangbo Long
 * @compile g++ -ggdb -o stitch stitch.cpp `pkg-config --cflags --libs opencv`
 * @run ./stitch 1.png 2.png
 */

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

/// Global variables
const char* source_window1 = "Source image 1";
const char* source_window2 = "Source image 2";
const char* warp_window = "Warp";
const char* dst_window = "Dst";

/**
 * @function main
 */
int main( int, char** argv )
{
    Mat warp_mat( 2, 3, CV_32FC1 );
    Mat src1, src2, img1, img2, dst;
    double alpha = 0.5, beta = 1 - alpha;

    /// Load the image
    src1 = imread( argv[1], IMREAD_COLOR );
    src2 = imread( argv[2], IMREAD_COLOR );

    /// Make the size of img1 twice than the original
    img1 = Mat::zeros( src1.rows, src1.cols * 2, src1.type() );
    src1.copyTo(img1.rowRange(0, src1.rows).colRange(0, src1.cols));

    /// Set img2 the same type and twice the size as src2
    img2 = Mat::zeros( src2.rows, src2.cols * 2, src2.type() );

    /// Set the Affine Transform
    // warp_mat.at<float>(0,0) = 0.941176, warp_mat.at<float>(0,1) = -0.0955882, warp_mat.at<float>(0,2) = 566.419;
    // warp_mat.at<float>(1,0) = -0.0588235, warp_mat.at<float>(1,1) = 0.779412, warp_mat.at<float>(1,2) = 91.0441;
    warp_mat.at<float>(0,0) = 0.833333, warp_mat.at<float>(0,1) = 0.0833333, warp_mat.at<float>(0,2) = 536.083;
    warp_mat.at<float>(1,0) = -0.166667, warp_mat.at<float>(1,1) = 1.08333, warp_mat.at<float>(1,2) = 27.0833;

    cout << "Affine transformation:" << endl;
    cout << warp_mat << endl;

    /// Apply the Affine Transform to the src2 image
    warpAffine( src2, img2, warp_mat, img2.size() );

    /// Blend two images
    // addWeighted( img1, alpha, img2, beta, 0.0, dst);
    dst = Mat::zeros( src2.rows, src2.cols * 2, src2.type() );
    for (int i = 0; i < img1.rows; i++) {
        for (int j = 0; j < img1.cols; j++) {
            for (int ch = 0; ch < 3; ch++) {
                if (img1.at<Vec3b>(i, j)[ch] == 0) {
                    dst.at<Vec3b>(i, j)[ch] = img2.at<Vec3b>(i, j)[ch];
                }
                if (img2.at<Vec3b>(i, j)[ch] == 0) {
                    dst.at<Vec3b>(i, j)[ch] = img1.at<Vec3b>(i, j)[ch];
                }
                if (img1.at<Vec3b>(i, j)[ch] != 0 && img2.at<Vec3b>(i, j)[ch] != 0) {
                    dst.at<Vec3b>(i, j)[ch] = (img1.at<Vec3b>(i, j)[ch] + img2.at<cv::Vec3b>(i, j)[ch]) / 2;
                }
            }
        }
    }

    /// Show what you got
    namedWindow( source_window1, WINDOW_AUTOSIZE );
    imshow( source_window1, src1 );
    namedWindow( source_window2, WINDOW_AUTOSIZE );
    imshow( source_window2, src2 );

    namedWindow( warp_window, WINDOW_AUTOSIZE );
    imshow( warp_window, img2 );

    namedWindow( dst_window, WINDOW_AUTOSIZE );
    imshow( dst_window, dst );

    /// Wait until user exits the program
    waitKey(0);

    return 0;
}
