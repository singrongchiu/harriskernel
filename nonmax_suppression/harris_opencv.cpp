// g++ -O2 harris_opencv.cpp -o harris_opencv `pkg-config --cflags --libs opencv4`
// ./harris_opencv "/home/mmpug/Desktop/18645--How-to-Write-Fast-Code-I-/project/harriskernel/nonmax_suppression/chessboard.jpg" 
// "/home/mmpug/Desktop/18645--How-to-Write-Fast-Code-I-/project/harriskernel/nonmax_suppression/corners_opencv.jpg"

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input_image> <output_image>\n";
        return 1;
    }

    const char* in_path  = argv[1];
    const char* out_path = argv[2];

    // Load image as grayscale
    cv::Mat src = cv::imread(in_path, cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: could not load " << in_path << "\n";
        return 1;
    }

    // Convert to float
    cv::Mat src_float;
    src.convertTo(src_float, CV_32F);

    // Harris response
    cv::Mat dst;
    int blockSize = 2;
    int ksize     = 3;
    double k      = 0.04;
    cv::cornerHarris(src_float, dst, blockSize, ksize, k);

    // Normalize for visualization (0–255)
    cv::Mat dst_norm;
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32F);

    cv::Mat dst_norm_u8;
    dst_norm.convertTo(dst_norm_u8, CV_8U);

    // Convert grayscale → BGR for drawing red points
    cv::Mat color;
    cv::cvtColor(src, color, cv::COLOR_GRAY2BGR);

    // Choose threshold in 0–255 range after normalization
    const int thresh = 110;
    int num_corners = 0;

    for (int y = 0; y < dst_norm_u8.rows; y++) {
        for (int x = 0; x < dst_norm_u8.cols; x++) {
            if (dst_norm_u8.at<unsigned char>(y, x) > thresh) {
                cv::circle(color, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
                num_corners++;
            }
        }
    }

    std::cout << "Detected " << num_corners << " corners.\n";

    // Save output
    if (!cv::imwrite(out_path, color)) {
        std::cerr << "Error: could not write " << out_path << "\n";
        return 1;
    }

    std::cout << "Saved: " << out_path << "\n";
    return 0;
}
