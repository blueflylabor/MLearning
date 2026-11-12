#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
using namespace std;

// 自定义离散傅里叶变换平移函数 (FFT Shift)
void fftshift(const cv::Mat& src, cv::Mat& dst) {
    dst = src.clone();
    int cx = dst.cols / 2;
    int cy = dst.rows / 2;

    // 定义四个象限
    cv::Mat q0(dst, cv::Rect(0, 0, cx, cy));   // 左上
    cv::Mat q1(dst, cv::Rect(cx, 0, cx, cy));  // 右上
    cv::Mat q2(dst, cv::Rect(0, cy, cx, cy));  // 左下
    cv::Mat q3(dst, cv::Rect(cx, cy, cx, cy)); // 右下

    // 对角象限互换 (左上 <-> 右下)
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    // 对角象限互换 (右上 <-> 左下)
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

// 计算频域幅值谱分布 S = log(1 + |DFT|)
cv::Mat getMagnitudeSpectrum(const cv::Mat& filter2d) {
    // 1. 扩展为双通道（实部为 filter2d，虚部为 0）
    cv::Mat planes[] = { cv::Mat_<float>(filter2d), cv::Mat::zeros(filter2d.size(), CV_32F) };
    cv::Mat complexMat;
    cv::merge(planes, 2, complexMat);

    // 2. 执行 2D DFT
    cv::dft(complexMat, complexMat);

    // 3. 分离实部与虚部，计算幅值 Magnitude = sqrt(Re^2 + Im^2)
    cv::split(complexMat, planes);
    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);

    // 4. 对数缩放 log(1 + magnitude)
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);

    // 5. 零频中心化 fftshift
    cv::Mat magShifted;
    fftshift(mag, magShifted);

    // 6. 归一化到 [0, 1] 方便显示
    cv::normalize(magShifted, magShifted, 0, 1, cv::NORM_MINMAX);

    return magShifted;
}

int main() {
    const int N = 100;

    // -------------------------------------------------------------
    // 1. Mean Filter (均值滤波器 3x3)
    // -------------------------------------------------------------
    Eigen::MatrixXf mean_eigen = Eigen::MatrixXf::Zero(N, N);
    mean_eigen.block<3, 3>(1, 1).setOnes();;
    cv::Mat mean_filter;
    cv::eigen2cv(mean_eigen, mean_filter);

    // -------------------------------------------------------------
    // 2. Gaussian Filter (高斯滤波器 5x5, sigma=10)
    // -------------------------------------------------------------
    cv::Mat g1d = cv::getGaussianKernel(5, 10, CV_32F);
    cv::Mat g2d = g1d * g1d.t(); // 外积得到 2D 高斯核

    Eigen::MatrixXf g2d_eigen;
    cv::cv2eigen(g2d, g2d_eigen);
    Eigen::MatrixXf gaussian_eigen = Eigen::MatrixXf::Zero(N, N);
    gaussian_eigen.block<5, 5>(0, 0) = g2d_eigen;
    
    cv::Mat gaussian_filter;
    cv::eigen2cv(gaussian_eigen, gaussian_filter);

    // -------------------------------------------------------------
    // 3. Laplacian Filter (拉普拉斯算子 3x3)
    // -------------------------------------------------------------
    Eigen::MatrixXf lap_kernel(3, 3);
    lap_kernel <<  0,  1,  0,
                   1, -4,  1,
                   0,  1,  0;
    Eigen::MatrixXf laplacian_eigen = Eigen::MatrixXf::Zero(N, N);
    laplacian_eigen.block<3, 3>(0, 0) = lap_kernel;
    cv::Mat laplacian_filter;
    cv::eigen2cv(laplacian_eigen, laplacian_filter);

    // -------------------------------------------------------------
    // 4. Sobel X Filter (Sobel 水平梯度 3x3)
    // -------------------------------------------------------------
    Eigen::MatrixXf sobel_x_kernel(3, 3);
    sobel_x_kernel << -1,  0,  1,
                      -2,  0,  2,
                      -1,  0,  1;
    Eigen::MatrixXf sobel_x_eigen = Eigen::MatrixXf::Zero(N, N);
    sobel_x_eigen.block<3, 3>(0, 0) = sobel_x_kernel;
    cv::Mat sobel_x_filter;
    cv::eigen2cv(sobel_x_eigen, sobel_x_filter);

    // -------------------------------------------------------------
    // 5. Sobel Y Filter (Sobel 垂直梯度 3x3)
    // -------------------------------------------------------------
    Eigen::MatrixXf sobel_y_kernel(3, 3);
    sobel_y_kernel << -1, -2, -1,
                       0,  0,  0,
                       1,  2,  1;
    Eigen::MatrixXf sobel_y_eigen = Eigen::MatrixXf::Zero(N, N);
    sobel_y_eigen.block<3, 3>(0, 0) = sobel_y_kernel;
    cv::Mat sobel_y_filter;
    cv::eigen2cv(sobel_y_eigen, sobel_y_filter);

    // -------------------------------------------------------------
    // 6. Scharr X Filter (Scharr 水平梯度 3x3)
    // -------------------------------------------------------------
    Eigen::MatrixXf scharr_kernel(3, 3);
    scharr_kernel <<  -3, 0,  3,
                     -10, 0, 10,
                      -3, 0,  3;
    Eigen::MatrixXf scharr_eigen = Eigen::MatrixXf::Zero(N, N);
    scharr_eigen.block<3, 3>(1, 1) = scharr_kernel;
    cv::Mat scharr_filter;
    cv::eigen2cv(scharr_eigen, scharr_filter);

    // -------------------------------------------------------------
    // 傅里叶变换与频域响应计算
    // -------------------------------------------------------------
    std::vector<cv::Mat> filters = {
        mean_filter, gaussian_filter, laplacian_filter, 
        sobel_x_filter, sobel_y_filter, scharr_filter
    };
    std::vector<std::string> titles = {
        "Mean Filter", "Gaussian", "Laplacian", 
        "Sobel X", "Sobel Y", "Scharr X"
    };

    // 拼接成 2x3 的网格图像统一显示
    int cell_size = N * 2; // 放大像素便于观察
    cv::Mat canvas = cv::Mat::zeros(cell_size * 2, cell_size * 3, CV_32F);

    for (size_t i = 0; i < filters.size(); ++i) {
        cv::Mat mag = getMagnitudeSpectrum(filters[i]);

        // 放大图像以便清晰展示
        cv::Mat mag_resized;
        cv::resize(mag, mag_resized, cv::Size(cell_size, cell_size), 0, 0, cv::INTER_NEAREST);

        int row = i / 3;
        int col = i % 3;
        cv::Rect roi(col * cell_size, row * cell_size, cell_size, cell_size);
        mag_resized.copyTo(canvas(roi));
    }

    // 转换为 8 位图像进行显示
    cv::Mat canvas_8u;
    canvas.convertTo(canvas_8u, CV_8UC1, 255.0);

    // 在窗口中显示图像
    cv::imshow("Frequency Domain Filter Visualizations", canvas_8u);
    cv::waitKey(0);




    cv::Mat img = cv::imread("../../lena.png", cv::IMREAD_GRAYSCALE);
    cv::imshow("", img);
    cv::waitKey(0);
    cv::Mat img_float;
    img.convertTo(img_float, CV_32F);
    cv::Mat fft_img;
    cv::dft(img_float, fft_img, cv::DFT_COMPLEX_OUTPUT);
    cv::imshow("Magnitude Spectrum", fft_img);
    cv::waitKey(0);
    cv::Mat ffts_img;
    fftshift(img_float, ffts_img);
    cv::imshow("Centered Spectrum", ffts_img);
    cv::waitKey(0);
    return 0;
}