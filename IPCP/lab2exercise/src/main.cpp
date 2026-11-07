#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

void first_cv_test()
{
    cv::Mat img = cv::Mat::zeros(500, 500, CV_8UC3);

    cv::putText(img, "Wue", cv::Point(100, 250),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

    cv::imshow("Test Window", img);
    cv::waitKey(0);
}

// 通用 2D 滑动窗口卷积实现 (手动遍历,不调用 cv::filter2D)
cv::Mat manualConv2D(const cv::Mat &src_gray, const cv::Mat &kernel)
{
    // 确保输入为双精度浮点类型以保证精度
    cv::Mat src;
    src_gray.convertTo(src, CV_64F);

    int rows = src.rows;
    int cols = src.cols;
    int kRows = kernel.rows;
    int kCols = kernel.cols;
    int padR = kRows / 2;
    int padW = kCols / 2;

    // 1. 图像边界填充 (Edge Replicate)
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, padR, padR, padW, padW, cv::BORDER_REPLICATE);

    // 2. 初始化输出矩阵
    cv::Mat dst = cv::Mat::zeros(rows, cols, CV_64F);

    // 3. 滑动窗口遍历 (四重循环)
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            double pixel_sum = 0.0;
            for (int kr = 0; kr < kRows; ++kr)
            {
                for (int kc = 0; kc < kCols; ++kc)
                {
                    double img_val = padded.at<double>(r + kr, c + kc);
                    double kernel_val = kernel.at<double>(kr, kc);
                    pixel_sum += img_val * kernel_val;
                }
            }
            dst.at<double>(r, c) = pixel_sum;
        }
    }
    return dst;
}

// 1. 模糊核 (5x5 高斯模糊)
cv::Mat getGaussianKernel(int ksize = 5, double sigma = 1.0) {
    cv::Mat kernel(ksize, ksize, CV_64F);
    int center = ksize / 2;
    double sum = 0.0;
    for (int i = 0; i < ksize; ++i) {
        for (int j = 0; j < ksize; ++j) {
            double x = i - center;
            double y = j - center;
            double val = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel.at<double>(i, j) = val;
            sum += val;
        }
    }
    return kernel / sum; // 归一化
}

// 2. 图像梯度核 (N4 / N8 邻域)
cv::Mat getN4GradientKernelX() {
    // 3x3 矩阵,仅考虑 N4 水平邻域
    return (cv::Mat_<double>(3, 3) << 
        0,  0, 0,
       -1,  0, 1,
        0,  0, 0);
}

cv::Mat getN8GradientKernelX() {
    // 3x3 矩阵,考虑到角线的 N8 水平平滑梯度
    return (cv::Mat_<double>(3, 3) << 
       -1,  0, 1,
       -1,  0, 1,
       -1,  0, 1) / 3.0;
}

// 3. 边缘检测核 (Sobel X 算子)
cv::Mat getSobelKernelX() {
    return (cv::Mat_<double>(3, 3) << 
       -1, 0, 1,
       -2, 0, 2,
       -1, 0, 1);
}

// 4. 角点检测：基于结构张量的 Harris 响应图生成
cv::Mat computeHarrisCornerResponse(const cv::Mat& src_gray, double k = 0.04) {
    cv::Mat kx = getSobelKernelX();
    cv::Mat ky = kx.t(); // 垂直梯度为转置矩阵

    cv::Mat Ix = manualConv2D(src_gray, kx);
    cv::Mat Iy = manualConv2D(src_gray, ky);

    cv::Mat Ix2, Iy2, Ixy;
    cv::multiply(Ix, Ix, Ix2);
    cv::multiply(Iy, Iy, Iy2);
    cv::multiply(Ix, Iy, Ixy);

    // 使用高斯模糊做局部积分窗口
    cv::Mat gKernel = getGaussianKernel(5, 1.0);
    cv::Mat Sx2 = manualConv2D(Ix2, gKernel);
    cv::Mat Sy2 = manualConv2D(Iy2, gKernel);
    cv::Mat Sxy = manualConv2D(Ixy, gKernel);

    // 计算响应函数 R = det(M) - k * (trace(M))^2
    cv::Mat R = cv::Mat::zeros(src_gray.size(), CV_64F);
    for (int r = 0; r < src_gray.rows; ++r) {
        for (int c = 0; c < src_gray.cols; ++c) {
            double sx2 = Sx2.at<double>(r, c);
            double sy2 = Sy2.at<double>(r, c);
            double sxy = Sxy.at<double>(r, c);

            double det = (sx2 * sy2) - (sxy * sxy);
            double trace = sx2 + sy2;
            R.at<double>(r, c) = det - k * (trace * trace);
        }
    }
    return R;
}

// 5. 高斯拉普拉斯核 (LoG Filter)
cv::Mat getLoGKernel(int ksize = 9, double sigma = 1.4) {
    cv::Mat kernel(ksize, ksize, CV_64F);
    int center = ksize / 2;
    double sum = 0.0;

    for (int r = 0; r < ksize; ++r) {
        for (int c = 0; c < ksize; ++c) {
            double x = c - center;
            double y = r - center;
            double norm = (x * x + y * y) / (2 * sigma * sigma);
            double val = -1.0 / (M_PI * std::pow(sigma, 4)) * (1.0 - norm) * std::exp(-norm);
            kernel.at<double>(r, c) = val;
            sum += val;
        }
    }
    // 扣除均值,强制保证核的权重之和为 0
    return kernel - (sum / (ksize * ksize));
}

// 6. Gabor 滤波核 generator
cv::Mat getGaborKernel(int ksize, double sigma, double theta, double lambda, double gamma, double psi) {
    cv::Mat kernel(ksize, ksize, CV_64F);
    int center = ksize / 2;
    for (int r = 0; r < ksize; ++r) {
        for (int c = 0; c < ksize; ++c) {
            double x = c - center;
            double y = r - center;
            double x_theta = x * std::cos(theta) + y * std::sin(theta);
            double y_theta = -x * std::sin(theta) + y * std::cos(theta);

            double val = std::exp(-0.5 * (x_theta * x_theta + gamma * gamma * y_theta * y_theta) / (sigma * sigma))
                       * std::cos(2 * M_PI * x_theta / lambda + psi);
            kernel.at<double>(r, c) = val;
        }
    }
    return kernel;
}

// 将 double 类型的矩阵/核归一化并放大显示
void displayVisualization(const std::string& winName, const cv::Mat& mat, bool isKernel = false) {
    cv::Mat normMat, ucharMat;
    // 将极小值和极大值安全地拉伸到 [0, 255]
    cv::normalize(mat, normMat, 0, 255, cv::NORM_MINMAX);
    normMat.convertTo(ucharMat, CV_8U);

    if (isKernel) {
        // 核矩阵尺寸太小（如 3x3）,使用 Nearest 插值放大到 180x180 像素方便观察
        cv::Mat resizedKernel;
        cv::resize(ucharMat, resizedKernel, cv::Size(180, 180), 0, 0, cv::INTER_NEAREST);

        // 给核加上伪彩色映射 (Coolwarm 风格：JET 映射),正负权重色彩鲜明
        cv::Mat colorMapKernel;
        cv::applyColorMap(resizedKernel, colorMapKernel, cv::COLORMAP_JET);
        cv::imshow(winName + " (Weights Plot)", colorMapKernel);
    } else {
        cv::imshow(winName + " (Result)", ucharMat);
    }
}

int main() {
    // 1. 读取测试图像并转换为灰度图
    cv::Mat img = cv::imread("../uw.jpeg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cout << "错误：未找到测试图片 input.jpg,生成随机合成图片测试!" << std::endl;
        img = cv::Mat::zeros(300, 300, CV_8U);
        cv::rectangle(img, cv::Rect(50, 50, 100, 100), cv::Scalar(255), -1);
        cv::circle(img, cv::Point(200, 200), 50, cv::Scalar(180), -1);
    }

    // --- Task 1: Blurring Filter ---
    cv::Mat gaussKernel = getGaussianKernel(7, 1.5);
    cv::Mat blurResult = manualConv2D(img, gaussKernel);
    displayVisualization("1. Gaussian Blur Kernel", gaussKernel, true);
    displayVisualization("1. Gaussian Blur", blurResult);
    cv::waitKey(0);
    // --- Task 2: Image Gradients (N4 vs N8) ---
    cv::Mat n4Kernel = getN4GradientKernelX();
    cv::Mat n8Kernel = getN8GradientKernelX();
    cv::Mat n4Result = manualConv2D(img, n4Kernel);
    cv::Mat n8Result = manualConv2D(img, n8Kernel);
    displayVisualization("2. Gradient N4 Kernel", n4Kernel, true);
    displayVisualization("2. Gradient N4 (Horizontal)", n4Result);
    displayVisualization("2. Gradient N8 (Horizontal)", n8Result);
    cv::waitKey(0);
    // --- Task 3: Edge Detection (Sobel) ---
    cv::Mat sobelKernel = getSobelKernelX();
    cv::Mat edgeResult = manualConv2D(img, sobelKernel);
    displayVisualization("3. Sobel Edge Kernel", sobelKernel, true);
    displayVisualization("3. Sobel Edge Detection", edgeResult);
    cv::waitKey(0);
    // --- Task 4: Corner Detection (Harris Response) ---
    cv::Mat harrisResult = computeHarrisCornerResponse(img, 0.04);
    displayVisualization("4. Harris Corner Response", harrisResult);
    cv::waitKey(0);
    // --- Task 5: Laplacian of Gaussian (LoG) ---
    cv::Mat logKernel = getLoGKernel(9, 1.4);
    cv::Mat logResult = manualConv2D(img, logKernel);
    displayVisualization("5. LoG Kernel (Mexican Hat)", logKernel, true);
    displayVisualization("5. LoG Blob Detection", logResult);
    cv::waitKey(0);
    // --- Task 6: Gabor Filter Bank (4 方向) ---
    std::vector<double> angles = {0, M_PI / 4.0, M_PI / 2.0, 3 * M_PI / 4.0};
    for (size_t i = 0; i < angles.size(); ++i) {
        cv::Mat gaborKernel = getGaborKernel(15, 3.0, angles[i], 10.0, 0.5, 0);
        cv::Mat gaborResult = manualConv2D(img, gaborKernel);
        std::string name = "6. Gabor Angle " + std::to_string(int(angles[i] * 180 / M_PI));
        displayVisualization(name + " Kernel", gaborKernel, true);
        displayVisualization(name, gaborResult);
    }
    std::cout << "全部处理完成!按任意键退出程序..." << std::endl;
    cv::waitKey(0);
    return 0;
}
