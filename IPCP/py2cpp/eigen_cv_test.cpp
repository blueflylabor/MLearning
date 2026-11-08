#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// 1. im2col 转换函数：把 3D 输入 (C_in, H, W) 转化为二维矩阵 (C_in * Kh * Kw, H_out * W_out)
Eigen::MatrixXf im2col(const std::vector<Eigen::MatrixXf>& input_channels, 
                       int K_H, int K_W, int stride, int padding) {
    int C_in = input_channels.size();
    int H = input_channels[0].rows();
    int W = input_channels[0].cols();

    int H_out = (H + 2 * padding - K_H) / stride + 1;
    int W_out = (W + 2 * padding - K_W) / stride + 1;

    // 对每个输入通道进行 Padding 扩展
    std::vector<Eigen::MatrixXf> padded_channels(C_in);
    for (int c = 0; c < C_in; ++c) {
        if (padding > 0) {
            padded_channels[c] = Eigen::MatrixXf::Zero(H + 2 * padding, W + 2 * padding);
            padded_channels[c].block(padding, padding, H, W) = input_channels[c];
        } else {
            padded_channels[c] = input_channels[c];
        }
    }

    // 初始化 X_col 矩阵
    int rows_col = C_in * K_H * K_W;
    int cols_col = H_out * W_out;
    Eigen::MatrixXf X_col(rows_col, cols_col);

    int col_idx = 0;
    // 遍历每一个滑动窗口
    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            int h_start = h * stride;
            int w_start = w * stride;

            int row_idx = 0;
            // 提取滑动窗口数据按列填入 X_col
            for (int c = 0; c < C_in; ++c) {
                for (int kh = 0; kh < K_H; ++kh) {
                    for (int kw = 0; kw < K_W; ++kw) {
                        X_col(row_idx++, col_idx) = padded_channels[c](h_start + kh, w_start + kw);
                    }
                }
            }
            col_idx++;
        }
    }
    return X_col;
}

// 2. 工业级 Conv2D 算子实现 (im2col + Eigen GEMM)
class Conv2DEigen {
public:
    int in_channels;
    int out_channels;
    int K_H, K_W;
    int stride;
    int padding;

    // 权重矩阵 W_mat: Shape (out_channels, in_channels * K_H * K_W)
    Eigen::MatrixXf W_mat;
    // 偏置向量 Bias: Shape (out_channels, 1)
    Eigen::VectorXf bias;

    Conv2DEigen(int in_c, int out_c, int k_h, int k_w, int s = 1, int p = 0)
        : in_channels(in_c), out_channels(out_c), K_H(k_h), K_W(k_w), stride(s), padding(p) {
        
        // 随机初始化权重矩阵与偏置
        W_mat = Eigen::MatrixXf::Random(out_c, in_c * k_h * k_w) * 0.1f;
        bias = Eigen::VectorXf::Zero(out_c);
    }

    // 前向传播
    std::vector<Eigen::MatrixXf> forward(const std::vector<Eigen::MatrixXf>& input_channels) {
        int H = input_channels[0].rows();
        int W = input_channels[0].cols();
        int H_out = (H + 2 * padding - K_H) / stride + 1;
        int W_out = (W + 2 * padding - K_W) / stride + 1;

        // Step 1: 执行 im2col 展开输入
        Eigen::MatrixXf X_col = im2col(input_channels, K_H, K_W, stride, padding);

        // Step 2: 矩阵乘法 GEMM (W_mat * X_col) 加上偏置广播
        // Eigen 的矩阵乘法会自动触发 SIMD 向量化加速（AVX2/NEON）
        Eigen::MatrixXf Y_mat = (W_mat * X_col).colwise() + bias;

        // Step 3: Reshape 回多通道二维矩阵输出
        std::vector<Eigen::MatrixXf> output_channels(out_channels, Eigen::MatrixXf(H_out, W_out));
        for (int c = 0; c < out_channels; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    output_channels[c](h, w) = Y_mat(c, h * W_out + w);
                }
            }
        }
        return output_channels;
    }
};

// 3. 基于 Eigen3 的矩阵级别 ReLU 算子
class ReLUEigen {
public:
    std::vector<Eigen::MatrixXf> forward(const std::vector<Eigen::MatrixXf>& input) {
        std::vector<Eigen::MatrixXf> output = input;
        for (auto& mat : output) {
            // 利用 Eigen 的系数运算函数 (Coefficient-wise max)
            mat = mat.cwiseMax(0.0f);
        }
        return output;
    }
};

int main() {
    std::cout << "=== 阶段三：基于 OpenCV 数据加载与 Eigen3 矩阵乘法的工业级 Conv2D ===" << std::endl;

    // 1. 使用 OpenCV 加载真实图像或构建假数据图像
    //cv::Mat image = cv::Mat::ones(100, 100, CV_8UC3); // 100x100 RGB 图像
    cv::Mat image = cv::imread("../../tree.jpg");
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    // 2. OpenCV Image -> Eigen Matrix 转换 (预处理: 归一化到 [0, 1])
    std::vector<cv::Mat> cv_channels;
    cv::split(image, cv_channels);
    
    std::vector<Eigen::MatrixXf> eigen_inputs(3);
    for (int c = 0; c < 3; ++c) {
        eigen_inputs[c] = Eigen::MatrixXf(image.rows, image.cols);
        for (int r = 0; r < image.rows; ++r) {
            for (int col = 0; col < image.cols; ++col) {
                eigen_inputs[c](r, col) = cv_channels[c].at<uchar>(r, col) / 255.0f;
            }
        }
    }

    // 3. 构建 3 入 8 出、3x3 卷积核的 Conv2D + ReLU 组合模块
    Conv2DEigen conv(3, 8, 3, 3, 1, 1);
    ReLUEigen relu;

    // 4. 前向传播计算
    std::vector<Eigen::MatrixXf> conv_out = conv.forward(eigen_inputs);
    std::vector<Eigen::MatrixXf> relu_out = relu.forward(conv_out);

    std::cout << "输入图像尺寸: (" << image.rows << ", " << image.cols << ", " << image.channels() << ")" << std::endl;
    std::cout << "Conv2D + ReLU 输出通道数: " << relu_out.size() << std::endl;
    std::cout << "输出单个特征图尺寸: (" << relu_out[0].rows() << ", " << relu_out[0].cols() << ")" << std::endl;
    std::cout << "Channel 0 左上角 3x3 输出数值:\n" << relu_out[0].block(0, 0, 3, 3) << std::endl;

    return 0;
}