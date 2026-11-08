#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

// 1. 自定义 NCHW 4D 张量结构体
struct Tensor4D {
    int N, C, H, W;
    std::vector<float> data;

    // 构造函数：初始化维度并分配连续内存
    Tensor4D(int n, int c, int h, int w, float init_val = 0.0f)
        : N(n), C(c), H(h), W(w), data(n * c * h * w, init_val) {}

    // 物理寻址函数：将 (n, c, h, w) 映射到一维数组索引
    inline size_t get_index(int n, int c, int h, int w) const {
        return n * (C * H * W) + c * (H * W) + h * W + w;
    }

    // 读写重载 () 运算符
    inline float& operator()(int n, int c, int h, int w) {
        return data[get_index(n, c, h, w)];
    }

    // 只读重载 () 运算符
    inline float operator()(int n, int c, int h, int w) const {
        return data[get_index(n, c, h, w)];
    }
};

// 2. 纯 C++ 裸手写 Conv2D 算子
class Conv2DRaw {
public:
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;

    Tensor4D weights; // Shape: (out_channels, in_channels, kernel_size, kernel_size)
    std::vector<float> bias; // Shape: (out_channels,)

    Conv2DRaw(int in_c, int out_c, int k_size, int s = 1, int p = 0)
        : in_channels(in_c), out_channels(out_c), kernel_size(k_size), stride(s), padding(p),
          weights(out_c, in_c, k_size, k_size, 0.0f), bias(out_c, 0.0f) {}

    // 前向传播函数
    Tensor4D forward(const Tensor4D& x) {
        int N = x.N;
        int C_in = x.C;
        int H = x.H;
        int W = x.W;

        int K = kernel_size;
        int S = stride;
        int P = padding;

        // 1. 计算输出特征图尺寸
        int H_out = (H + 2 * P - K) / S + 1;
        int W_out = (W + 2 * P - K) / S + 1;

        // 2. 图像 Padding 填充处理
        Tensor4D x_padded = x;
        if (P > 0) {
            x_padded = Tensor4D(N, C_in, H + 2 * P, W + 2 * P, 0.0f);
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C_in; ++c) {
                    for (int h = 0; h < H; ++h) {
                        for (int w = 0; w < W; ++w) {
                            x_padded(n, c, h + P, w + P) = x(n, c, h, w);
                        }
                    }
                }
            }
        }

        // 3. 初始化输出张量
        Tensor4D out(N, out_channels, H_out, W_out, 0.0f);

        // 4. 六重循环卷积核心逻辑
        for (int n = 0; n < N; ++n) {
            for (int c_out = 0; c_out < out_channels; ++c_out) {
                for (int h = 0; h < H_out; ++h) {
                    for (int w = 0; w < W_out; ++w) {
                        
                        int h_start = h * S;
                        int w_start = w * S;
                        float pixel_sum = 0.0f;

                        for (int c_in = 0; c_in < C_in; ++c_in) {
                            for (int kh = 0; kh < K; ++kh) {
                                for (int kw = 0; kw < K; ++kw) {
                                    float val = x_padded(n, c_in, h_start + kh, w_start + kw);
                                    float w_val = weights(c_out, c_in, kh, kw);
                                    pixel_sum += val * w_val;
                                }
                            }
                        }

                        // 加上偏置项
                        out(n, c_out, h, w) = pixel_sum + bias[c_out];
                    }
                }
            }
        }
        return out;
    }
};

// 3. 纯 C++ 裸手写 ReLU 算子
class ReLURaw {
public:
    Tensor4D forward(const Tensor4D& x) {
        Tensor4D out = x;
        // 由于内存连续，直接遍历底层 vector 速度最快
        for (size_t i = 0; i < out.data.size(); ++i) {
            out.data[i] = std::max(0.0f, out.data[i]);
        }
        return out;
    }
};

int main() {
    // 构造输入：Batch=1, Channels=1, Height=5, Width=5
    Tensor4D x_input(1, 1, 5, 5, 0.0f);
    
    // 给输入填入一些伪测试数据 (含正数和负数)
    float test_val = -1.2f;
    for (int h = 0; h < 5; ++h) {
        for (int w = 0; w < 5; ++w) {
            x_input(0, 0, h, w) = test_val;
            test_val += 0.15f;
        }
    }

    std::cout << "=== 输入张量 X Shape: (" 
              << x_input.N << ", " << x_input.C << ", " << x_input.H << ", " << x_input.W << ") ===\n";

    // 初始化模块: 输入 1 通道, 输出 2 通道, 3x3 卷积核, Stride=1, Padding=1
    Conv2DRaw conv(1, 2, 3, 1, 1);
    
    // 给卷积核赋指定测试权重
    for (int kh = 0; kh < 3; ++kh) {
        for (int kw = 0; kw < 3; ++kw) {
            conv.weights(0, 0, kh, kw) = 0.1f * (kh + kw);
            conv.weights(1, 0, kh, kw) = -0.05f * (kh * kw + 1);
        }
    }
    conv.bias[0] = 0.5f;
    conv.bias[1] = -0.2f;

    ReLURaw relu;

    // 前向传播计算
    Tensor4D conv_out = conv.forward(x_input);
    Tensor4D relu_out = relu.forward(conv_out);

    std::cout << "\n=== Conv2D 输出张量 Shape: (" 
              << conv_out.N << ", " << conv_out.C << ", " << conv_out.H << ", " << conv_out.W << ") ===\n";

    std::cout << "\n=== ReLU 输出数值结果 (Channel 0 前 3x3 区域) ===\n";
    std::cout << std::fixed << std::setprecision(4);
    for (int h = 0; h < 3; ++h) {
        for (int w = 0; w < 3; ++w) {
            std::cout << relu_out(0, 0, h, w) << "  ";
        }
        std::cout << "\n";
    }

    return 0;
}