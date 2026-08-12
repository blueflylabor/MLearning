#pragma once
#include <iostream>
#include <vector>
#include <cassert>
#include <iomanip>
#include <sstream>
#include <numeric>
#include <cmath>
#include <random>

/***
mat1 = [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]]
shape: 4,3 stride: 12, 1

1  2  3  3  4  5  5  6  7  7  8  9
00 01 02 03 04 05 06 07 08 09 10 11

mat2 = [ [ [1, 2], [2, 3], [3, 4] ], [ [4, 5], [5, 6], [6, 7] ], [ [7, 8], [8, 9], [9, 10] ], [ [10, 11], [11, 12], [12, 13] ] ]
shape: 2,3,4 stride 2*3*4 / 1, 2*3*4 / 2*3, 1

1  2  2  3  3  4  4  5  5  6  6  7  7  8  8  9  9  10 10 11 11 12 12 13
00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23

at({1,1,2}) offset = 1 *12 + 1 *4 + 2 *1 = 18

***/

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <imnintrin.h>
#define HAS_AVX2
#endif

class Tensor {
public:
    std::vector<size_t> shape;      // 各维度大小，如 {2, 3, 4}
    std::vector<size_t> strides;    // 各维度步长，如 {12, 4, 1}
    std::vector<double> data;       // 一维底层存储

    // 构造函数1：花括号初始化列表，如 Tensor t({2, 3, 4})
    Tensor(std::initializer_list<size_t> dims, double init_val = 0.0) {
        shape.assign(dims);
        init_data(init_val);
    }

    // 构造函数2：vector 传参，如 Tensor t(shape_vec)
    // 【修复关键】增加这个构造函数，解决 squeeze/transpose 里的 vector 传参问题
    Tensor(const std::vector<size_t>& dims, double init_val = 0.0) {
        shape = dims;
        init_data(init_val);
    }

    // 统一初始化逻辑
    void init_data(double init_val) {
        size_t total = 1;
        for (auto d : shape) total *= d;
        data.resize(total, init_val);
        compute_strides();
    }

    // 根据 shape 计算行优先 strides
    void compute_strides() {
        strides.resize(shape.size());
        size_t stride = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    // 核心：多维索引 -> 一维偏移（高维解析为1维的关键）
    size_t offset(const std::vector<size_t>& idx) const {
        assert(idx.size() == shape.size());
        size_t flat = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            assert(idx[i] < shape[i]);
            flat += idx[i] * strides[i];
        }
        return flat;
    }

    // 访问元素：tensor.at({r, c, d})
    double& at(std::initializer_list<size_t> idx_list) {
        std::vector<size_t> idx(idx_list);
        return data[offset(idx)];
    }
    const double& at(std::initializer_list<size_t> idx_list) const {
        std::vector<size_t> idx(idx_list);
        return data[offset(idx)];
    }

    // 元素总数
    size_t numel() const { return data.size(); }

    size_t ndim() const {return shape.size();}
    // ========== 维度转换 API ==========

    // 展平为 1D
    Tensor flatten() const {
        Tensor res({numel()});
        res.data = this->data;
        return res;
    }

    // 重塑为任意新形状（initializer_list 版本）
    Tensor reshape(std::initializer_list<size_t> new_shape) const {
        size_t new_total = 1;
        for (auto d : new_shape) new_total *= d;
        assert(new_total == numel() && "Element count mismatch!");

        Tensor res(new_shape);
        res.data = this->data;
        return res;
    }

    // 重塑为任意新形状（vector 版本，内部函数调用用）
    Tensor reshape(const std::vector<size_t>& new_shape) const {
        size_t new_total = 1;
        for (auto d : new_shape) new_total *= d;
        assert(new_total == numel() && "Element count mismatch!");

        Tensor res(new_shape);  // 现在可以匹配到新增的构造函数了
        res.data = this->data;
        return res;
    }

    // 去掉所有大小为 1 的维度
    Tensor squeeze() const {
        std::vector<size_t> new_shape;
        for (auto d : shape) if (d != 1) new_shape.push_back(d);
        if (new_shape.empty()) new_shape.push_back(1);
        return reshape(new_shape);  // 现在不会报错了
    }

    // 在指定位置插入一个大小为 1 的维度
    Tensor unsqueeze(size_t dim) const {
        std::vector<size_t> new_shape = shape;
        assert(dim <= new_shape.size());
        new_shape.insert(new_shape.begin() + dim, 1);
        return reshape(new_shape);
    }

    // 转置任意两个维度
    Tensor transpose(size_t dim0, size_t dim1) const {
        assert(dim0 < shape.size() && dim1 < shape.size());
        Tensor res(shape);  // 现在可以匹配到新增的构造函数了
        res.data = this->data;
        std::swap(res.shape[dim0], res.shape[dim1]);
        std::swap(res.strides[dim0], res.strides[dim1]);
        return res;
    }

    bool allclose(const Tensor& other, double rtol = 1e-5, double atol = 1e-8) const {
        if (shape != other.shape) return false;
        for (size_t i = 0; i < numel(); ++i) {
            double diff = std::abs(data[i] - other.data[i]);
            if (diff > atol + rtol * std::abs(other.data[i])) return false;
        }
        return true;
    }
    
    
    
    void print_() const {
        std::cout << "Shape: (";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i] << (i + 1 == shape.size() ? "" : ", ");
        }
        std::cout << "), Strides: (";
        for (size_t i = 0; i < strides.size(); ++i) {
            std::cout << strides[i] << (i + 1 == strides.size() ? "" : ", ");
        }
        std::cout << ")\nData: [ ";
        for (auto v : data) std::cout << v << " ";
        std::cout << "]\n\n";
    }

    void print(const std::string& name = "") const {
        if (!name.empty()) std::cout << name << " ";
        std::cout << "tensor(";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i + 1 < shape.size()) std::cout << ", ";
        }
        std::cout << ")";
        if (shape.empty()) {
            std::cout << " = " << (data.empty() ? 0 : data[0]) << std::endl;
            return;
        }
        std::cout << std::endl;
        size_t max_width = compute_max_width();
        std::string indent = " ";
        std::vector<size_t> idx(shape.size(), 0);
        print_recursive(0, idx, indent, max_width);
        std::cout << std::endl;
    }

    Tensor operator+(const Tensor& t) const {
        assert(same_shape(t) && "Shape mismatch for /");
        Tensor res(shape);
        for (size_t i = 0; i < numel(); ++i) {
            res.data[i] = data[i] + t.data[i];
        }
        return res;
    }

    Tensor operator-(const Tensor& t) const {
        assert(same_shape(t) && "Shape mismatch for /");
        Tensor res(shape);
        for (size_t i = 0; i < numel(); ++i) {
            res.data[i] = data[i] - t.data[i];
        }
        return res;
    }

    Tensor operator*(const Tensor& t) const {
        assert(same_shape(t) && "Shape mismatch for /");
        Tensor res(shape);
        for (size_t i = 0; i < numel(); ++i) {
            res.data[i] = data[i] * t.data[i];
        }
        return res;
    }

    Tensor operator/(const Tensor& t) const {
        assert(same_shape(t) && "Shape mismatch for /");
        Tensor res(shape);
        for (size_t i = 0; i < numel(); ++i) {
            res.data[i] = data[i] / t.data[i];
        }
        return res;
    }
    
    static Tensor ones(std::initializer_list<size_t> dims){
        return Tensor(dims, 1.);
    }
    
    static Tensor zeros(std::initializer_list<size_t> dims){
        return Tensor(dims, 0.);
    }
    
    void fill(double val){
        for (auto& v : data) v = val;
    }

    static Tensor randn(std::initializer_list<size_t> dims, double mean = 0.0, double stddev = 1.0, unsigned int seed = 42) {
        static std::mt19937 gen(seed);
        std::normal_distribution<double> dist(mean, stddev);
        Tensor res(dims);
        for (size_t i = 0; i < res.numel(); ++i) {
            res.data[i] = dist(gen);
        }
        return res;
    }
    
    Tensor matmul_naive(const Tensor&B) const{
        assert(ndim() == 2 && B.ndim() == 2);
        assert(shape[1] == B.shape[0]);
        size_t M = shape[0], K = shape[1], N = B.shape[1];
        Tensor C({M, N}, 0.0);
        for (size_t i = 0; i < M; i++) {
            for(size_t j = 0; j < N; j++){
                double sum = .0;
                for(size_t k = 0; k < K; k++)
                    sum += data[i * K + k] * B.data[k * N + j];
                C.data[i * N + j] = sum;
            }
        }
        return C;
    }
    Tensor matmul_1X4(const Tensor &B) const{
        assert(ndim() == 2 && B.ndim() == 2);
        assert(shape[1] == B.shape[0]);
        size_t M = shape[0], K = shape[1], N = B.shape[1];
        Tensor C({M, N}, 0.0);
        for(size_t i = 0; i < M; i++)
            for(size_t j = 0; j < N; j+=4){
                size_t j_end = std::min(j + 4, N);
                for(size_t k = 0; k < K; ++k){
                    double a = data[i * K + k];
                    for(size_t jj = j; jj < j_end; ++jj){
                        C.data[i * N + jj] += a * B.data[k * N + jj];
                    }
                }
            }
        return C;
    }

    Tensor matmul_1x4_reg(const Tensor& B) const {
        assert(ndim() == 2 && B.ndim() == 2);
        assert(shape[1] == B.shape[0]);
        size_t M = shape[0], K = shape[1], N = B.shape[1];
        Tensor C({M, N}, 0.0);

        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; j += 4) {
                size_t j_end = std::min(j + 4, N);
                // 初始化 C 累加器（寄存器）
                double c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                
                for (size_t k = 0; k < K; ++k) {
                    double a_reg = data[i * K + k]; // 加载到寄存器，复用 4 次
                    if (j_end - j > 0) c0 += a_reg * B.data[k * N + j + 0];
                    if (j_end - j > 1) c1 += a_reg * B.data[k * N + j + 1];
                    if (j_end - j > 2) c2 += a_reg * B.data[k * N + j + 2];
                    if (j_end - j > 3) c3 += a_reg * B.data[k * N + j + 3];
                }
                // 写回内存
                if (j_end - j > 0) C.data[i * N + j + 0] = c0;
                if (j_end - j > 1) C.data[i * N + j + 1] = c1;
                if (j_end - j > 2) C.data[i * N + j + 2] = c2;
                if (j_end - j > 3) C.data[i * N + j + 3] = c3;
            }
        }
        return C;
    }

private:
    size_t compute_max_width() const {
        size_t max_w = 1;
        for (double v : data) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4) << v;
            max_w = std::max(max_w,oss.str().length());
        }
        return max_w + 1;
    }

    void print_recursive(size_t dim, std::vector<size_t>& idx, const std::string& indent, size_t cell_width) const {
        std::cout << std::string(dim, ' ') << "[";
        if (dim == shape.size() -1) {
            for (size_t i = 0; i < shape[dim]; ++i) {
                idx[dim] = i;
                double val = data[offset((idx))];
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(4) << val;
                std::string s = oss.str();
                std::cout << std::string(cell_width - s.length(), ' ') << s;
                if (i + 1 < shape[dim]) std::cout << ",";
            }
            std::cout << "]";
        }else {
            std::cout << std::endl;
            for (size_t i = 0; i < shape[dim]; ++i) {
                idx[dim] = i;
                print_recursive(dim+1, idx, indent, cell_width);
                if (i + 1 < shape[dim]) std::cout << "," <<std::endl;
            }
            std::cout << std::endl << std::string(dim, ' ') << "]";
        }
    }

    bool same_shape(const Tensor& other) const {
        if (shape.size() != other.shape.size()) return false;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] != other.shape[i]) return false;
        }
        return true;
    }
};


class Timer {
    std::chrono::high_resolution_clock::time_point start;
    std::string name;
public:
    Timer(const std::string& n = "") : name(n) {
        start = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[Timer] " << name << ": " << std::fixed << std::setprecision(3)
                  << ms << " ms\n";
    }
};
