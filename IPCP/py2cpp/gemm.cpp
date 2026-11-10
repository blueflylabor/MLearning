#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <random>
#include <Eigen/Dense>

// 高性能矩阵乘法实现：内含 RAII 管理与 Cache 优化
class OptimizedGEMM {
public:
    // 普通三重循环 (Cache 不友好)
    static void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    // B[k * N + j] 是按列访问，跳跃步长为 N，Cache 命中率低
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }

    // 优化三重循环 (转置 B 矩阵 -> 全连续访问 -> Cache 极度友好)
    static void optimized_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
        // 1. 使用 RAII 申请临时空间保存 B 的转置矩阵 B_T (尺寸 K x N -> N x K)
        auto B_T = std::make_unique<float[]>(N * K);

        // 将 B 转置为 B_T: B_T[j, k] = B[k, j]
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                B_T[j * K + k] = B[k * N + j];
            }
        }

        // 2. 核心GEMM：A 的第 i 行与 B_T 的第 j 行做内积
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                const float* ptr_A = &A[i * K];
                const float* ptr_B_T = &B_T[j * K];

                // 最内层循环：ptr_A 与 ptr_B_T 均以 1 为步长连续自增！
                // 触发 CPU L1 Cache 预取与自动向量化 (SIMD)
                for (int k = 0; k < K; ++k) {
                    sum += ptr_A[k] * ptr_B_T[k];
                }
                C[i * N + j] = sum;
            }
        }
    }
};

int main() {
    int M = 1024, N = 1024, K = 1024;

    // RAII 分配矩阵内存
    auto A = std::make_unique<float[]>(M * K);
    auto B = std::make_unique<float[]>(K * N);
    auto C1 = std::make_unique<float[]>(M * N);
    auto C2 = std::make_unique<float[]>(M * N);

    // 填充随机数据
    for (int i = 0; i < M * K; ++i) A[i] = 0.01f * (i % 100);
    for (int i = 0; i < K * N; ++i) B[i] = 0.01f * (i % 100);

    std::cout << "正在运行 GEMM 矩阵乘法基准测试 (" << M << "x" << K << " * " << K << "x" << N << ")...\n";
    
    // 1. 测试原始粗糙 GEMM
    auto t1 = std::chrono::high_resolution_clock::now();
    OptimizedGEMM::naive_gemm(A.get(), B.get(), C1.get(), M, N, K);
    auto t2 = std::chrono::high_resolution_clock::now();
    double time_naive = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // 2. 测试 Cache 优化 GEMM
    auto t3 = std::chrono::high_resolution_clock::now();
    OptimizedGEMM::optimized_gemm(A.get(), B.get(), C2.get(), M, N, K);
    auto t4 = std::chrono::high_resolution_clock::now();
    double time_opt = std::chrono::duration<double, std::milli>(t4 - t3).count();

    std::cout << "1. 原始 GEMM 耗时: " << time_naive << " ms\n";
    std::cout << "2. Cache 优化后 GEMM 耗时: " << time_opt << " ms\n";
    std::cout << "性能提升比例: " << time_naive / time_opt << " 倍！\n";
    return 0;
}