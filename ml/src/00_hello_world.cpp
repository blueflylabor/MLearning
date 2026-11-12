#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>
#include <Eigen/Dense>

int main() {
    std::string test_string = "Hello World!";
    std::cout << test_string << std::endl;

    // Quick sanity check to verify Eigen is working
    Eigen::Vector3d vec(1.0, 2.0, 3.0);
    std::cout << "Eigen Test Vector:\n" << vec << std::endl;
    
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "====================================================================\n";
    std::cout << "   Eigen generate synthetic data\n";
    std::cout << "====================================================================\n\n";

    // =========================================================================
    // 第一步：构造伪造数据集 (Synthetic Data Generation)
    // 场景：生成 100 条样本，包含 3 个特征 + 1 个目标值 y (共 100x4 矩阵)
    // =========================================================================
    std::cout << ">>> [1/7] 生成伪造数据集 (100x4)...\n";
    const Eigen::Index numSamples = 100;
    const Eigen::Index numFeatures = 3;

    // 1.1 使用 Eigen 内置随机分布与等差数列生成特征
    Eigen::VectorXd f0 = Eigen::VectorXd::LinSpaced(numSamples, 10.0, 50.0);     // 特征0: 线性增长 [10, 50]
    Eigen::VectorXd f1 = (Eigen::VectorXd::Random(numSamples).array() + 1.0) * 5.0; // 特征1: 均匀分布 [0, 10]
    Eigen::VectorXd f2 = Eigen::VectorXd::Constant(numSamples, 2.0) + Eigen::VectorXd::Random(numSamples); // 特征2: 均值2波动

    std::cout << "f0[0:5] : ";
    std::cout << f0.topRows(std::min<Eigen::Index>(5, f0.rows())).transpose() << "\n\n";
    std::cout << "f1[0:5] : ";
    std::cout << f1.topRows(std::min<Eigen::Index>(5, f1.rows())).transpose() << "\n\n";
    std::cout << "f2[0:5] : ";
    std::cout << f2.topRows(std::min<Eigen::Index>(5, f2.rows())).transpose() << "\n\n";

    // 1.2 设定真实权重与噪声，生成目标标签 y = 1.5*f0 - 2.0*f1 + 3.0*f2 + 10.0 + noise
    Eigen::VectorXd trueWeights(3);
    trueWeights << 1.5, -2.0, 3.0;
    double trueBias = 10.0;

        // 水平拼接特征矩阵 X (100x3)
    Eigen::MatrixXd rawX(numSamples, numFeatures);
    rawX << f0, f1, f2;

    Eigen::VectorXd noise = Eigen::VectorXd::Random(numSamples) * 0.5;
    Eigen::VectorXd rawY = (rawX * trueWeights).array() + trueBias + noise.array();

    // 1.3 合并为完整数据集 dataset (100x4)，并故意注入几个异常值/脏数据
    Eigen::MatrixXd dataset(numSamples, numFeatures + 1);
    dataset << rawX, rawY;

    dataset(0, 1) = -999.0;  // 注入异常负值 (模拟缺失值占位符)
    dataset(5, 2) = 888.0;   // 注入极大异常值 (Outlier)
    dataset(10, 0) = -999.0; // 注入异常负值

    std::cout << "数据集维度: " << dataset.rows() << " 行 x " << dataset.cols() << " 列 (总元素: " << dataset.size() << ")\n";
    std::cout << "前 3 行原始数据 (注意第0行有-999脏数据):\n" << dataset.topRows(3) << "\n\n";

    // =========================================================================
    // 第二步：切片分离特征与标签 (Slicing & Block Operations)
    // =========================================================================
    std::cout << ">>> [2/7] 切片分离特征 X 与目标 y...\n";
    Eigen::MatrixXd X = dataset.leftCols(numFeatures);
    Eigen::VectorXd y = dataset.rightCols(1);
    Eigen::MatrixXd subBlock = X.block(2, 0, 3, 2);    // 从 (2,0) 提取 3x2 子矩阵
    Eigen::MatrixXd stridedCols(X.rows(), (X.cols() + 1) / 2);
    for (Eigen::Index col = 0; col < X.cols(); col += 2) {
        stridedCols.col(col / 2) = X.col(col);
    } // 步长切片：只取第 0 列和第 2 列

    std::cout << "Feature X dimension: " << X.rows() << "x" << X.cols() << ", stride slice dimension: " << stridedCols.rows() << "x" << stridedCols.cols() << "\n\n";

    long missingCount = (X.array() == -999.0).count();
    Eigen::Index maxR, maxC;
    double maxVal = X.maxCoeff(&maxR, &maxC);
    std::cout << "detect missing count -999: " << missingCount << "\n";
    std::cout << "the maximum before cleaning data: " << maxVal << " at (" << maxR << ", " << maxC << ")\n";

    X = (X.array() == -999.0).select(0.0, X);
    X = X.array().max(0.0).min(50.0);
    std::cout << "After cleaning Max: " << X.maxCoeff() << " Min: " << X.minCoeff();

    return 0;
}
