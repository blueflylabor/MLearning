# Lesson 1: Transitioning Machine Learning from Python to C++ (CMake & Eigen Setup)

## 1. Overview & Objectives
Welcome to the first lesson in our series on implementing Machine Learning algorithms from scratch in C++. In typical Python-based ML workflows (such as Jupyter Notebooks using `NumPy` and `Pandas`), high-level abstractions hide memory management and linear algebra optimizations. 

In this lesson, we will translate the foundational concepts of a standard introductory ML notebook—**environment setup, array/matrix initialization, element-wise operations, and linear algebra fundamentals**—into modern C++ (C++17) using **CMake** and the **Eigen** library.

### Learning Objectives:
* Configure a cross-platform C++ build system using `CMake`.
* Integrate the **Eigen** linear algebra library (the C++ equivalent of `NumPy`).
* Translate basic `NumPy` array operations, broadcasting, and matrix multiplications into Eigen syntax.
* Implement a basic CSV data loader using only the C++ Standard Library (`<fstream>`, `<sstream>`, `<vector>`).

---

## 2. Project Structure & Requirements

### System Requirements
* **C++ Compiler:** GCC 9+, Clang 10+, or MSVC 2019+ (supporting C++17)
* **Build System:** CMake 3.14+
* **Linear Algebra Library:** Eigen 3.4+

### Directory Layout
Create the following directory structure for Lesson 1:

```text
lesson_01_intro/
├── CMakeLists.txt
├── data/
│   └── sample_data.csv
└── src/
    └── main.cpp
```

---

## 3. Setting Up CMake (`CMakeLists.txt`)

In Python, you manage dependencies with `pip` or `conda`. In C++, we use `CMakeLists.txt` to locate libraries and compile our executable. Eigen is a **header-only** library, making it remarkably fast to integrate.

Create `CMakeLists.txt` in your root folder:

```cmake
cmake_minimum_required(VERSION 3.14)
project(ML_From_Scratch_Lesson1 VERSION 1.0 LANGUAGES CXX)

# Enforce C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for speed (Crucial for ML workloads!)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# Option 1: Find Eigen installed on your system (e.g., via apt, brew, or vcpkg)
find_package(Eigen3 3.3 QUIET NO_MODULE)

# Option 2: Automatically fetch Eigen via CMake if not installed locally
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()

# Define the executable
add_executable(lesson1 src/main.cpp)

# Link Eigen to our executable
target_link_libraries(lesson1 PRIVATE Eigen3::Eigen)
```

---

## 4. Python (`NumPy`) vs. C++ (`Eigen`) Cheatsheet

Before diving into the code, examine how standard introductory Jupyter Notebook cells map to Eigen:

| Operation | Python (`NumPy`) | C++ (`Eigen`) |
| :--- | :--- | :--- |
| **Import Library** | `import numpy as np` | `#include <Eigen/Dense>` |
| **1D Vector (Dynamic)** | `v = np.array([1.0, 2.0, 3.0])` | `Eigen::VectorXd v(3); v << 1.0, 2.0, 3.0;` |
| **2D Matrix (Dynamic)** | `M = np.zeros((3, 4))` | `Eigen::MatrixXd M = Eigen::MatrixXd::Zero(3, 4);` |
| **Random Matrix** | `R = np.random.randn(3, 3)` | `Eigen::MatrixXd R = Eigen::MatrixXd::Random(3, 3);` |
| **Matrix Multiplication** | `C = A @ B` or `np.dot(A, B)` | `Eigen::MatrixXd C = A * B;` |
| **Element-wise Mult** | `C = A * B` | `Eigen::MatrixXd C = A.array() * B.array();` |
| **Transpose** | `A.T` | `A.transpose()` |
| **Mean / Sum** | `np.mean(A)`, `np.sum(A)` | `A.mean()`, `A.sum()` |
| **Column-wise Mean** | `np.mean(A, axis=0)` | `A.colwise().mean()` |

> **Crucial Concept (`Matrix` vs `Array` in Eigen):**  
> In `NumPy`, the `*` operator defaults to **element-wise** multiplication, and `@` is used for **matrix** multiplication. In `Eigen`, `*` defaults to **linear algebra matrix multiplication**. To perform element-wise operations (like adding a scalar or multiplying element-by-element), you must call `.array()` on the Eigen object first.

---

## 5. Step-by-Step Implementation (`src/main.cpp`)

Below is the complete C++ translation of a typical "Notebook 1: Linear Algebra & Data Loading" exercise.

### Sample Dataset (`data/sample_data.csv`)
Create a small dummy dataset in `data/sample_data.csv` representing features (e.g., *House Size*, *Bedrooms*) and a target (*Price*):
```csv
size,bedrooms,price
1500,3,300000
2000,4,410000
1200,2,240000
2500,4,500000
1800,3,350000
```

### C++ Source Code (`src/main.cpp`)

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

// Helper function to load a CSV file into an Eigen::MatrixXd
// Equivalent to: data = np.loadtxt("data/sample_data.csv", delimiter=",", skiprows=1)
Eigen::MatrixXd loadCSV(const std::string& filepath, bool skipHeader = true) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::vector<double> values;
    std::string line;
    size_t rows = 0;
    size_t cols = 0;

    if (skipHeader && std::getline(file, line)) {
        // Skip the first row
    }

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        size_t currentCols = 0;

        while (std::getline(ss, cell, ',')) {
            values.push_back(std::stod(cell));
            currentCols++;
        }

        if (rows == 0) {
            cols = currentCols;
        } else if (currentCols != cols) {
            throw std::runtime_error("Inconsistent number of columns in CSV.");
        }
        rows++;
    }

    // Eigen stores matrices in Column-Major order by default.
    // Since we read the CSV row-by-row, we map it as RowMajor and convert to standard MatrixXd.
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        rowMajorMatrix(values.data(), rows, cols);
    
    return Eigen::MatrixXd(rowMajorMatrix);
}

int main() {
    std::cout << "=== Lesson 1: ML Foundations in C++ with Eigen ===\n\n";

    // ---------------------------------------------------------
    // Part 1: Vector and Matrix Initialization
    // ---------------------------------------------------------
    Eigen::VectorXd weights(2);
    weights << 150.0, 10000.0; // Initial hypothetical weights
    double bias = 50000.0;

    std::cout << "Initial Weights:\n" << weights.transpose() << "\n\n";

    // ---------------------------------------------------------
    // Part 2: Loading Data (Standard Library -> Eigen)
    // ---------------------------------------------------------
    Eigen::MatrixXd dataset;
    try {
        dataset = loadCSV("../data/sample_data.csv", true);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Make sure you run the executable from the 'build/' directory!\n";
        return 1;
    }

    std::cout << "Loaded Dataset (" << dataset.rows() << "x" << dataset.cols() << "):\n";
    std::cout << dataset << "\n\n";

    // ---------------------------------------------------------
    // Part 3: Slicing Features (X) and Target (y)
    // Python equivalent: X = dataset[:, :-1]; y = dataset[:, -1]
    // ---------------------------------------------------------
    Eigen::MatrixXd X = dataset.leftCols(dataset.cols() - 1);
    Eigen::VectorXd y = dataset.rightCols(1);

    std::cout << "Features (X):\n" << X << "\n\n";
    std::cout << "Targets (y):\n" << y.transpose() << "\n\n";

    // ---------------------------------------------------------
    // Part 4: Feature Normalization (Standardization / Z-score)
    // Python equivalent: X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    // ---------------------------------------------------------
    Eigen::RowVectorXd mean = X.colwise().mean();
    
    // Calculate standard deviation per column
    Eigen::MatrixXd centered = X.rowwise() - mean;
    Eigen::RowVectorXd std_dev = (centered.array().square().colwise().sum() / X.rows()).sqrt();
    
    // Normalize using array() for element-wise division across rows
    Eigen::MatrixXd X_norm = centered.array().rowwise() / std_dev.array();

    std::cout << "Column Means: " << mean << "\n";
    std::cout << "Column StdDevs: " << std_dev << "\n";
    std::cout << "Normalized Features (X_norm):\n" << X_norm << "\n\n";

    // ---------------------------------------------------------
    // Part 5: Forward Pass / Linear Prediction (y_hat = X * w + b)
    // Python equivalent: y_hat = X @ weights + bias
    // ---------------------------------------------------------
    Eigen::VectorXd predictions = (X * weights).array() + bias;

    std::cout << "Initial Predictions (y_hat):\n" << predictions.transpose() << "\n";
    std::cout << "Actual Targets (y):\n" << y.transpose() << "\n\n";

    // ---------------------------------------------------------
    // Part 6: Computing Mean Squared Error (MSE) Loss
    // Python equivalent: mse = np.mean((y_hat - y) ** 2)
    // ---------------------------------------------------------
    Eigen::VectorXd errors = predictions - y;
    double mse = errors.array().square().mean();

    std::cout << "Mean Squared Error (MSE): " << mse << "\n";

    return 0;
}
```

---

## 6. Compiling and Running the Project

Open your terminal in the `lesson_01_intro/` root directory and run the following commands:

```bash
# 1. Create a build directory (out-of-source build)
mkdir build && cd build

# 2. Configure the project with CMake
cmake ..

# 3. Compile the code
cmake --build .

# 4. Run the executable
./lesson1
```

---

## 7. Exercises for Self-Study
Before moving to Lesson 2 (where we will implement **Gradient Descent**), try modifying `src/main.cpp` to complete these tasks:
1. **Min-Max Scaling:** Implement Min-Max normalization $X_{\text{scaled}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$ using `.colwise().minCoeff()` and `.colwise().maxCoeff()`.
2. **Bias Trick (Augmenting $X$):** Instead of adding `bias` separately, prepend a column of `1.0`s to `X` using `Eigen::MatrixXd::Ones(X.rows(), 1)` and append `bias` to the `weights` vector so that the entire prediction is a single matrix-vector multiplication: $\hat{y} = X_{\text{aug}} \cdot w_{\text{aug}}$.
3. **Train/Test Split:** Split the loaded `dataset` matrix into an 80% training set and a 20% testing set using `.topRows()` and `.bottomRows()`.



| 功能模块             | 操作说明                     | Eigen (C++) 写法                           | NumPy (Python) 写法                             |
| -------------------- | ---------------------------- | ------------------------------------------ | ----------------------------------------------- |
| **数据结构与初始化** | 动态大小矩阵 / 二维数组      | `Eigen::MatrixXd`                          | `np.ndarray` (2D, `dtype=float64`)              |
|                      | 动态大小列向量 / 一维数组    | `Eigen::VectorXd`                          | `np.ndarray` (1D, `dtype=float64`)              |
|                      | 创建全为固定值的向量         | `Eigen::VectorXd::Constant(n, 150.0)`      | `np.full(n, 150.0)`                             |
|                      | 获取矩阵维度（行数 / 列数）  | `X.rows()`, `X.cols()`                     | `X.shape[0]`, `X.shape[1]`                      |
| **切片与索引**       | 去掉最后一列（取特征 \(X\)） | `dataset.leftCols(dataset.cols() - 1)`     | `dataset[:, :-1]`                               |
|                      | 只取最后一列（取目标 \(y\)） | `dataset.rightCols(1)`                     | `dataset[:, -1]`                                |
|                      | 取前 \(k\) 行                | `dataset.topRows(k)`                       | `dataset[:k, :]`                                |
|                      | 取向量的前 \(k\) 个元素      | `y.head(k)`                                | `y[:k]`                                         |
|                      | 取第 0 行                    | `X.row(0)`                                 | `X[0, :]`                                       |
| **统计与归约**       | 按列求均值（返回行向量）     | `X.colwise().mean()`                       | `np.mean(X, axis=0)`                            |
|                      | 按列求和                     | `mat.colwise().sum()`                      | `np.sum(mat, axis=0)`                           |
|                      | 求所有元素的均值             | `vec.mean()`                               | `np.mean(vec)`                                  |
| **广播与逐元素运算** | 矩阵每行减去行向量（广播）   | `X.rowwise() - mean`                       | `X - mean`                                      |
|                      | 逐元素平方                   | `mat.array().square()`                     | `mat ** 2` 或 `np.square(mat)`                  |
|                      | 逐元素开平方根               | `vec.array().sqrt()`                       | `np.sqrt(vec)`                                  |
|                      | 按行逐元素相除（广播）       | `centered.array().rowwise() / std.array()` | `centered / std`                                |
|                      | 向量加上标量（广播）         | `vec.array() + bias`                       | `vec + bias`                                    |
|                      | 条件替换（防止除以 0）       | `(std.array() == 0.0).select(1.0, std)`    | `np.where(std == 0.0, 1.0, std)`                |
| **线性代数运算**     | 矩阵与向量乘法               | `X_norm * weights`                         | `X_norm @ weights` 或 `np.dot(X_norm, weights)` |
|                      | 矩阵 / 向量转置              | `weights.transpose()`                      | `weights.T`                                     |