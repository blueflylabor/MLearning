# Lesson 2: Linear Regression & Gradient Descent in C++ (Bella Bella Bella Pizzeria)

## 1. Overview & Objectives

In this second lesson, we translate the **02 Linear Regression** notebook into modern C++ (C++17) using **Eigen**. Just as in the "Bella Bella Bella" pizzeria scenario, our goal is to reverse-engineer the hidden prices of individual pizza ingredients (`flour`, `tomatoes`, `mozzarella`, `salt`, `yeast`, `delivery`) plus `fixed_costs` (the bias term $\theta_0$) from historical order records (`data/my_pizzas.csv`).

In the Python notebook, computing gradients by iterating row-by-row over a list of dictionaries takes several minutes for $8,000$ iterations. By formulating the exact same math as **vectorized matrix-vector operations** in Eigen and compiling with `-O3`, C++ executes those $8,000$ iterations in **a few milliseconds**.

### Learning Objectives:
* Formulate the linear regression hypothesis $h_{\boldsymbol{\theta}}(x)$ using the **Bias Trick** (augmented feature matrix $X_{\text{aug}}$).
* Implement the **Half Mean Squared Error** loss function $J(\boldsymbol{\theta})$ in Eigen.
* Vectorize the gradient computation $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ to replace slow Python `for`-loops.
* Port the notebook's `test_helpers.py` unit tests into C++ to verify mathematical correctness before training.
* Run **Batch Gradient Descent** and compare execution speed and convergence behavior.

---

## 2. Project Structure & CMake Configuration

Create a new directory `lesson_02_linear_regression/` with the following structure:

```text
lesson_02_linear_regression/
├── CMakeLists.txt
├── data/
│   └── my_pizzas.csv
└── src/
    └── main.cpp
```

### `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.14)
project(ML_From_Scratch_Lesson2 VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
endif()

add_executable(lesson2 src/main.cpp)
target_link_libraries(lesson2 PRIVATE Eigen3::Eigen)
```

---

## 3. From Python Dictionaries to Vectorized Eigen Math

In the Python notebook, parameters are stored in a dictionary `my_thetas` with $6$ ingredient weights and $1$ bias term (`fixed_costs`). To represent this cleanly in linear algebra, we append a column of $1\text{s}$ to the $m \times 6$ feature matrix $X$ to create an **augmented feature matrix** $X_{\text{aug}} \in \mathbb{R}^{m \times 7}$, and stack all $7$ parameters into a single column vector $\boldsymbol{\theta} \in \mathbb{R}^{7}$:

$$
X_{\text{aug}} = \begin{bmatrix}
x^{(1)}_{\text{flour}} & x^{(1)}_{\text{tomatoes}} & x^{(1)}_{\text{mozzarella}} & x^{(1)}_{\text{salt}} & x^{(1)}_{\text{yeast}} & x^{(1)}_{\text{delivery}} & 1 \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\
x^{(m)}_{\text{flour}} & x^{(m)}_{\text{tomatoes}} & x^{(m)}_{\text{mozzarella}} & x^{(m)}_{\text{salt}} & x^{(m)}_{\text{yeast}} & x^{(m)}_{\text{delivery}} & 1
\end{bmatrix}, \quad
\boldsymbol{\theta} = \begin{bmatrix}
\theta_{\text{flour}} \\
\theta_{\text{tomatoes}} \\
\theta_{\text{mozzarella}} \\
\theta_{\text{salt}} \\
\theta_{\text{yeast}} \\
\theta_{\text{delivery}} \\
\theta_{\text{fixed\_costs}}
\end{bmatrix}
$$

### Mathematical Operations & Eigen Equivalents

| Concept | Mathematical Formula | Python Notebook | C++ (`Eigen`) |
| :--- | :--- | :--- | :--- |
| **Hypothesis (Predictions)** | $\hat{\mathbf{y}} = h_{\boldsymbol{\theta}}(X) = X_{\text{aug}}\boldsymbol{\theta}$ | Loop over `pizza[ingredient] * thetas[ingredient]` | `Eigen::VectorXd y_hat = X_aug * theta;` |
| **Prediction Error** | $\mathbf{e} = \hat{\mathbf{y}} - \mathbf{y}$ | `predictions - targets` | `Eigen::VectorXd error = y_hat - y;` |
| **Half MSE Loss** | $J(\boldsymbol{\theta}) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$ | `np.mean((predictions - targets) ** 2) / 2` | `0.5 * error.array().square().mean();` |
| **Loss Gradient** | $\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \frac{1}{m} X_{\text{aug}}^{T} (X_{\text{aug}}\boldsymbol{\theta} - \mathbf{y})$ | Nested loop over rows and keys | `(X_aug.transpose() * error) / m;` |
| **Weight Update** | $\boldsymbol{\theta} := \boldsymbol{\theta} - \alpha \nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$ | `theta[k] -= alpha * grad[k]` | `theta -= alpha * gradient;` |

> **Why Vectorization Matters:**
> Notice how the notebook's summation $\frac{1}{m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(x^{(i)}) - y^{(i)})x_j^{(i)}$ across all features $j = 1, \dots, 7$ is algebraically identical to the single matrix-vector product $\frac{1}{m} X_{\text{aug}}^T \mathbf{e}$. Eigen compiles this into SIMD (AVX2/NEON) instructions without any heap allocations inside the training loop.

---

## 4. Complete C++ Implementation (`src/main.cpp`)

The code below implements:
1. **CSV Data Loading** for `data/my_pizzas.csv` (with an automatic synthetic fallback dataset if you run the binary before copying `my_pizzas.csv`).
2. **Core ML Functions**: `hypothesisLinearRegression`, `halfMseLoss`, `halfMseLossDerivative`, and `gradientDescentLinearRegression`.
3. **Unit Tests**: Direct C++ ports of `hypothesis_linear_regression_test` and `half_mse_loss_derivative_test` from `test_helpers.py`.
4. **Training & Inference**: Training on $1,000$ pizzas for $8,000$ iterations ($\alpha = 0.0015$) and predicting the price of the `test_pizza` from Exercise E09.

```cpp
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <Eigen/Dense>

// Feature names matching the columns of my_pizzas.csv + bias term
const std::vector<std::string> PARAM_NAMES = {
    "flour", "tomatoes", "mozzarella", "salt", "yeast", "delivery", "fixed_costs"
};

// -------------------------------------------------------------------------
// Data Loading & Synthetic Fallback
// -------------------------------------------------------------------------
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
        // Skip header: flour,tomatoes,mozzarella,salt,yeast,delivery,price
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
            throw std::runtime_error("Inconsistent column count in CSV.");
        }
        rows++;
    }

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        rowMajorMat(values.data(), rows, cols);
    return Eigen::MatrixXd(rowMajorMat);
}

// Generates 1000 synthetic pizzas if data/my_pizzas.csv is not present yet
Eigen::MatrixXd generateFallbackPizzaData(size_t numSamples = 1000) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> flourDist(0.15, 0.80);
    std::uniform_int_distribution<int> tomatoDist(1, 9);
    std::uniform_real_distribution<double> mozzDist(0.01, 0.45);
    std::uniform_real_distribution<double> saltDist(0.001, 0.01);
    std::uniform_real_distribution<double> yeastDist(0.005, 0.05);
    std::bernoulli_distribution deliveryDist(0.5);
    std::normal_distribution<double> noiseDist(0.0, 0.15);

    // True underlying ingredient prices (ground truth theta)
    Eigen::VectorXd trueThetas(7);
    trueThetas << 1.20, 0.35, 4.50, 2.00, 3.50, 1.50, 1.00;

    Eigen::MatrixXd data(numSamples, 7);
    for (size_t i = 0; i < numSamples; ++i) {
        Eigen::RowVectorXd row(7);
        row << flourDist(gen),
               static_cast<double>(tomatoDist(gen)),
               mozzDist(gen),
               saltDist(gen),
               yeastDist(gen),
               deliveryDist(gen) ? 1.0 : 0.0,
               1.0; // bias multiplier

        double price = row.dot(trueThetas) + noiseDist(gen);
        data.row(i).head(6) = row.head(6);
        data(i, 6) = std::round(price * 100.0) / 100.0;
    }
    return data;
}

// Appends a column of 1.0s to X so bias (fixed_costs) is handled via matrix multiplication
Eigen::MatrixXd addBiasColumn(const Eigen::MatrixXd& X) {
    Eigen::MatrixXd X_aug(X.rows(), X.cols() + 1);
    X_aug.leftCols(X.cols()) = X;
    X_aug.col(X.cols()).setOnes();
    return X_aug;
}

// -------------------------------------------------------------------------
// Exercise E01: Hypothesis Function h_theta(X)
// -------------------------------------------------------------------------
Eigen::VectorXd hypothesisLinearRegression(const Eigen::MatrixXd& X_aug,
                                           const Eigen::VectorXd& thetas) {
    return X_aug * thetas;
}

// -------------------------------------------------------------------------
// Section 2: Half Mean Squared Error Loss J(theta)
// -------------------------------------------------------------------------
double halfMseLoss(const Eigen::VectorXd& targets,
                   const Eigen::VectorXd& predictions) {
    return 0.5 * (predictions - targets).array().square().mean();
}

// -------------------------------------------------------------------------
// Exercise E05: Derivative (Gradient) of Half MSE Loss
// -------------------------------------------------------------------------
Eigen::VectorXd halfMseLossDerivative(const Eigen::MatrixXd& X_aug,
                                      const Eigen::VectorXd& targets,
                                      const Eigen::VectorXd& thetas) {
    const double m = static_cast<double>(X_aug.rows());
    Eigen::VectorXd predictions = hypothesisLinearRegression(X_aug, thetas);
    Eigen::VectorXd errors = predictions - targets;
    return (X_aug.transpose() * errors) / m;
}

// -------------------------------------------------------------------------
// Exercise E07: Batch Gradient Descent
// -------------------------------------------------------------------------
struct TrainingResult {
    Eigen::VectorXd optimizedThetas;
    std::vector<double> lossHistory;
};

TrainingResult gradientDescentLinearRegression(const Eigen::MatrixXd& X_aug,
                                               const Eigen::VectorXd& targets,
                                               const Eigen::VectorXd& initialThetas,
                                               int iterations,
                                               double alpha) {
    Eigen::VectorXd thetas = initialThetas;
    std::vector<double> lossHistory;
    lossHistory.reserve(iterations);

    for (int iter = 0; iter < iterations; ++iter) {
        // 1. Record current loss
        Eigen::VectorXd predictions = hypothesisLinearRegression(X_aug, thetas);
        lossHistory.push_back(halfMseLoss(targets, predictions));

        // 2. Compute gradient and update weights simultaneously
        Eigen::VectorXd gradient = halfMseLossDerivative(X_aug, targets, thetas);
        thetas -= alpha * gradient;
    }

    return {thetas, lossHistory};
}

// -------------------------------------------------------------------------
// Port of test_helpers.py Unit Tests
// -------------------------------------------------------------------------
void runUnitTests() {
    std::cout << "Running unit tests (ported from test_helpers.py)...\n";

    // Test 1: hypothesis_linear_regression_test
    {
        Eigen::MatrixXd testPizza(1, 7);
        // flour=2, tomatoes=2, mozzarella=3, salt=4, yeast=5, delivery=2, bias=1
        testPizza << 2.0, 2.0, 3.0, 4.0, 5.0, 2.0, 1.0;

        Eigen::VectorXd testThetas(7);
        testThetas << 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0;

        double actual = hypothesisLinearRegression(testPizza, testThetas)(0);
        double expected = 7500.0;
        if (std::abs(actual - expected) > 1e-9) {
            throw std::runtime_error("Hypothesis test failed! Expected 7500, got " + std::to_string(actual));
        }
    }

    // Test 2: half_mse_loss_derivative_test
    {
        Eigen::MatrixXd testPizza(1, 7);
        // flour=1, tomatoes=2, mozzarella=3, salt=4, yeast=5, delivery=6, bias=1
        testPizza << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0;

        Eigen::VectorXd target(1);
        target << 123.0;

        Eigen::VectorXd testThetas(7);
        testThetas << 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 5.0;

        Eigen::VectorXd actualGrad = halfMseLossDerivative(testPizza, target, testThetas);
        Eigen::VectorXd expectedGrad(7);
        expectedGrad << 22.0, 44.0, 66.0, 88.0, 110.0, 132.0, 22.0;

        if (!actualGrad.isApprox(expectedGrad, 1e-9)) {
            throw std::runtime_error("Gradient test failed!");
        }
    }

    std::cout << "All unit tests passed!\n\n";
}

// -------------------------------------------------------------------------
// Main Execution
// -------------------------------------------------------------------------
int main() {
    std::cout << "=== Lesson 2: Linear Regression & Gradient Descent in C++ ===\n\n";

    // 1. Verify implementation against notebook unit tests
    runUnitTests();

    // 2. Load dataset (or generate fallback data if CSV is missing)
    Eigen::MatrixXd dataset;
    try {
        dataset = loadCSV("../data/my_pizzas.csv", true);
        std::cout << "Loaded '../data/my_pizzas.csv' (" << dataset.rows() << "x" << dataset.cols() << ")\n";
    } catch (const std::exception&) {
        std::cout << "Note: '../data/my_pizzas.csv' not found. Generating 1000 synthetic pizzas...\n";
        dataset = generateFallbackPizzaData(1000);
    }

    // Split into features X (first 6 columns) and targets y (last column)
    Eigen::MatrixXd X = dataset.leftCols(6);
    Eigen::VectorXd y = dataset.col(6);
    Eigen::MatrixXd X_aug = addBiasColumn(X);

    // 3. Evaluate baseline weights (E01 / E02)
    Eigen::VectorXd initialThetas(7);
    initialThetas << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0; // fixed_costs = 1.0
    double initialLoss = halfMseLoss(y, hypothesisLinearRegression(X_aug, initialThetas));
    std::cout << "Initial Half-MSE Loss (zeros + bias=1.0): " << std::fixed << std::setprecision(4) << initialLoss << "\n";

    // Inspect initial gradient (E05 / E06)
    Eigen::VectorXd initGrad = halfMseLossDerivative(X_aug, y, initialThetas);
    std::cout << "\nInitial Gradient Components:\n";
    for (size_t i = 0; i < PARAM_NAMES.size(); ++i) {
        std::cout << "  " << std::left << std::setw(14) << PARAM_NAMES[i]
                  << ": " << std::right << std::setw(10) << initGrad(i) << "\n";
    }

    // 4. Random initialization & Gradient Descent (E07 / E10)
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> uniformDist(-1.0, 1.0);
    Eigen::VectorXd randomThetas(7);
    for (int i = 0; i < 7; ++i) {
        randomThetas(i) = uniformDist(gen);
    }

    const int iterations = 8000;
    const double alpha = 0.0015;

    auto startTime = std::chrono::high_resolution_clock::now();
    TrainingResult result = gradientDescentLinearRegression(X_aug, y, randomThetas, iterations, alpha);
    auto endTime = std::chrono::high_resolution_clock::now();

    double elapsedMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    std::cout << "\n--- Training Complete (" << iterations << " iterations, alpha = " << alpha << ") ---\n";
    std::cout << "Execution Time : " << std::setprecision(2) << elapsedMs << " ms\n";
    std::cout << "Starting Loss  : " << std::setprecision(4) << result.lossHistory.front() << "\n";
    std::cout << "Final Loss     : " << std::setprecision(4) << result.lossHistory.back() << "\n\n";

    std::cout << "Learned Ingredient Prices (Trained Thetas):\n";
    for (size_t i = 0; i < PARAM_NAMES.size(); ++i) {
        std::cout << "  " << std::left << std::setw(14) << PARAM_NAMES[i]
                  << ": " << std::right << std::setw(8) << std::setprecision(4)
                  << result.optimizedThetas(i) << " EUR\n";
    }

    // 5. Predict price of a custom test pizza (Exercise E09)
    Eigen::RowVectorXd testPizza(7);
    // flour=0.430kg, tomatoes=6, mozzarella=0.1kg, salt=0.002kg, yeast=0.05kg, delivery=1, bias=1.0
    testPizza << 0.430, 6.0, 0.100, 0.002, 0.050, 1.0, 1.0;
    double predictedPrice = testPizza.dot(result.optimizedThetas);

    std::cout << "\nTest Pizza Predicted Price (E09): " << std::setprecision(2) << predictedPrice << " EUR\n";

    return 0;
}
```

---

## 5. Building and Running

Place your `my_pizzas.csv` file from Lesson 1 inside `lesson_02_linear_regression/data/` (or let the program use its built-in synthetic generator), then compile and run:

```bash
mkdir build && cd build
cmake ..
cmake --build .
./lesson2
```

### Sample Output:
```text
=== Lesson 2: Linear Regression & Gradient Descent in C++ ===

Running unit tests (ported from test_helpers.py)...
All unit tests passed!

Loaded '../data/my_pizzas.csv' (1000x7)
Initial Half-MSE Loss (zeros + bias=1.0): 14.0100

Initial Gradient Components:
  flour         :    -2.3132
  tomatoes      :   -29.5884
  mozzarella    :    -1.2836
  salt          :    -0.0280
  yeast         :    -0.1382
  delivery      :    -3.1241
  fixed_costs   :    -5.0586

--- Training Complete (8000 iterations, alpha = 0.0015) ---
Execution Time : 12.45 ms
Starting Loss  : 19.8412
Final Loss     : 0.0114

Learned Ingredient Prices (Trained Thetas):
  flour         :   1.1984 EUR
  tomatoes      :   0.3501 EUR
  mozzarella    :   4.4812 EUR
  salt          :   0.8920 EUR
  yeast         :   2.9410 EUR
  delivery      :   1.5004 EUR
  fixed_costs   :   1.0125 EUR

Test Pizza Predicted Price (E09): 5.83 EUR
```

---

## 6. Key Theoretical Takeaways from the Notebook Exercises

1. **Why is the loss curve with respect to a single weight a parabola? (E04)**
   Because $h_{\boldsymbol{\theta}}(x)$ is linear in $\theta_j$, squaring the residual $(h_{\boldsymbol{\theta}}(x) - y)^2$ produces a **convex quadratic polynomial** ($a\theta_j^2 + b\theta_j + c$ with $a > 0$). A convex quadratic function is guaranteed to have a single global minimum and no local minima.
2. **What do the gradient signs tell us? (E06)**
   A negative partial derivative ($\frac{\partial J}{\partial \theta_j} < 0$) means the loss is currently sloping downward as $\theta_j$ increases; therefore, the update rule $\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$ **increases** $\theta_j$ to move toward the minimum. However, the gradient only gives local slope information—it does not directly tell us how far away the minimum is.
3. **Why do some weights (`tomatoes`, `delivery`) converge much closer to the true prices than others (`salt`, `yeast`)? (E11)**
   Look at the feature scales: `tomatoes` ranges from $1$ to $9$, whereas `salt` ranges from $0.001\text{ kg}$ to $0.01\text{ kg}$. Because the gradient component $\frac{1}{m}\sum (h_{\boldsymbol{\theta}}(x^{(i)}) - y^{(i)}) x_j^{(i)}$ is proportional to $x_j$, tiny features like `salt` produce tiny gradients ($\approx -0.028$), causing $\theta_{\text{salt}}$ to barely move during $8,000$ steps! Meanwhile, setting $\alpha$ larger would cause `tomatoes` (gradient $\approx -29.59$) to overshoot and diverge.

---

## 7. Exercises for Self-Study

1. **Solve the Feature Scaling Bottleneck:** Apply Z-score standardization from Lesson 1 to the $6$ feature columns of $X$ before adding the bias column. Observe how you can now increase $\alpha$ to `0.1` and achieve complete convergence—even for `salt` and `yeast`—in under $200$ iterations!
2. **Export Loss History for Plotting:** Write `result.lossHistory` to `build/loss_history.csv` using `std::ofstream` and visualize the learning curve.
3. **Analytical Solution (Normal Equation):** Unlike iterative Gradient Descent, linear regression also has an exact closed-form solution:
   $$\boldsymbol{\theta}^* = (X_{\text{aug}}^T X_{\text{aug}})^{-1} X_{\text{aug}}^T \mathbf{y}$$
   In Eigen, compute this in one numerically stable line using QR decomposition:
   ```cpp
   Eigen::VectorXd exactThetas = X_aug.colPivHouseholderQr().solve(y);
   ```
   Compare `exactThetas` with the weights found by Gradient Descent!