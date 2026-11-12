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
const std::vector<std::string> DEFAULT_FEATURE_NAMES = {
    "flour", "tomatoes", "mozzarella", "salt", "yeast", "delivery"};

// -------------------------------------------------------------------------
// Data Loading & Synthetic Fallback
// -------------------------------------------------------------------------
Eigen::MatrixXd loadCSV(const std::string &filepath, bool skipHeader = true)
{
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    std::vector<double> values;
    std::string line;
    size_t rows = 0;
    size_t cols = 0;

    if (skipHeader && std::getline(file, line))
    {
        // Skip header
    }

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;
        std::stringstream ss(line);
        std::string cell;
        size_t currentCols = 0;

        while (std::getline(ss, cell, ','))
        {
            values.push_back(std::stod(cell));
            currentCols++;
        }

        if (rows == 0)
        {
            cols = currentCols;
        }
        else if (currentCols != cols)
        {
            throw std::runtime_error("Inconsistent column count in CSV.");
        }
        rows++;
    }

    if (rows == 0 || cols < 2)
    {
        throw std::runtime_error("CSV must have at least 1 row and 2 columns.");
    }

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        rowMajorMat(values.data(), rows, cols);
    return Eigen::MatrixXd(rowMajorMat);
}

// Generates 1000 synthetic pizzas if data/my_pizzas.csv is not present yet
Eigen::MatrixXd generateFallbackPizzaData(size_t numSamples = 1000)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> flourDist(0.15, 0.80);
    std::uniform_int_distribution<int> tomatoDist(1, 9);
    std::uniform_real_distribution<double> mozzDist(0.01, 0.45);
    std::uniform_real_distribution<double> saltDist(0.001, 0.01);
    std::uniform_real_distribution<double> yeastDist(0.005, 0.05);
    std::bernoulli_distribution deliveryDist(0.5);
    std::normal_distribution<double> noiseDist(0.0, 0.15);

    Eigen::VectorXd trueThetas(7);
    trueThetas << 1.20, 0.35, 4.50, 2.00, 3.50, 1.50, 1.00;

    Eigen::MatrixXd data(numSamples, 7);
    for (size_t i = 0; i < numSamples; ++i)
    {
        Eigen::RowVectorXd row(7);
        row << flourDist(gen),
            static_cast<double>(tomatoDist(gen)),
            mozzDist(gen),
            saltDist(gen),
            yeastDist(gen),
            deliveryDist(gen) ? 1.0 : 0.0,
            1.0;

        double price = row.dot(trueThetas) + noiseDist(gen);
        data.row(i).head(6) = row.head(6);
        data(i, 6) = std::round(price * 100.0) / 100.0;
    }
    return data;
}

// Appends a column of 1.0s to X so bias (fixed_costs) is handled via matrix multiplication
Eigen::MatrixXd addBiasColumn(const Eigen::MatrixXd &X)
{
    Eigen::MatrixXd X_aug(X.rows(), X.cols() + 1);
    X_aug.leftCols(X.cols()) = X;
    X_aug.col(X.cols()).setOnes();
    return X_aug;
}

// -------------------------------------------------------------------------
// Exercise E01: Hypothesis Function h_theta(X)
// -------------------------------------------------------------------------
Eigen::VectorXd hypothesisLinearRegression(const Eigen::MatrixXd &X_aug,
                                           const Eigen::VectorXd &thetas)
{
    return X_aug * thetas;
}

// -------------------------------------------------------------------------
// Section 2: Half Mean Squared Error Loss J(theta)
// -------------------------------------------------------------------------
double halfMseLoss(const Eigen::VectorXd &targets,
                   const Eigen::VectorXd &predictions)
{
    return 0.5 * (predictions - targets).array().square().mean();
}

// -------------------------------------------------------------------------
// Exercise E05: Derivative (Gradient) of Half MSE Loss
// -------------------------------------------------------------------------
Eigen::VectorXd halfMseLossDerivative(const Eigen::MatrixXd &X_aug,
                                      const Eigen::VectorXd &targets,
                                      const Eigen::VectorXd &thetas)
{
    const double m = static_cast<double>(X_aug.rows());
    Eigen::VectorXd predictions = hypothesisLinearRegression(X_aug, thetas);
    Eigen::VectorXd errors = predictions - targets;
    return (X_aug.transpose() * errors) / m;
}

// -------------------------------------------------------------------------
// Exercise E07: Batch Gradient Descent
// -------------------------------------------------------------------------
struct TrainingResult
{
    Eigen::VectorXd optimizedThetas;
    std::vector<double> lossHistory;
};

TrainingResult gradientDescentLinearRegression(const Eigen::MatrixXd &X_aug,
                                               const Eigen::VectorXd &targets,
                                               const Eigen::VectorXd &initialThetas,
                                               int iterations,
                                               double alpha)
{
    Eigen::VectorXd thetas = initialThetas;
    std::vector<double> lossHistory;
    lossHistory.reserve(iterations);

    for (int iter = 0; iter < iterations; ++iter)
    {
        Eigen::VectorXd predictions = hypothesisLinearRegression(X_aug, thetas);
        lossHistory.push_back(halfMseLoss(targets, predictions));

        Eigen::VectorXd gradient = halfMseLossDerivative(X_aug, targets, thetas);
        thetas -= alpha * gradient;
    }

    return {thetas, lossHistory};
}

// -------------------------------------------------------------------------
// Port of test_helpers.py Unit Tests
// -------------------------------------------------------------------------
void runUnitTests()
{
    std::cout << "Running unit tests (ported from test_helpers.py)...\n";

    // Test 1: hypothesis_linear_regression_test
    {
        Eigen::MatrixXd testPizza(1, 7);
        testPizza << 2.0, 2.0, 3.0, 4.0, 5.0, 2.0, 1.0;

        Eigen::VectorXd testThetas(7);
        testThetas << 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0;

        double actual = hypothesisLinearRegression(testPizza, testThetas)(0);
        double expected = 7500.0;
        if (std::abs(actual - expected) > 1e-9)
        {
            throw std::runtime_error("Hypothesis test failed! Expected 7500, got " + std::to_string(actual));
        }
    }

    // Test 2: half_mse_loss_derivative_test
    {
        Eigen::MatrixXd testPizza(1, 7);
        testPizza << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0;

        Eigen::VectorXd target(1);
        target << 123.0;

        Eigen::VectorXd testThetas(7);
        testThetas << 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 5.0;

        Eigen::VectorXd actualGrad = halfMseLossDerivative(testPizza, target, testThetas);
        Eigen::VectorXd expectedGrad(7);
        expectedGrad << 22.0, 44.0, 66.0, 88.0, 110.0, 132.0, 22.0;

        if (!actualGrad.isApprox(expectedGrad, 1e-9))
        {
            throw std::runtime_error("Gradient test failed!");
        }
    }

    std::cout << "All unit tests passed!\n\n";
}

// -------------------------------------------------------------------------
// Main Execution
// -------------------------------------------------------------------------
int main()
{
    std::cout << "=== Lesson 2: Linear Regression & Gradient Descent in C++ ===\n\n";

    // 1. Verify implementation against notebook unit tests
    runUnitTests();

    // 2. Load dataset (or generate fallback data if CSV is missing)
    Eigen::MatrixXd dataset;
    try
    {
        dataset = loadCSV("../data/my_pizzas.csv", true);
        std::cout << "Loaded '../data/my_pizzas.csv' (" << dataset.rows() << "x" << dataset.cols() << ")\n";
    }
    catch (const std::exception &)
    {
        std::cout << "Note: '../data/my_pizzas.csv' not found. Generating 1000 synthetic pizzas...\n";
        dataset = generateFallbackPizzaData(1000);
    }

    // Dynamically determine feature count & parameter count (fixes out-of-bounds error)
    const Eigen::Index numFeatures = dataset.cols() - 1;
    const Eigen::Index numParams = numFeatures + 1;

    // Build parameter names dynamically to match CSV columns + bias
    std::vector<std::string> paramNames;
    for (Eigen::Index i = 0; i < numFeatures; ++i)
    {
        if (i < static_cast<Eigen::Index>(DEFAULT_FEATURE_NAMES.size()))
        {
            paramNames.push_back(DEFAULT_FEATURE_NAMES[i]);
        }
        else
        {
            paramNames.push_back("feature_" + std::to_string(i));
        }
    }
    paramNames.push_back("fixed_costs");

    // Split into features X (first numFeatures columns) and targets y (last column)
    Eigen::MatrixXd X = dataset.leftCols(numFeatures);
    Eigen::VectorXd y = dataset.rightCols(1);
    Eigen::MatrixXd X_aug = addBiasColumn(X);

    // 3. Evaluate baseline weights (E01 / E02)
    Eigen::VectorXd initialThetas = Eigen::VectorXd::Zero(numParams);
    initialThetas(numParams - 1) = 1.0; // fixed_costs = 1.0

    double initialLoss = halfMseLoss(y, hypothesisLinearRegression(X_aug, initialThetas));
    std::cout << "Initial Half-MSE Loss (zeros + bias=1.0): " << std::fixed << std::setprecision(4) << initialLoss << "\n";

    // Inspect initial gradient (E05 / E06)
    Eigen::VectorXd initGrad = halfMseLossDerivative(X_aug, y, initialThetas);
    std::cout << "\nInitial Gradient Components:\n";
    for (Eigen::Index i = 0; i < numParams; ++i)
    {
        std::cout << "  " << std::left << std::setw(14) << paramNames[i]
                  << ": " << std::right << std::setw(10) << initGrad(i) << "\n";
    }

    // 4. Random initialization & Gradient Descent (E07 / E10)
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> uniformDist(-1.0, 1.0);
    Eigen::VectorXd randomThetas(numParams);
    for (Eigen::Index i = 0; i < numParams; ++i)
    {
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
    for (Eigen::Index i = 0; i < numParams; ++i)
    {
        std::cout << "  " << std::left << std::setw(14) << paramNames[i]
                  << ": " << std::right << std::setw(8) << std::setprecision(4)
                  << result.optimizedThetas(i) << " EUR\n";
    }
    // 5. Predict price of a custom test pizza (Exercise E09)
    Eigen::RowVectorXd defaultTestValues(7);
    defaultTestValues << 0.430, 6.0, 0.100, 0.002, 0.050, 1.0, 1.0;
    Eigen::RowVectorXd testPizza(numParams);
    for (Eigen::Index i = 0; i < numFeatures; ++i)
    {
        testPizza(i) = (i < 6) ? defaultTestValues(i) : 0.0;
    }
    testPizza(numParams - 1) = 1.0; // bias multiplier = 1.0
    double predictedPrice = testPizza.dot(result.optimizedThetas);
    std::cout << "\nTest Pizza Predicted Price (E09): " << std::setprecision(2) << predictedPrice << " EUR\n";
    return 0;
}