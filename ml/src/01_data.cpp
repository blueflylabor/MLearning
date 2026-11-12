#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <Eigen/Dense>

// Helper function to load a CSV file into an Eigen::MatrixXd
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

    if (rows == 0 || cols < 2) {
        throw std::runtime_error("CSV must contain at least 1 row and 2 columns (features + target).");
    }

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> 
        rowMajorMatrix(values.data(), rows, cols);
    
    return Eigen::MatrixXd(rowMajorMatrix);
}

int main() {
    std::cout << "=== Lesson 1: ML Foundations in C++ with Eigen ===\n\n";

    // ---------------------------------------------------------
    // Part 1: Loading Data (Standard Library -> Eigen)
    // ---------------------------------------------------------
    Eigen::MatrixXd dataset;
    try {
        dataset = loadCSV("../data/my_pizzas.csv", true);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Make sure you run the executable from the 'build/' directory!\n";
        return 1;
    }

    std::cout << "Loaded Dataset (" << dataset.rows() << "x" << dataset.cols() << "):\n";
    std::cout << dataset.topRows(std::min<Eigen::Index>(10, dataset.rows())) << "\n\n";

    // ---------------------------------------------------------
    // Part 2: Slicing Features (X) and Target (y)
    // ---------------------------------------------------------
    Eigen::MatrixXd X = dataset.leftCols(dataset.cols() - 1);
    Eigen::VectorXd y = dataset.rightCols(1);

    std::cout << "Features (X) shape: " << X.rows() << "x" << X.cols() << "\n";
    std::cout << "First row of X: " << X.row(0) << "\n";
    std::cout << "Targets (y) first 5: " << y.head(std::min<Eigen::Index>(5, y.size())).transpose() << "\n\n";

    // ---------------------------------------------------------
    // Part 3: Dynamic Weights and Bias Initialization
    // ---------------------------------------------------------
    // Dynamically match the number of feature columns in X
    Eigen::VectorXd weights = Eigen::VectorXd::Constant(X.cols(), 150.0);
    if (X.cols() >= 2) {
        weights(1) = 10000.0; // Keep your original second weight if 2+ features exist
    }
    double bias = 50000.0;

    std::cout << "Initialized Weights (" << weights.size() << "x1): " << weights.transpose() << "\n";
    std::cout << "Initialized Bias: " << bias << "\n\n";

    // ---------------------------------------------------------
    // Part 4: Feature Normalization (Standardization / Z-score)
    // ---------------------------------------------------------
    Eigen::RowVectorXd mean = X.colwise().mean();
    Eigen::MatrixXd centered = X.rowwise() - mean;
    Eigen::RowVectorXd std_dev = (centered.array().square().colwise().sum() / X.rows()).sqrt();
    
    // Prevent division by zero if a feature column is constant
    Eigen::RowVectorXd safe_std = (std_dev.array() == 0.0).select(1.0, std_dev);
    Eigen::MatrixXd X_norm = centered.array().rowwise() / safe_std.array();

    std::cout << "Column Means: " << mean << "\n";
    std::cout << "Column StdDevs: " << std_dev << "\n";
    std::cout << "First row of X_norm: " << X_norm.row(0) << "\n\n";

    // ---------------------------------------------------------
    // Part 5: Forward Pass / Linear Prediction (y_hat = X_norm * w + b)
    // ---------------------------------------------------------
    // Note: Change X_norm back to X if you want predictions on unnormalized features
    Eigen::VectorXd predictions = (X_norm * weights).array() + bias;

    Eigen::Index previewCount = std::min<Eigen::Index>(5, predictions.size());
    std::cout << "Initial Predictions (first " << previewCount << "): " << predictions.head(previewCount).transpose() << "\n";
    std::cout << "Actual Targets      (first " << previewCount << "): " << y.head(previewCount).transpose() << "\n\n";

    // ---------------------------------------------------------
    // Part 6: Computing Mean Squared Error (MSE) Loss
    // ---------------------------------------------------------
    Eigen::VectorXd errors = predictions - y;
    double mse = errors.array().square().mean();

    std::cout << "Mean Squared Error (MSE): " << mse << "\n";

    return 0;
}
