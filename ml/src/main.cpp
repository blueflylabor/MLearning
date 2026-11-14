#include <iostream>
#include <string>
#include <opencv2/core.hpp>

int main() {
    // Equivalent to: test_string = "Hello World!"
    std::string test_string = "Hello World!";
    std::cout << test_string << std::endl;

    // Verify OpenCV integration (our NumPy equivalent)
    cv::Mat matrix = cv::Mat::eye(3, 3, CV_64F);
    std::cout << "\nOpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "3x3 Identity Matrix:\n" << matrix << std::endl;

    return 0;
}