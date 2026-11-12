#include <iostream>
#include <string>
#include <Eigen/Dense>

int main() {
    std::string test_string = "Hello World!";
    std::cout << test_string << std::endl;

    // Quick sanity check to verify Eigen is working
    Eigen::Vector3d vec(1.0, 2.0, 3.0);
    std::cout << "Eigen Test Vector:\n" << vec << std::endl;
    return 0;
}
