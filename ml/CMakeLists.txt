cmake_minimum_required(VERSION 3.16)
project(00_hello_world LANGUAGES CXX)

# Use C++17 or C++20 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Automatically fetch Eigen3 (header-only linear algebra library)
include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Eigen)

# Define the executable
add_executable(00hello_world src/00_hello_world.cpp)
add_executable(01data src/01_data.cpp)
# Link Eigen to our executable
target_link_libraries(00hello_world PRIVATE Eigen3::Eigen)
target_link_libraries(01data PRIVATE Eigen3::Eigen)
