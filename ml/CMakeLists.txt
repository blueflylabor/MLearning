cmake_minimum_required(VERSION 3.15)
project(00_HelloWorld VERSION 1.0 LANGUAGES CXX)

# Enforce C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package installed on your system
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Define the executable target
add_executable(hello_world src/main.cpp)

# Link OpenCV libraries to our executable
target_link_libraries(hello_world PRIVATE ${OpenCV_LIBS})