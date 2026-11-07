cmake_minimum_required(VERSION 3.10)
project(ImageProcessingHomework)

set(CMAKE_CXX_STANDARD 17)

# 寻找系统的 OpenCV 库
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(homework_filters src/main.cpp)
target_link_libraries(homework_filters ${OpenCV_LIBS})
