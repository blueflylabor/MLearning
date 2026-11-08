cmake_minimum_required(VERSION 3.15)
project(Engine3CVProject LANGUAGES CXX)

# 1. 设置 C++ 标准（OpenCV 和 Eigen3 均需要较新的标准支持）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 寻找 Eigen3 库
# Eigen3 官方推荐通过 NO_MODULE 查找其导出的 CMake 配置文件
find_package(Eigen3 REQUIRED NO_MODULE)

# 3. 寻找 OpenCV 库
find_package(OpenCV REQUIRED)

# 4. 添加你的可执行文件目标
add_executable(raii raii.cpp)
add_executable(engine3_cv_demo eigen_cv_test.cpp)

add_executable(l3cache l3cache.cpp)

add_executable(tensorTest tensor_test.cpp)
target_link_libraries(tensorTest ${OpenCV_LIBS})

# 5. 链接库到目标
# Eigen3::Eigen 和 ${OpenCV_LIBS} 会自动传递对应的 Include 路径和 Lib 路径
target_link_libraries(engine3_cv_demo
    PRIVATE
        Eigen3::Eigen
        ${OpenCV_LIBS}
)
