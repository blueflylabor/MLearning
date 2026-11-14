# Machine Learning Practical Course (C++ Edition)
## Lesson 00: Hello World, CMake, & Course Overview

**Before you start:**
* **Version Control & Backups:** Just as you would copy an assignment notebook before editing, always create a separate working directory or a new Git branch (e.g., `git checkout -b my_00_hello_world`) before modifying starter templates. This prevents merge conflicts and accidental overwrites.
* **Feedback:** If you find an error, think a concept is unclear, or have an idea for improving these C++ guides, please let us know!

---

## Part 1: Course Overview & Project Roadmap

Welcome to the **Machine Learning Lab**! While the original course uses Python and Jupyter Notebooks, this adapted track guides you through building machine learning algorithms from scratch in **Modern C++ (C++17/20)** using the **C++ Standard Library (STL)** and **OpenCV (`cv::Mat`)** for matrix math and computer vision.

### 1.1 Examples of Machine Learning in Practice
* [ChatGPT](https://chatgpt.com) / [Claude](https://claude.ai) / [Gemini](https://gemini.google.com) (Large Language Models)
* [DeepL](https://www.deepl.com/translator) (Neural Machine Translation)
* [Quick, Draw!](https://quickdraw.withgoogle.com) (Real-time Sketch Recognition)
* [ThisPersonDoesNotExist.com](https://thispersondoesnotexist.com) (Generative Adversarial Networks)

### 1.2 General Goal & Final Project
* **General Goal:** Learn to develop neural networks from scratch—without relying on high-level deep learning frameworks like PyTorch or TensorFlow—and apply them as a real-time Human-Computer Interaction (HCI) tool.
* **The Objective:** Build a C++ application that captures webcam input (via OpenCV) and lets you control a slideshow using hand gestures recognized by your own custom-built neural network.

### 1.3 Course Phases & Timeline
Based on the course structure (`phases.png`), your progression follows four main stages:
1. **Initial Setup & Foundations (You are here):** Setting up your C++ compiler, CMake, and OpenCV. Learning how C++ handles memory, matrices, and basic data pipelines.
2. **Midterm Submission (Dataset Creation):** Collecting gesture data, analyzing class distributions, and preprocessing images into clean feature vectors.
3. **Core ML & Backpropagation (Peak Difficulty):** Implementing linear regression, multi-layer perceptrons (MLPs), activation functions, and backpropagation purely using `std::vector` and `cv::Mat`.
4. **Independent Project Work & Final Presentation:** Integrating your trained model into a real-time C++ gesture-controlled slideshow application.

---

## Part 2: Setting Up Your C++ & CMake Environment

In Python, you write code in interactive browser "cells" backed by a persistent Python kernel. In C++, code is **compiled** ahead of time into a native machine executable. This gives you massive performance benefits for real-time computer vision, though it changes how we structure and run our experiments.

### 2.1 Tools You Need
1. **A C++ Compiler:** GCC (`g++`), Clang, or MSVC supporting at least **C++17**.
2. **CMake (version 3.15+):** Our build system generator. Instead of typing long compiler commands manually, CMake reads a `CMakeLists.txt` file and links libraries like OpenCV automatically.
3. **OpenCV (4.x):** We use OpenCV not just for camera capture, but also as our core N-dimensional array/matrix library (`cv::Mat`), replacing Python's NumPy.
4. **An IDE / Editor:** Visual Studio Code (with the *C/C++* and *CMake Tools* extensions) or CLion.

### 2.2 Project Directory Structure
Create a folder for this first lesson with the following layout:

```text
00_hello_world/
├── CMakeLists.txt
└── src/
    └── main.cpp
```

### 2.3 Creating the `CMakeLists.txt`
Create a file named `CMakeLists.txt` in the root of `00_hello_world/` and add the following configuration:

```cmake
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
```

---

## Part 3: From Notebook Cells to C++ Execution

Let's translate the introductory notebook exercises into C++ to understand how execution, state, and compilation errors differ from Python.

### 3.1 Hello World & OpenCV Verification
In the notebook, you ran `test_string = "Hello World!"` and printed it. Let's do the same in `src/main.cpp`, while also verifying that OpenCV is properly linked via CMake.

```cpp
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
```

### 3.2 Building and Running via Terminal
Open your terminal inside the `00_hello_world/` directory and run:

```bash
# 1. Create a build directory (keeps compiled files separate from source code)
mkdir build && cd build

# 2. Configure the project with CMake
cmake ..

# 3. Compile the code
cmake --build .

# 4. Run the executable (use .\Debug\hello_world.exe on Windows MSVC)
./hello_world
```

---

## Part 4: Exercises (Notebook Pitfalls vs. C++ Guarantees)

In Jupyter Notebooks, all cells share the same runtime memory. Running cells out of order or renaming variables without restarting the kernel causes subtle bugs. Let's examine how C++ handles these exact scenarios.

### E01: Execution Order & Deterministic State
In Python notebooks, running `a = 2`, `b = a + 4`, and `a = a * b` out of order produces different results depending on the hidden kernel state. 

In C++, execution always starts strictly at `int main()` and flows top-to-bottom. There is no "hidden kernel state" left over from previous runs.

**Task:** Add the following block to `main.cpp`, compile, and verify the output:
```cpp
int a = 2;
int b = a + 4;
a = a * b;
std::cout << "Result of a: " << a << std::endl; // Always 12
```

### E02: Typos & Compile-Time Safety
In the Python exercise:
1. You defined `universaty = "JMU"` and printed it.
2. You fixed the typo on line 1 to `university = "JMU"`, leaving `print(universaty)` on line 2.
3. Python **did not complain** until you restarted the kernel, because `universaty` lingered in memory.

**Task:** Try simulating this in C++:
```cpp
std::string university = "JMU"; // Fixed typo
std::cout << "My university is " << universaty << std::endl; // Intentional typo
```
Try compiling this with `cmake --build .`. Notice that the C++ compiler immediately halts with `error: 'universaty' was not declared in this scope`. Static typing and compilation catch these state bugs before your program ever runs!

---

## Part 5: Essential Workflow Tips (No Mouse Needed!)

Just as Jupyter uses keyboard shortcuts (`Ctrl+Enter`, `Shift+Enter`, `A`, `B`), efficient C++ development relies on terminal and editor shortcuts:
* **VS Code Build Shortcut:** Press `Ctrl+Shift+B` (or `Cmd+Shift+B` on Mac) to trigger CMake build directly without touching the mouse.
* **Terminal History:** Press the `Up Arrow` in your terminal to recall `cmake --build . && ./hello_world` to recompile and run in one step.
* **Documentation:** Keep the [OpenCV `cv::Mat` Documentation](https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html) and [cppreference.com](https://en.cppreference.com/) bookmarked—they are your primary references for this course.