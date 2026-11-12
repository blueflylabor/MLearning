# 00: Hello World & C++ Environment Setup (CMake + Eigen)

**Before you start:**

* **Version Control & Workspace:** Don't work directly on the template branch! Create a copy or a new branch (e.g., `git checkout -b my-assignment-00` or copy `00_hello_world_template/` to `00_hello_world_my_assignment/`). This ensures there won't be any merge conflicts and your changes won't be accidentally overwritten.
* **Feedback:** If you find an error, think something is too confusing or not well explained, or have an idea for an improvement, please drop us a line. We constantly try to improve these guides.

---

Welcome to the Machine Learning Practical Course (C++ Edition)! Before we get down to linear regression, neural networks, etc., let's get you started with modern C++, CMake, and essential linear algebra libraries like **Eigen**.

**If you are new to C++**, we recommend checking out [LearnCpp.com](https://www.learncpp.com/) for a comprehensive introduction to the fundamentals. Make sure you understand static typing, compilation vs. execution, references/pointers, conditions, loops, and the Standard Template Library (`std::vector`, `std::string`). This guide covers the basics of our C++ workflow, though it cannot cover every language detail you will need.

---

### How C++ Differs from Jupyter Notebooks

In Python courses, you often use a browser-based "notebook" consisting of interactive "cells" connected to a background Python kernel. 

In C++, we use a **compiled workflow** inside a proper IDE (like **VS Code**, **CLion**, or **Visual Studio**). Instead of running isolated cells in memory, we write source files (`.cpp`, `.hpp`), configure the build system using **CMake**, compile the code into a fast native binary, and run it in the terminal.

Let's try your first C++ program!

#### 1. Project Structure

Create a folder named `00_hello_world` with the following two files:

**`CMakeLists.txt`**
```cmake
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
add_executable(hello_world main.cpp)

# Link Eigen to our executable
target_link_libraries(hello_world PRIVATE Eigen3::Eigen)
```

**`main.cpp`**
```cpp
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
```

#### 2. Building and Running the Code

Open your terminal inside the `00_hello_world` directory and run:

```bash
# 1. Configure the project and download Eigen
cmake -B build

# 2. Compile the C++ source code
cmake --build build

# 3. Execute the compiled binary
./build/hello_world        # On Linux/macOS
# .\build\Debug\hello_world.exe  # On Windows
```

**Expected Output:**
```text
Hello World!
Eigen Test Vector:
1
2
3
```

---

### State & Execution Order: Why C++ Saves You from Notebook Pitfalls

In Jupyter notebooks, all cells share the same background memory session. Running cells out of order or renaming variables without restarting the kernel leads to hidden bugs. In C++, the compiler enforces strict top-to-bottom execution within `main()` and catches variable errors *before* your code even runs.

#### E01: Execution Order & State Mutation

In a notebook, running statements out of order changes variable states unpredictably. In C++, statements execute strictly sequentially from top to bottom inside a function scope:

```cpp
#include <iostream>

int main() {
    int a = 2;
    int b = a + 4; // b is 6
    a = a * b;     // a becomes 12
    
    std::cout << "Final value of a: " << a << std::endl;
    return 0;
}
```
*Try swapping the lines `int b = a + 4;` and `a = a * b;` in `main.cpp` and recompiling. Notice how C++ either refuses to compile (if `b` is used before declaration) or produces completely deterministic results based on your explicit code order.*

#### E02: Typos and Static Type Checking

In Python notebooks, if you make a typo in a variable name, fix the declaration, and forget to fix the `print` statement, Python might still run without complaining if the old misspelled variable lingers in memory!

Let's test this in C++:
1. Write the following code in `main.cpp` and compile it:
```cpp
#include <iostream>
#include <string>

int main() {
    std::string universaty = "JMU";
    std::cout << "My university is " << universaty << std::endl;
    return 0;
}
```
2. Now, fix the typo **only** in the first line (`std::string university = "JMU";`), leaving `universaty` in the `std::cout` line.
3. Recompile (`cmake --build build`).

**What happens?**
Unlike Python, the C++ compiler immediately halts and throws a compile-time error:
`error: 'universaty' was not declared in this scope; did you mean 'university'?`

C++ has no lingering "ghost variables" in memory between runs. Every build starts with a clean slate, protecting you from subtle state bugs as our machine learning codebases grow larger.

---

### IDE Shortcuts (VS Code / CLion)

Having to switch from mouse to keyboard all the time slows you down. Master your IDE shortcuts early:

* **Build Project:** `Ctrl + Shift + B` (VS Code) / `Ctrl + F9` (CLion)
* **Run / Debug:** `F5` (VS Code) / `Shift + F10` (CLion)
* **Toggle Terminal:** `` Ctrl + ` `` (VS Code) / `Alt + F12` (CLion)
* **Comment / Uncomment Line:** `Ctrl + /`
* **Go to Definition:** `F12` or `Ctrl + Click`
* **Rename Symbol Safely (Refactors across all files!):** `F2` (VS Code) / `Shift + F6` (CLion)

---

### Documentation & Markdown

Just like Markdown cells in notebooks, C++ projects rely heavily on `README.md` files and inline Doxygen comments (`///`) to document architecture, math formulas, and build instructions.

**Try it:**
Create a `NOTES.md` file in your project root and practice formatting:
* Use `**bold**` for emphasis.
* Use `` `inline code` `` for C++ types like `Eigen::MatrixXd`.
* Use `$$ ... $$` for LaTeX math equations (e.g., $\hat{y} = Wx + b$).
* Reference the [Markdown Cheat Sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet).

---

### How We Will Use C++ & CMake in This Course

1. **Clean Separation:** Every assignment will be structured as a clean CMake target. Always copy the template directory or branch before starting your work.
2. **Read & Trace:** Read the mathematical explanations in the `.md` guides first, then trace through the C++ implementations step-by-step.
3. **Experiment:** Add helper functions, print matrix dimensions (`mat.rows()`, `mat.cols()`), and test edge cases. Treat your `main.cpp` as your experimental playground.
4. **Why C++ instead of Python scripts?** While Python is great for quick prototyping, real-world Human-Computer Interaction (HCI) and real-time computer vision/gesture recognition require **low latency and high throughput**. Writing neural networks from scratch in C++ with Eigen gives you direct control over memory layout, SIMD vectorization, and deterministic real-time performance.