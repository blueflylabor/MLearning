# Machine Learning Practical Course (C++ & OpenCV Edition)
## Lesson 01: Data Generation & Tabular Processing with C++ STL and `cv::Mat`

**Before you start:**

* **Don't work directly in the original template folder!** Copy the starter project `01_data.assignment/` to `01_data.my_assignment/` (or create a dedicated Git branch such as `git checkout -b my_01_data`) before editing any files. This prevents merge conflicts and ensures your changes are never accidentally overwritten.
* **Feedback:** If you find an error, think something is too confusing or not well explained, or have an idea for an improvement, please let us know. We constantly strive to improve these tutorials.

---

## 0. Project Setup & `CMakeLists.txt`

Before writing code, set up your project directory for this assignment:

```text
01_data.my_assignment/
├── CMakeLists.txt
└── src/
    └── main.cpp
```

### `CMakeLists.txt`
In this lesson, we use the **C++17 Standard Library** (`<random>`, `<vector>`, `<fstream>`, `<sstream>`, `<algorithm>`, `<cmath>`) alongside **OpenCV** (`cv::Mat`) to replace Python's `random` and `pandas` libraries.

Create `CMakeLists.txt` in your project root:

```cmake
cmake_minimum_required(VERSION 3.15)
project(01_DataGeneration VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(data_generator src/main.cpp)
target_link_libraries(data_generator PRIVATE ${OpenCV_LIBS})
```

---

## 1. Motivation

Machine learning is all about data. Ideally, you already have a dataset to work with and can train your machine learning models on it. However, sometimes it is useful to create your own synthetic data—for example, when you do not have real data yet, want to test your model under controlled conditions, or want to understand the underlying structure of data in the first place.

Therefore, as our first step, we will generate our own dataset and later apply our machine learning models to it. This way, we can verify whether our C++ algorithms can reverse-engineer the exact ground-truth parameters we used to create the data.

To make things vivid, let's introduce an example scenario: suppose we examine the order books of the pizzeria **"Bella Bella Bella"**! The pizzeria produces pizzas throughout the year and records every ingredient used along with the final pizza price.

The problem: we made up the pizzeria, so no actual order book exists yet! We have to generate our own. **The goal of this lesson is to generate an order book containing $1,000$ pizzas and their individual prices, manipulate it using `cv::Mat`, and export it to `my_pizzas.csv`.**

---

## 2. Generate Data

### 2.1 Representing Structured Records: Python `dict` vs. C++ `struct`
In Python, a single pizza order is represented using a [dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) (`{"flour": 0.200, "tomatoes": 2, ...}`). While C++ offers `std::unordered_map<std::string, double>` as a direct dictionary equivalent, C++ developers strongly prefer a strongly-typed **`struct`** when the field names are known in advance. A `struct` gives you compile-time typo checking, zero hash-lookup overhead, and clean dot-notation access (`pizza_A.flour`).

Let's define `struct Pizza` and `struct PriceList`, and create our first pizza in `src/main.cpp`:

```cpp
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <opencv2/core.hpp>

// Equivalent to a Python dictionary representing one pizza's ingredients
struct Pizza {
    double flour;       // kg
    int tomatoes;       // number of tomatoes (integer)
    double mozzarella;  // kg
    double salt;        // kg
    double yeast;       // kg
    int delivery;       // yes or no: 1 or 0 (integer)
};

// Equivalent to `pizza_ingredients_price_list` dictionary
struct PriceList {
    double flour;        // €/kg
    double tomatoes;     // €/tomato
    double mozzarella;   // €/kg
    double salt;         // €/kg
    double yeast;        // €/kg
    double delivery;     // €/delivery
    double fixed_costs;  // €/pizza (labor, rent, oven electricity, etc.)
};
```

In `main()`, initialize `pizza_A` (feel free to change the quantities):

```cpp
    Pizza pizza_A{
        0.200,  // flour (kg)
        2,      // tomatoes (count)
        0.300,  // mozzarella (kg)
        0.1,    // salt (kg)
        0.2,    // yeast (kg)
        1       // delivery (1 = yes, 0 = no)
    };
```

### 2.2 Defining the Price List & Calculating Pizza Prices

**Task 1:** Initialize `pizza_ingredients_price_list` with realistic prices (in Euros) for each item instead of `0.0`:

```cpp
    PriceList pizza_ingredients_price_list{
        0.0,  // flour (€/kg)
        0.0,  // tomatoes (€/tomato)
        0.0,  // mozzarella (€/kg)
        0.0,  // salt (€/kg)
        0.0,  // yeast (€/kg)
        0.0,  // delivery (€/delivery)
        0.0   // fixed_costs (€/pizza)
    };
```

Mathematically, the noiseless price $y$ of a pizza is a linear combination of its ingredient quantities $\mathbf{x}$ and unit prices $\mathbf{w}$ plus fixed overhead $b$:

$$y = w_{\text{flour}} x_{\text{flour}} + w_{\text{tomatoes}} x_{\text{tomatoes}} + w_{\text{mozzarella}} x_{\text{mozzarella}} + w_{\text{salt}} x_{\text{salt}} + w_{\text{yeast}} x_{\text{yeast}} + w_{\text{delivery}} x_{\text{delivery}} + b_{\text{fixed}}$$

Let's encapsulate this formula in a reusable C++ function (`calc_pizza_price`). Notice that we pass `const Pizza&` and `const PriceList&` by **const reference** to avoid unnecessary copying while guaranteeing the function cannot accidentally modify the inputs:

```cpp
double calc_pizza_price(const Pizza& ingredients, const PriceList& price_list) {
    double price = price_list.flour * ingredients.flour +
                   price_list.tomatoes * ingredients.tomatoes +
                   price_list.mozzarella * ingredients.mozzarella +
                   price_list.salt * ingredients.salt +
                   price_list.yeast * ingredients.yeast +
                   price_list.delivery * ingredients.delivery +
                   price_list.fixed_costs;
    return price;
}
```

Test the function on `pizza_A`, and create two additional pizzas (`pizza_B` and `pizza_C`) to inspect their prices:

```cpp
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Pizza A costs " << calc_pizza_price(pizza_A, pizza_ingredients_price_list) << " EUR\n";

    Pizza pizza_B{0.250, 3, 0.200, 0.005, 0.010, 0};
    Pizza pizza_C{0.350, 5, 0.400, 0.008, 0.020, 1};

    double price_B = calc_pizza_price(pizza_B, pizza_ingredients_price_list);
    double price_C = calc_pizza_price(pizza_C, pizza_ingredients_price_list);

    std::cout << "Pizza B costs " << price_B << " EUR\n";
    std::cout << "Pizza C costs " << price_C << " EUR\n";
```

### 2.3 Random Number Generation in C++ (`<random>`)
Generating $1,000$ pizzas by hand would be far too tedious. In Python, we used `random.uniform(a, b)` and `random.randint(a, b)`.

In Modern C++, we avoid the legacy C `rand()` function and instead use the `<random>` header, which separates the **random engine** from the **statistical distribution**:
1. **Engine (`std::mt19937`):** A high-quality Mersenne Twister pseudo-random number generator, seeded once via `std::random_device` (or a fixed integer seed like `42` when you want reproducible results).
2. **Distributions:**
   * `std::uniform_real_distribution<double>(min, max)` — equivalent to Python's `uniform(min, max)` over $[min, max)$.
   * `std::uniform_int_distribution<int>(min, max)` — equivalent to Python's `randint(min, max)` over the inclusive integer interval $[min, max]$.

```cpp
// Example usage:
std::random_device rd;
std::mt19937 gen(rd()); // Seed the Mersenne Twister engine once
std::uniform_real_distribution<double> dist(5.0, 10.0);
double sample = dist(gen); // Produces a random double in [5.0, 10.0)
```

**Task 2:** Implement `random_pizza_generator(std::mt19937& gen)` so that it returns a `Pizza` struct populated with random quantities within realistic intervals. Make sure `tomatoes` and `delivery` are generated as integers (using `std::uniform_int_distribution<int>` rather than rounding floats!).

```cpp
Pizza random_pizza_generator(std::mt19937& gen) {
    // TODO: Define appropriate distributions for each ingredient and return a random Pizza
    std::uniform_real_distribution<double> flour_dist(0.1, 0.4);
    std::uniform_int_distribution<int> tomatoes_dist(0, 10);
    std::uniform_real_distribution<double> mozzarella_dist(0.0, 0.5);
    std::uniform_real_distribution<double> salt_dist(0.001, 0.01);
    std::uniform_real_distribution<double> yeast_dist(0.005, 0.05);
    std::uniform_int_distribution<int> delivery_dist(0, 1);

    return Pizza{
        flour_dist(gen),
        tomatoes_dist(gen),
        mozzarella_dist(gen),
        salt_dist(gen),
        yeast_dist(gen),
        delivery_dist(gen)
    };
}
```

Test your generator with a loop over $10$ random pizzas:

```cpp
    int number_of_pizzas = 10;
    for (int idx = 0; idx < number_of_pizzas; ++idx) {
        Pizza random_pizza = random_pizza_generator(gen);
        double pizza_price = calc_pizza_price(random_pizza, pizza_ingredients_price_list);
        std::cout << idx << ": flour=" << random_pizza.flour
                  << ", tomatoes=" << random_pizza.tomatoes
                  << " -> price=" << pizza_price << " EUR\n";
    }
```

### 2.4 Simulating Real-World Noise
In real life, pizza prices fluctuate throughout the year due to ingredient supply changes, discounts, or special requests. Real-world machine learning datasets **always** contain noise. We will simulate this by randomly scaling each pizza's calculated price by up to $\pm 10\%$ (multiplying by a random factor $\epsilon \sim \mathcal{U}(0.9, 1.1)$) and rounding to two decimal places:

$$y_{\text{noisy}} = \frac{\text{round}(100 \cdot y \cdot \epsilon)}{100}, \quad \epsilon \sim \mathcal{U}(0.9, 1.1)$$

**Task 3:** Implement `calc_noisy_pizza_price`:
1. Receive `const Pizza& ingredients`, `const PriceList& price_list`, and `std::mt19937& gen`.
2. Call `calc_pizza_price` to compute `original_price`.
3. Draw a random multiplier in $[0.9, 1.1]$ using `std::uniform_real_distribution<double>`.
4. Multiply `original_price` by the noise factor and round to two decimal places using `std::round(price * 100.0) / 100.0`.
5. Return the noisy price.

```cpp
double calc_noisy_pizza_price(const Pizza& ingredients,
                              const PriceList& price_list,
                              std::mt19937& gen) {
    // TODO: Implement noisy price calculation
    return 0.0;
}
```

### 2.5 Generating 1,000 Pizzas & C++ Alternatives to List Comprehensions
In Python, you can generate a list of $1,000$ pizzas using either a `for` loop with `.append()` or a **list comprehension** (`[random_pizza_generator() for i in range(1000)]`).

In C++, we store dynamic arrays in `std::vector<T>`:
* **Standard `for` loop with `.reserve()` and `.push_back()`:** Pre-allocating memory with `pizzas.reserve(1000)` prevents repeated memory reallocations as the vector grows.
* **Functional style with `std::generate_n` / `std::transform`:** C++'s `<algorithm>` header provides `std::transform`, which applies a lambda function to every element of an input range and writes the result into an output vector—serving as C++'s direct equivalent to a list comprehension!

```cpp
    // 1. Generate 1,000 random pizzas
    std::vector<Pizza> pizzas;
    pizzas.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        pizzas.push_back(random_pizza_generator(gen));
    }

    // 2. Calculate prices using a standard loop
    std::vector<double> prices;
    prices.reserve(pizzas.size());
    for (const auto& pizza : pizzas) {
        prices.push_back(calc_noisy_pizza_price(pizza, pizza_ingredients_price_list, gen));
    }
```

**Task 4:** Rewrite the `prices` calculation above using `std::transform` and a C++ lambda expression (mirroring Python's list comprehension):

```cpp
    std::vector<double> prices_functional(pizzas.size());
    // TODO: Use std::transform(pizzas.begin(), pizzas.end(), prices_functional.begin(), [&](const Pizza& p) { ... });
```

---

## 3. Using OpenCV (`cv::Mat`) & C++ STL to Process Tabular Data

In the Python course, `pandas.DataFrame` is used to inspect, slice, filter, transform, and save tabular datasets. In our C++ stack, **`cv::Mat`** (configured with 64-bit floating point type `CV_64F`) serves as our core 2D numerical table/matrix engine—giving us fast row/column slicing, boolean masks, and vectorized linear algebra without needing any external DataFrame dependencies.

### 3.1 Converting `std::vector<Pizza>` into a `cv::Mat`
Let's convert our `1,000` pizzas into a $1000 \times 6$ matrix (`pizzas_mat`) and keep track of our column names in a `std::vector<std::string>`:

```cpp
cv::Mat pizzas_to_mat(const std::vector<Pizza>& pizzas) {
    // 1000 rows, 6 feature columns, 64-bit double precision (CV_64F)
    cv::Mat mat(static_cast<int>(pizzas.size()), 6, CV_64F);

    for (int i = 0; i < static_cast<int>(pizzas.size()); ++i) {
        mat.at<double>(i, 0) = pizzas[i].flour;
        mat.at<double>(i, 1) = static_cast<double>(pizzas[i].tomatoes);
        mat.at<double>(i, 2) = pizzas[i].mozzarella;
        mat.at<double>(i, 3) = pizzas[i].salt;
        mat.at<double>(i, 4) = pizzas[i].yeast;
        mat.at<double>(i, 5) = static_cast<double>(pizzas[i].delivery);
    }
    return mat;
}
```

In `main()`:
```cpp
    std::vector<std::string> feature_names = {
        "flour", "tomatoes", "mozzarella", "salt", "yeast", "delivery"
    };
    cv::Mat pizzas_mat = pizzas_to_mat(pizzas);
    std::cout << "Matrix dimensions: " << pizzas_mat.rows << " x " << pizzas_mat.cols << "\n";
```

> **Important OpenCV Memory Rule:** In OpenCV, `.col(j)`, `.row(i)`, `.colRange(a, b)`, and `.rowRange(a, b)` return **lightweight views (headers)** pointing to the *same underlying memory* as `pizzas_mat`—they do **not** copy data! If you want an independent copy that will not mutate `pizzas_mat`, always append `.clone()` (e.g., `pizzas_mat.col(1).clone()`).

### 3.2 Selecting Columns & Rows (Equivalent to `df[col]` and `df.iloc[...]`)

* **Select specific columns:**
  ```cpp
  // Equivalent to: pizzas_df["tomatoes"] (column index 1)
  cv::Mat tomatoes_col = pizzas_mat.col(1);

  // Equivalent to: pizzas_df[["tomatoes", "mozzarella"]] (columns 1 and 2; end index is exclusive)
  cv::Mat tom_and_mozz = pizzas_mat.colRange(1, 3);
  ```

* **Select specific rows:**
  ```cpp
  // Equivalent to: pizzas_df.iloc[0]
  cv::Mat first_pizza = pizzas_mat.row(0);
  std::cout << "First pizza row: " << first_pizza << "\n";

  // Equivalent to: pizzas_df.iloc[0:10]
  cv::Mat first_ten_pizzas = pizzas_mat.rowRange(0, 10);
  ```

### 3.3 Conditional Filtering (Boolean Masks in OpenCV)
In pandas, you filter rows with boolean expressions:
```python
pizzas_df[(pizzas_df["tomatoes"] > 5) & (pizzas_df["mozzarella"] < 0.25)]
```
In OpenCV, comparison operators on `cv::Mat` produce an 8-bit mask (`CV_8U`) where matching entries are `255` and non-matching entries are `0`. You can combine masks with bitwise `&` and collect matching rows:

```cpp
    // Equivalent to: (pizzas_df["tomatoes"] > 5) & (pizzas_df["mozzarella"] < 0.25)
    cv::Mat mask = (pizzas_mat.col(1) > 5.0) & (pizzas_mat.col(2) < 0.25);

    cv::Mat filtered_pizzas;
    for (int r = 0; r < pizzas_mat.rows; ++r) {
        if (mask.at<uchar>(r, 0) > 0) {
            filtered_pizzas.push_back(pizzas_mat.row(r));
        }
    }
    std::cout << "Pizzas with >5 tomatoes and <0.25kg mozzarella: "
              << filtered_pizzas.rows << "\n";
```

### 3.4 Vectorized Mathematical Operations
Just like NumPy and pandas, `cv::Mat` supports element-wise arithmetic directly on columns, rows, or the entire matrix:

```cpp
    // Equivalent to: pizzas_df["flour"] * 2
    cv::Mat double_flour = pizzas_mat.col(0) * 2.0;

    // Equivalent to: pizzas_df.iloc[0] / 5
    cv::Mat scaled_first_row = pizzas_mat.row(0) / 5.0;

    // Equivalent to: pizzas_df + 100
    cv::Mat plus_hundred = pizzas_mat + 100.0;
```

### 3.5 Appending the `price` Column (`cv::hconcat`)
Right now, `pizzas_mat` has $6$ columns (`flour` through `delivery`), and `prices` is a `std::vector<double>` of length $1,000$. Let's convert `prices` into a $1000 \times 1$ `cv::Mat` and horizontally concatenate it onto `pizzas_mat` using `cv::hconcat`:

```cpp
    // Convert std::vector<double> to a 1000 x 1 CV_64F column matrix (clone to own memory)
    cv::Mat prices_col = cv::Mat(prices, true);

    // Combine features (1000 x 6) and target prices (1000 x 1) -> dataset_mat (1000 x 7)
    cv::Mat dataset_mat;
    cv::hconcat(pizzas_mat, prices_col, dataset_mat);

    std::vector<std::string> all_columns = feature_names;
    all_columns.push_back("price");

    std::cout << "Full dataset matrix: " << dataset_mat.rows << " x " << dataset_mat.cols << "\n";
```

### 3.6 Saving & Loading `my_pizzas.csv`
Finally, let's write clean C++ `<fstream>` functions to export `dataset_mat` to `my_pizzas.csv` (matching `pizzas_df.to_csv("my_pizzas.csv", index=False)`) and read it back (`pd.read_csv("my_pizzas.csv")`):

```cpp
void save_to_csv(const std::string& filename,
                 const cv::Mat& data,
                 const std::vector<std::string>& headers) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // 1. Write header row
    for (size_t c = 0; c < headers.size(); ++c) {
        file << headers[c] << (c + 1 < headers.size() ? "," : "\n");
    }

    // 2. Write data rows
    file << std::fixed << std::setprecision(6);
    for (int r = 0; r < data.rows; ++r) {
        for (int c = 0; c < data.cols; ++c) {
            file << data.at<double>(r, c) << (c + 1 < data.cols ? "," : "\n");
        }
    }
}

cv::Mat load_from_csv(const std::string& filename, std::vector<std::string>& out_headers) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    out_headers.clear();
    std::string line;

    // 1. Parse header row
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string col_name;
        while (std::getline(ss, col_name, ',')) {
            out_headers.push_back(col_name);
        }
    }

    // 2. Parse numeric rows
    std::vector<std::vector<double>> rows_data;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        rows_data.push_back(row);
    }

    int num_rows = static_cast<int>(rows_data.size());
    int num_cols = num_rows > 0 ? static_cast<int>(rows_data[0].size()) : 0;
    cv::Mat loaded_mat(num_rows, num_cols, CV_64F);

    for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
            loaded_mat.at<double>(r, c) = rows_data[r][c];
        }
    }
    return loaded_mat;
}
```

In `main()`, export your dataset and verify that loading it back succeeds:

```cpp
    save_to_csv("my_pizzas.csv", dataset_mat, all_columns);
    std::cout << "Saved 1,000 pizzas to my_pizzas.csv!\n";

    std::vector<std::string> loaded_headers;
    cv::Mat loaded_df = load_from_csv("my_pizzas.csv", loaded_headers);
    std::cout << "Reloaded CSV shape: " << loaded_df.rows << " x " << loaded_df.cols << "\n";
    std::cout << "First reloaded row: " << loaded_df.row(0) << "\n";
```

You can inspect the generated file directly from your terminal using `head -n 10 my_pizzas.csv` (Linux/macOS) or `Get-Content my_pizzas.csv -TotalCount 10` (Windows PowerShell).

---

## 4. What's Next?

Now we have data! Why is this so important for machine learning? Because machine learning is *all* about discovering patterns in data.

In this lesson, we started with a known set of ground-truth parameters—our `pizza_ingredients_price_list`—and generated a noisy dataset (`my_pizzas.csv`) of $1,000$ pizzas and their prices. **In the upcoming lessons, we will pretend we do not know `pizza_ingredients_price_list` and implement machine learning algorithms from scratch in C++ and `cv::Mat` to reverse-engineer those exact ingredient prices directly from `my_pizzas.csv`!**

---

## 5. Reference Solutions

### Solution 1: Realistic Price List
```cpp
PriceList pizza_ingredients_price_list{
    1.20,  // flour (€/kg)
    0.50,  // tomatoes (€/tomato)
    8.50,  // mozzarella (€/kg)
    0.80,  // salt (€/kg)
    5.00,  // yeast (€/kg)
    2.50,  // delivery (€/delivery)
    4.00   // fixed_costs (€/pizza)
};
```

### Solution 2: `random_pizza_generator`
```cpp
Pizza random_pizza_generator(std::mt19937& gen) {
    std::uniform_real_distribution<double> flour_dist(0.1, 0.4);
    std::uniform_int_distribution<int> tomatoes_dist(0, 10);
    std::uniform_real_distribution<double> mozzarella_dist(0.0, 0.5);
    std::uniform_real_distribution<double> salt_dist(0.001, 0.01);
    std::uniform_real_distribution<double> yeast_dist(0.005, 0.05);
    std::uniform_int_distribution<int> delivery_dist(0, 1);

    return Pizza{
        flour_dist(gen),
        tomatoes_dist(gen),
        mozzarella_dist(gen),
        salt_dist(gen),
        yeast_dist(gen),
        delivery_dist(gen)
    };
}
```

### Solution 3: `calc_noisy_pizza_price`
```cpp
double calc_noisy_pizza_price(const Pizza& ingredients,
                              const PriceList& price_list,
                              std::mt19937& gen) {
    double original_price = calc_pizza_price(ingredients, price_list);

    // Simulate price fluctuations by randomly scaling by +/- 10%
    std::uniform_real_distribution<double> noise_dist(0.9, 1.1);
    double noise = noise_dist(gen);
    double price = original_price * noise;

    // Round to 2 decimal places
    price = std::round(price * 100.0) / 100.0;
    return price;
}
```

### Solution 4: Calculating Prices with `std::transform` (List Comprehension Equivalent)
```cpp
std::vector<double> prices(pizzas.size());
std::transform(
    pizzas.begin(),
    pizzas.end(),
    prices.begin(),
    [&](const Pizza& pizza) {
        return calc_noisy_pizza_price(pizza, pizza_ingredients_price_list, gen);
    }
);
```