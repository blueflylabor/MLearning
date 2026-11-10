#include<iostream>
#include<vector>
#include<chrono>

constexpr int ROWS = 4096;
constexpr int COLS = 4096;

int main(){
    std::vector<float> mat(ROWS * COLS, 1.0f);
    auto start_row = std::chrono::high_resolution_clock::now();
    double sum_row = 0.0;
    for(int r=0; r < ROWS; ++r){
        for(int c=0; c < COLS; ++c){
            sum_row += mat[r * COLS + c];
        }
    }
    auto end_row = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_row = end_row - start_row;
    auto start_col = std::chrono::high_resolution_clock::now();
    double sum_col = 0.0;
    for(int c = 0; c < COLS; ++c){
        for(int r = 0; r < ROWS; ++r){
            sum_col += mat[r * COLS + c];
        }
    }
    auto end_col = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_col = end_col - start_col;
    std::cout << "Mat size: " << ROWS << "*" << COLS << ":"<< (ROWS * COLS * 4) / (1024 * 1024) << "MB" << std::endl;
    std::cout << duration_row.count() << "ms" << std::endl;
    std::cout << duration_col.count() << "ms" << std::endl;

    return 0;
}