#include "tensor.h"


void test1();
void test2();

// 使用示例
int main() {
    Tensor t1({1, 3, 4}, 2.);
    t1.print();


    return 0;
}


void test1(){
    // 创建一个 2×3×4 的 3D 张量
    Tensor t({2, 3, 2});
    for (size_t i = 0; i < t.numel(); ++i) t.data[i] = i;
    t.print();                          // Shape: (2, 3, 4)
    // 高维 -> 低维
    auto t2 = t.reshape({3, 4});
    t2.print();                         // Shape: (6, 4)
    // 完全展平
    auto t1 = t.flatten();
    t1.print();                         // Shape: (24)
    // 升维
    auto t4 = t.unsqueeze(0);           // Shape: (1, 2, 3, 4)
    t4.print();
    // 访问：第1层，第2行，第3列
    std::cout << "t.at({1,2,1}) = " << t.at({1, 2, 1}) << std::endl;

    Tensor tensor({2, 3, 2, 2});
    for (size_t i = 0; i < tensor.numel(); ++i) tensor.data[i]= .1;
    tensor.print();

    Tensor tensor2({2, 3, 2, 2}, 3.1);
    Tensor tensor3 = tensor + tensor2;
    tensor3.print();
}

void test2(){
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(10);
    oss << double(1) << 3.14159265535 << 3.14 << 3.14159;
    std::cout << oss.str();
    
}
