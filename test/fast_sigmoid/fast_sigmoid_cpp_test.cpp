#include <iostream>
#include <extension_cpp/fast_sigmoid.h>

int main() {
    auto x = torch::linspace(-15, 15, 10);
    std::cout << "Input: " << x << std::endl;
    
    // Linespace can return tensors of different dtypes?
    if (x.dtype() != torch::kFloat32) {
        x = x.to(torch::kFloat32);
    }
    
    auto y = extension_cpp::fast_sigmoid(x);
    std::cout << "Fast Sigmoid: " << y << std::endl;
    
    return 0;
}
