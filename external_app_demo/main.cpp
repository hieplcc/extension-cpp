#include <iostream>
#include <torch/torch.h>
#include <chrono>
#include <extension_cpp/fast_sigmoid.h> 

int main() {
    auto x = torch::linspace(-15, 15, 10, torch::dtype(torch::kFloat32));
    std::cout << "Input: " << x << std::endl;

    auto y = extension_cpp::fast_sigmoid(x);
    std::cout << "Fast Sigmoid: " << y << std::endl;

    auto expected = torch::sigmoid(x);
    std::cout << "Torch Sigmoid: " << expected << std::endl;
    return 0;
}

