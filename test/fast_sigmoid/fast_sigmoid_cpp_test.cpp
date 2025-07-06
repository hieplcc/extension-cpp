#include <iostream>
#include <torch/torch.h>
#include <chrono>
#include <extension_cpp/fast_sigmoid.h> 

void test_fast_sigmoid() {
    auto x = torch::linspace(-15, 15, 10, torch::dtype(torch::kFloat32));
    std::cout << "Input: " << x << std::endl;

    auto y = extension_cpp::fast_sigmoid(x);
    std::cout << "Fast Sigmoid: " << y << std::endl;

    auto expected = torch::sigmoid(x);
    std::cout << "Torch Sigmoid: " << expected << std::endl;
}

void test_lookuptable_cache() {
    auto input = torch::linspace(-15, 15, 10, torch::dtype(torch::kFloat32));

    // Cache miss
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto output1 = extension_cpp::fast_sigmoid(input, -20.0, 20.0, 1000);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Elapsed time (cache miss): " << elapsed.count() << std::endl;
    }

    // Cache hit
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto output2 = extension_cpp::fast_sigmoid(input, -20.0, 20.0, 1000);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Elapsed time (cache hit): " << elapsed.count() << std::endl;
    }
}

int main() {
    test_fast_sigmoid();
    test_lookuptable_cache();
    return 0;
}

