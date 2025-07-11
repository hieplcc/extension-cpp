#include <iostream>
#include <torch/torch.h>
#include <chrono>
#include <extension_cpp/fast_sigmoid.h> 

void test_fast_sigmoid() {
    std::cout << "====Testing fast sigmoid...===="<< std::endl;
    auto x = torch::linspace(-15, 15, 10, torch::dtype(torch::kFloat32));
    std::cout << "Input: " << x << std::endl;

    auto y = extension_cpp::fast_sigmoid(x);
    std::cout << "Fast Sigmoid: " << y << std::endl;

    auto expected = torch::sigmoid(x);
    std::cout << "Torch Sigmoid: " << expected << std::endl;
    std::cout << std::endl;
}

void test_lookuptable_cache() {
    std::cout << "====Testing lookup table cache...====" << std::endl;
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
    std::cout << std::endl;
}

void test_floating_point_rounding() {
    std::cout << "====Testing floating point rounding...====" << std::endl;
    double min_val = -10.0f;
    double max_val = 10.0f;
    int64_t num_entries = 1000;
    float x = 9.9999990463f;

    const float min_val_f = static_cast<float>(min_val);
    const float max_val_f = static_cast<float>(max_val);

    const float scale = static_cast<float>(num_entries - 1) / (max_val_f - min_val_f);

    if (x <= min_val_f) { 
        std::cout << "Value is below min_val: " << x << std::endl;
    } else if (x >= max_val_f) {
        std::cout << "Value is above max_val: " << x << std::endl;
    } else {
        std::cout << "Value is within range: " << x << std::endl;
        float idx_f = (x - min_val_f) * scale;
        int64_t idx = static_cast<int64_t>(std::floor(idx_f));
        std::cout << "Index: " << idx << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    test_fast_sigmoid();
    test_lookuptable_cache();
    test_floating_point_rounding();
    return 0;
}

