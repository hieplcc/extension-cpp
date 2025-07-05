#pragma once
#include <torch/torch.h>

namespace extension_cpp {

//swe_cpp_hometest.md: For simplicity, assume `fast_sigmoid` runs on CPU and accepts float32 input only.
TORCH_API torch::Tensor fast_sigmoid( 
    const torch::Tensor& input,
    double min_val = -10.0, 
    double max_val = 10.0, 
    int64_t num_entries = 1000
);

} // namespace extension_cpp
