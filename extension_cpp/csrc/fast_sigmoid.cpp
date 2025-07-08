#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <shared_mutex>
#include <map>
#include <iostream>
#include "../../utils/lru_cache.h"

// #define USE_LRU_CACHE

#if defined(__has_cpp_attribute)
#  if __has_cpp_attribute(likely) && __cplusplus >= 202002L
#    define LIKELY [[likely]]
#    define UNLIKELY [[unlikely]]
#  else
#    define LIKELY
#    define UNLIKELY
#  endif
#else
#  define LIKELY
#  define UNLIKELY
#endif

namespace extension_cpp {

using LookupTableKey = std::tuple<float, float, int64_t>;   // Key: a tuple of (min_val, max_val, num_entries)
using LookupTableValue = std::pair<at::Tensor, at::Tensor>; // Value: a pair of tensors (x_vals, y_vals)

#ifdef USE_LRU_CACHE

static LRUCache<LookupTableKey, LookupTableValue> lookup_table_lru_cache(20); 
std::pair<at::Tensor, at::Tensor> get_or_create_lookup_table(float min_val, float max_val, int64_t num_entries) {
    LookupTableKey key{min_val, max_val, num_entries};
    LookupTableValue value;

    if (lookup_table_lru_cache.get(key, value)) {
        return value;
    }

    at::Tensor x_vals = torch::linspace(min_val, max_val, num_entries, torch::dtype(torch::kFloat32).device(torch::kCPU));
    at::Tensor y_vals = torch::sigmoid(x_vals);
    value = std::make_pair(x_vals, y_vals);

    lookup_table_lru_cache.put(key, value);
    return value;
}

#else

static std::map<LookupTableKey, LookupTableValue> lookup_table_cache;

// shared_mutex for multiple readers and single writer (read operations normally outnumber writes)
static std::shared_mutex lookup_table_mutex; 

//Get the cached lookup table or create a new if it doesn't exist
std::pair<at::Tensor, at::Tensor> get_or_create_lookup_table(float min_val, float max_val, int64_t num_entries) {
    LookupTableKey key{min_val, max_val, num_entries};
    
    // Read from cache with shared lock (allows multiple readers)
    {
        std::shared_lock<std::shared_mutex> read_lock(lookup_table_mutex);
        auto it = lookup_table_cache.find(key);
        if (it != lookup_table_cache.end()) {
            // std::cout << "Cache hit for key: (" << min_val << ", " << max_val << ", " << num_entries << ")" << std::endl;
            return it->second;
        }
    }

    // std::cout << "Cache miss for key: (" << min_val << ", " << max_val << ", " << num_entries << ")" << std::endl;  
    // Cache miss - acquire exclusive lock (only one writer at a time)
    std::unique_lock<std::shared_mutex> write_lock(lookup_table_mutex);
    
    // Double-check pattern: another thread might have created it while we waited
    auto it = lookup_table_cache.find(key);
    if (it != lookup_table_cache.end()) {

        return it->second;
    }

    // Create new lookup table
    at::Tensor x_vals = torch::linspace(min_val, max_val, num_entries, torch::dtype(torch::kFloat32).device(torch::kCPU));
    at::Tensor y_vals = torch::sigmoid(x_vals);
    
    lookup_table_cache[key] = std::make_pair(x_vals, y_vals);
    return std::make_pair(x_vals, y_vals);
}

#endif

// Forward computation
at::Tensor fast_sigmoid_cpu(const at::Tensor& input, double min_val, double max_val, int64_t num_entries) {
    TORCH_CHECK(input.dtype() == torch::kFloat32);
    TORCH_CHECK(min_val < max_val);
    TORCH_CHECK(num_entries > 1);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CPU);

    const auto& [x_vals, y_vals] = get_or_create_lookup_table(min_val, max_val, num_entries);
    
    at::Tensor input_contig = input.contiguous();
    at::Tensor output = torch::empty(input_contig.sizes(), input_contig.options());
    
    const float* input_ptr = input_contig.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* y_vals_ptr = y_vals.data_ptr<float>();
    
    const float scale = static_cast<float>(num_entries - 1) / (max_val - min_val);
    
    for (int64_t i = 0; i < input_contig.numel(); i++) {
        float x = input_ptr[i];
        
        if (x <= min_val) UNLIKELY { 
            output_ptr[i] = y_vals_ptr[0];
        } else if (x >= max_val) UNLIKELY {
            output_ptr[i] = y_vals_ptr[num_entries - 1];
        } else LIKELY {
            float idx_f = (x - min_val) * scale;
            int64_t idx = static_cast<int64_t>(std::floor(idx_f));

            // Make sure that lower is less than num_entries - 1. Otherwise, upper (lower + 1) will be out of bounds
            // (num_entries - 2 >= 0 due to num_entries > 1 check)
            int64_t lower = std::min(idx, num_entries - 2);
            int64_t upper = lower + 1;
            
            float alpha = idx_f - static_cast<float>(lower);
            float y1 = y_vals_ptr[lower];
            float y2 = y_vals_ptr[upper];
            
            output_ptr[i] = y1 + alpha * (y2 - y1);
        }
    }
    
    return output;
}

// Backward computation
at::Tensor fast_sigmoid_backward_cpu(const at::Tensor& grad_output, const at::Tensor& input, 
                                   double min_val, double max_val, int64_t num_entries) {
    TORCH_CHECK(input.dtype() == torch::kFloat32);
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32);
    TORCH_CHECK(input.sizes() == grad_output.sizes());
    TORCH_CHECK(min_val < max_val);
    TORCH_CHECK(num_entries > 1);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(grad_output.device().type() == at::DeviceType::CPU);
    
    const auto& [x_vals, y_vals] = get_or_create_lookup_table(min_val, max_val, num_entries);
    
    at::Tensor input_contig = input.contiguous();
    at::Tensor grad_contig = grad_output.contiguous();
    at::Tensor grad_input = torch::zeros_like(input_contig);
    
    const float* input_ptr = input_contig.data_ptr<float>();
    const float* grad_ptr = grad_contig.data_ptr<float>();
    float* grad_input_ptr = grad_input.data_ptr<float>();
    const float* x_vals_ptr = x_vals.data_ptr<float>();
    const float* y_vals_ptr = y_vals.data_ptr<float>();
    
    const float scale = static_cast<float>(num_entries - 1) / (max_val - min_val);
    
    for (int64_t i = 0; i < input_contig.numel(); i++) {
        float x = input_ptr[i];
        
        if (x <= min_val || x >= max_val) UNLIKELY {
            grad_input_ptr[i] = 0.0f;
        } else LIKELY {
            float idx_f = (x - min_val) * scale;
            
            // Make sure that idx is less than num_entries - 1. Otherwise, (idx + 1) will be out of bounds
            // (num_entries - 2 always >= 0 due to num_entries > 1 check)
            int64_t idx = std::min(static_cast<int64_t>(std::floor(idx_f)), num_entries - 2);
            
            float x0 = x_vals_ptr[idx];
            float x1 = x_vals_ptr[idx + 1];
            float y0 = y_vals_ptr[idx];
            float y1 = y_vals_ptr[idx + 1];
            
            float local_grad = (y1 - y0) / (x1 - x0);
            grad_input_ptr[i] = grad_ptr[i] * local_grad;
        }
    }
    
    return grad_input;
}

torch::Tensor fast_sigmoid(const torch::Tensor& input, double min_val, double max_val, int64_t num_entries) {
    return fast_sigmoid_cpu(input, min_val, max_val, num_entries);
}

// Register the operators with PyTorch
TORCH_LIBRARY_FRAGMENT(extension_cpp, m) {
    m.def("fast_sigmoid(Tensor input, float min_val, float max_val, int num_entries) -> Tensor");
    m.def("fast_sigmoid_backward(Tensor grad_output, Tensor input, float min_val, float max_val, int num_entries) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
    m.impl("fast_sigmoid", &extension_cpp::fast_sigmoid_cpu);
    m.impl("fast_sigmoid_backward", &extension_cpp::fast_sigmoid_backward_cpu);
}

} 

