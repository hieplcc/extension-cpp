#include <torch/torch.h>
#include <map>
#include <tuple>
#include <cmath>

namespace extension_cpp {

using LookupTableKey = std::tuple<float, float, int64_t>;
static std::map<LookupTableKey, std::pair<at::Tensor, at::Tensor>> lookup_table_cache;

//Get the cached lookup table or create a new if it doesn't exist
std::pair<at::Tensor, at::Tensor> get_or_create_lookup_table(float min_val, float max_val, int64_t num_entries) {
    LookupTableKey key{min_val, max_val, num_entries};
    
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

// Forward computation
at::Tensor fast_sigmoid_cpu(const at::Tensor& input, float min_val, float max_val, int64_t num_entries) {
    TORCH_CHECK(input.dtype() == torch::kFloat32);
    TORCH_CHECK(min_val < max_val);
    TORCH_CHECK(num_entries > 1);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CPU);
    
    auto [x_vals, y_vals] = get_or_create_lookup_table(min_val, max_val, num_entries);
    
    at::Tensor input_contig = input.contiguous();
    at::Tensor output = torch::empty(input_contig.sizes(), input_contig.options());
    
    const float* input_ptr = input_contig.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    const float* y_vals_ptr = y_vals.data_ptr<float>();
    
    const float scale = static_cast<float>(num_entries - 1) / (max_val - min_val);
    
    for (int64_t i = 0; i < input_contig.numel(); i++) {
        float x = input_ptr[i];
        
        if (x <= min_val) [[unlikely]] { 
            output_ptr[i] = y_vals_ptr[0];
        } else if (x >= max_val) [[unlikely]] {
            output_ptr[i] = y_vals_ptr[num_entries - 1];
        } else [[likely]] {
            float idx_f = (x - min_val) * scale;
            int64_t lower = static_cast<int64_t>(std::floor(idx_f));
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
                                   float min_val, float max_val, int64_t num_entries) {
    TORCH_CHECK(input.dtype() == torch::kFloat32);
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32);
    TORCH_CHECK(input.sizes() == grad_output.sizes());
    TORCH_CHECK(min_val < max_val);
    TORCH_CHECK(num_entries > 1);
    TORCH_INTERNAL_ASSERT(input.device().type() == at::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(grad_output.device().type() == at::DeviceType::CPU);
    
    auto [x_vals, y_vals] = get_or_create_lookup_table(min_val, max_val, num_entries);
    
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
        
        if (x <= min_val || x >= max_val) [[unlikely]] {
            grad_input_ptr[i] = 0.0f;
        } else [[likely]] {
            float idx_f = (x - min_val) * scale;
            int64_t idx = static_cast<int64_t>(std::floor(idx_f));
            
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

torch::Tensor fast_sigmoid(const torch::Tensor& input, float min_val, float max_val, int64_t num_entries) {
    return fast_sigmoid_cpu(input, min_val, max_val, num_entries);
}

} 
