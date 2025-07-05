import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest

import pytest
from extension_cpp.ops import fast_sigmoid


class TestMyMulAdd(TestCase):
    def test_fast_sigmoid_basic(self):
        """Test basic fast_sigmoid functionality"""
        x = torch.randn(10, dtype=torch.float32)
        result = fast_sigmoid(x)
        
        # Check output is in valid sigmoid range [0, 1]
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)

        assert result.dtype == torch.float32

    def test_fast_sigmoid_vs_torch_sigmoid(self):
        """Test fast_sigmoid approximation quality against torch.sigmoid"""
        x = torch.linspace(-10, 10, 100, dtype=torch.float32)
        
        fast_result = fast_sigmoid(x)
        torch_result = torch.sigmoid(x)
        
        relative_error = torch.abs(fast_result - torch_result) / (torch_result + 1e-8)
        mean_error = torch.mean(relative_error)
        
        print(f"Testing fast_sigmoid against torch.sigmoid - mean relative error: {mean_error:.6f}" )
        assert mean_error < 0.01  # Less than 1% error on average

    def test_fast_sigmoid_torch_compile(self):
        """Test compatibility with torch.compile"""
        if hasattr(torch, 'compile'):
            @torch.compile
            def compiled_fast_sigmoid(x):
                return fast_sigmoid(x, min_val=-10.0, max_val=10.0, num_entries=1024)
            
            x = torch.randn(10, dtype=torch.float32)
            result = compiled_fast_sigmoid(x)
            
            assert torch.all(result >= 0.0)
            assert torch.all(result <= 1.0)

if __name__ == "__main__":
    unittest.main()