import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest

import pytest
from extension_cpp.ops import fast_sigmoid

class TestFastSigmoid(TestCase):
    def sample_inputs(self, device, *, requires_grad=False):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=requires_grad)

        def make_nondiff_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3)],
            [make_tensor(20)],
            [make_tensor(32)],
            [make_nondiff_tensor(2, 3)],
        ]
    
    def test_op_check(self):
        samples = self.sample_inputs('cpu', requires_grad=True)
        for args in samples:
            opcheck(
                torch.ops.extension_cpp.fast_sigmoid.default,
                args,
                kwargs={"min_val": -10.0, "max_val": 10.0, "num_entries": 1000}
            )

    def test_fast_sigmoid_basic(self):
        """Test basic fast_sigmoid functionality"""
        x = torch.randn(10, dtype=torch.float32)
        result = fast_sigmoid(x)
        
        # Check output is in valid sigmoid range [0, 1]
        assert torch.all(result >= 0.0)
        assert torch.all(result <= 1.0)
        assert result.dtype == torch.float32

    def test_fast_sigmoid_correctness(self):
        """Test fast_sigmoid approximation quality against torch.sigmoid"""
        x = torch.linspace(-10, 10, 100, dtype=torch.float32)
        
        fast_result = fast_sigmoid(x)
        torch_result = torch.sigmoid(x)
        
        #Less than 1% relative error
        assert torch.allclose(fast_result, torch_result, rtol=1e-2, atol=1e-4)

    def test_invalid_range_or_entries(self):
        """Invalid min/max or num_entries"""
        x = torch.randn(10, dtype=torch.float32)
        with pytest.raises(RuntimeError):
            fast_sigmoid(x, min_val=10.0, max_val=-10.0, num_entries=100)
        with pytest.raises(RuntimeError):
            fast_sigmoid(x, min_val=-10.0, max_val=10.0, num_entries=1)

    def test_fast_sigmoid_extreme_inputs(self):
        """Test clamping behavior at min and max"""
        x = torch.tensor([-100.0, 0.0, 100.0], dtype=torch.float32)
        result = fast_sigmoid(x, min_val=-10.0, max_val=10.0, num_entries=1000)
        assert result[0] == result.min()
        assert result[2] == result.max()

    def test_fast_sigmoid_backward(self):
        """Test autograd with fast_sigmoid using .backward()"""
        x = torch.randn(10, dtype=torch.float32, requires_grad=True)
        y = fast_sigmoid(x)
        y.sum().backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.all(torch.isfinite(x.grad))

    def test_fast_sigmoid_torch_compile(self):
        """Test compatibility with torch.compile"""
        if hasattr(torch, 'compile'):
            @torch.compile
            def compiled_fast_sigmoid(x):
                return fast_sigmoid(x, min_val=-10.0, max_val=10.0, num_entries=1024)
            
            x = torch.linspace(-10, 10, 100, dtype=torch.float32)
            fast_result = compiled_fast_sigmoid(x)
            torch_result = torch.sigmoid(x)
            
            assert torch.all(fast_result >= 0.0)
            assert torch.all(fast_result <= 1.0)

            #Less than 1% relative error
            assert torch.allclose(fast_result, torch_result, rtol=1e-2, atol=1e-4) 

if __name__ == "__main__":
    unittest.main()