import torch
from torch import Tensor

__all__ = ["mymuladd", "myadd_out", "fast_sigmoid"]

def mymuladd(a: Tensor, b: Tensor, c: float) -> Tensor:
    """Performs a * b + c in an efficient fused kernel"""
    return torch.ops.extension_cpp.mymuladd.default(a, b, c)


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("extension_cpp::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = torch.ops.extension_cpp.mymul.default(grad, b)
    if ctx.needs_input_grad[1]:
        grad_b = torch.ops.extension_cpp.mymul.default(grad, a)
    return grad_a, grad_b, None


def _setup_context(ctx, inputs, output):
    a, b, c = inputs
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)


# This adds training support for the operator. You must provide us
# the backward formula for the operator and a `setup_context` function
# to save values to be used in the backward.
torch.library.register_autograd(
    "extension_cpp::mymuladd", _backward, setup_context=_setup_context)


@torch.library.register_fake("extension_cpp::mymul")
def _(a, b):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)


def myadd_out(a: Tensor, b: Tensor, out: Tensor) -> None:
    """Writes a + b into out"""
    torch.ops.extension_cpp.myadd_out.default(a, b, out)


def fast_sigmoid(
    input: Tensor, 
    min_val: float = -10.0, 
    max_val: float = 10.0, 
    num_entries: int = 1000
) -> Tensor:
    """
    Fast sigmoid approximation using lookup table with linear interpolation.
    """
    return torch.ops.extension_cpp.fast_sigmoid.default(input, min_val, max_val, num_entries)


@torch.library.register_fake("extension_cpp::fast_sigmoid")
def _(input, min_val, max_val, num_entries):
    torch._check(input.dtype == torch.float32)
    torch._check(input.device.type == "cpu")
    torch._check(min_val < max_val)
    torch._check(num_entries > 1)
    return torch.empty_like(input)


def _fast_sigmoid_backward(ctx, grad_output):
    input, = ctx.saved_tensors
    grad_input = torch.ops.extension_cpp.fast_sigmoid_backward.default(
        grad_output, input, ctx.min_val, ctx.max_val, ctx.num_entries
    )
    return grad_input, None, None, None


def _fast_sigmoid_setup_context(ctx, inputs, output):
    input, min_val, max_val, num_entries = inputs
    ctx.min_val = min_val
    ctx.max_val = max_val
    ctx.num_entries = num_entries
    ctx.save_for_backward(input)

# Register autograd support for fast_sigmoid
torch.library.register_autograd(
    "extension_cpp::fast_sigmoid", 
    _fast_sigmoid_backward, 
    setup_context=_fast_sigmoid_setup_context
)


@torch.library.register_fake("extension_cpp::fast_sigmoid_backward")
def _(grad_output, input, min_val, max_val, num_entries):
    torch._check(input.dtype == torch.float32)
    torch._check(grad_output.dtype == torch.float32)
    torch._check(input.device.type == "cpu")
    torch._check(grad_output.device.type == "cpu")
    torch._check(input.sizes() == grad_output.sizes())
    torch._check(min_val < max_val)
    torch._check(num_entries > 1)
    return torch.empty_like(input)