# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

from typing import Tuple

import torch

from ...common import use_contiguous_mm # noqa: TID252


def check_mats(input: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    input = input.contiguous()
    if use_contiguous_mm:
        weight = weight.contiguous()
    elif weight.is_contiguous():
        weight = weight.t().contiguous().t()
    return input, weight


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    # Dequantize weights and ensure they match the input dtype
    dequantized_weight = self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down)
    # Cast to input dtype to avoid dtype mismatch (e.g., BFloat16 input vs Float32 weight)
    if dequantized_weight.dtype != input.dtype:
        dequantized_weight = dequantized_weight.to(dtype=input.dtype)
    # Also ensure bias dtype matches if present
    bias = self.bias
    if bias is not None and bias.dtype != input.dtype:
        bias = bias.to(dtype=input.dtype)
    return torch.nn.functional.linear(input, dequantized_weight, bias)
