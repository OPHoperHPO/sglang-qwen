# SPDX-License-Identifier: Apache-2.0
"""
SDNQ weight dequantization utilities for SGLang model loading.

This module provides utilities for loading SDNQ-quantized model weights
and dequantizing them for use with SGLang's customized model architectures.

SDNQ is a quantization method that stores weights in uint4/int8 format
with SVD low-rank adapters for quality preservation.
"""

import json
import os
from collections import defaultdict
from collections.abc import Generator
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def is_sdnq_quantized_model(model_path: str) -> bool:
    """Check if a model is SDNQ-quantized by reading its config.json.

    Args:
        model_path: Path to the model directory

    Returns:
        True if the model is SDNQ-quantized, False otherwise
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        quant_config = config.get("quantization_config", {})
        quant_method = quant_config.get("quant_method", "").lower()

        return quant_method == "sdnq"
    except Exception as e:
        logger.debug("Could not check for SDNQ config in %s: %s", model_path, e)
        return False


def get_sdnq_config(model_path: str) -> dict | None:
    """Get SDNQ quantization config from model's config.json.

    Args:
        model_path: Path to the model directory

    Returns:
        SDNQ quantization config dict or None if not found
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        quant_config = config.get("quantization_config", {})
        if quant_config.get("quant_method", "").lower() == "sdnq":
            return quant_config
    except Exception:
        pass

    return None


def _dequantize_uint4(
    packed_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    target_shape: tuple[int, ...],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize uint4 packed weights.

    SDNQ stores uint4 weights packed as uint8 (2 values per byte).
    This function unpacks and dequantizes them.

    Args:
        packed_weight: Packed uint4 weights
        scale: Quantization scale
        zero_point: Quantization zero point
        target_shape: Target shape for the dequantized weight
        dtype: Output dtype

    Returns:
        Dequantized weight tensor
    """
    # Unpack uint4 from uint8 (2 values per byte)
    # Low nibble first, then high nibble
    if packed_weight.dtype == torch.uint8:
        low = packed_weight & 0x0F
        high = (packed_weight >> 4) & 0x0F
        unpacked = torch.stack([low, high], dim=-1).flatten()
    else:
        unpacked = packed_weight.flatten()

    # Calculate expected size
    target_numel = 1
    for dim in target_shape:
        target_numel *= dim

    # Trim or pad to match target size
    if unpacked.numel() > target_numel:
        unpacked = unpacked[:target_numel]
    elif unpacked.numel() < target_numel:
        # Pad with zeros if needed
        padding = torch.zeros(
            target_numel - unpacked.numel(), dtype=unpacked.dtype, device=unpacked.device
        )
        unpacked = torch.cat([unpacked, padding])

    # Reshape to target shape
    unpacked = unpacked.reshape(target_shape)

    # Dequantize: weight = (quantized - zero_point) * scale
    # Handle broadcasting for scale and zero_point
    scale = scale.to(dtype)
    zero_point = zero_point.to(dtype)

    dequantized = (unpacked.to(dtype) - zero_point) * scale

    return dequantized


def _apply_svd_adapter(
    weight: torch.Tensor, svd_down: torch.Tensor, svd_up: torch.Tensor
) -> torch.Tensor:
    """Apply SVD low-rank adapter to weight.

    SDNQ uses SVD adapters to improve quality: weight = base + down @ up

    Args:
        weight: Base weight tensor
        svd_down: SVD down projection matrix
        svd_up: SVD up projection matrix

    Returns:
        Weight with SVD adapter applied
    """
    # SVD adapter: out_features x rank, rank x in_features
    # Result: out_features x in_features
    adapter = torch.mm(svd_down, svd_up)
    return weight + adapter.to(weight.dtype)


class SDNQWeightDequantizer:
    """Dequantizes SDNQ weights during model loading.

    This class collects SDNQ-related parameters (scale, zero_point, svd_down,
    svd_up, packed_weight) and dequantizes them to produce regular float weights.
    """

    # SDNQ parameter suffixes
    SDNQ_SUFFIXES = [".scale", ".zero_point", ".svd_down", ".svd_up"]

    def __init__(self, quant_config: dict | None = None):
        """Initialize the dequantizer.

        Args:
            quant_config: SDNQ quantization config from model
        """
        self.quant_config = quant_config or {}
        self.bits = self.quant_config.get("bits", 4)

        # Buffer to collect SDNQ parameters for each layer
        # Key: base_name (e.g., "transformer_blocks.0.attn.to_q")
        # Value: dict of param_type -> tensor
        self._param_buffer: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

        # Set of base names that have been dequantized
        self._dequantized: set[str] = set()

    def is_sdnq_param(self, param_name: str) -> bool:
        """Check if a parameter is an SDNQ auxiliary parameter.

        Args:
            param_name: Full parameter name

        Returns:
            True if this is an SDNQ auxiliary parameter (scale, zero_point, etc.)
        """
        return any(param_name.endswith(suffix) for suffix in self.SDNQ_SUFFIXES)

    def get_base_name(self, param_name: str) -> str:
        """Get the base parameter name without SDNQ suffix.

        Args:
            param_name: Full parameter name

        Returns:
            Base parameter name
        """
        for suffix in self.SDNQ_SUFFIXES:
            if param_name.endswith(suffix):
                return param_name[: -len(suffix)]
        return param_name

    def get_param_type(self, param_name: str) -> str:
        """Get the SDNQ parameter type from name.

        Args:
            param_name: Full parameter name

        Returns:
            Parameter type (e.g., "scale", "zero_point", "weight")
        """
        for suffix in self.SDNQ_SUFFIXES:
            if param_name.endswith(suffix):
                return suffix[1:]  # Remove leading dot
        return "weight"

    def add_param(self, param_name: str, tensor: torch.Tensor) -> None:
        """Add a parameter to the buffer for later dequantization.

        Args:
            param_name: Full parameter name
            tensor: Parameter tensor
        """
        base_name = self.get_base_name(param_name)
        param_type = self.get_param_type(param_name)
        self._param_buffer[base_name][param_type] = tensor

    def can_dequantize(self, base_name: str) -> bool:
        """Check if we have all required params to dequantize a layer.

        Args:
            base_name: Base parameter name

        Returns:
            True if all required parameters are available
        """
        if base_name in self._dequantized:
            return False

        params = self._param_buffer.get(base_name, {})
        # Minimum: need weight, scale, zero_point
        required = ["weight", "scale", "zero_point"]
        return all(p in params for p in required)

    def dequantize(
        self, base_name: str, target_shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Dequantize a layer's weight.

        Args:
            base_name: Base parameter name
            target_shape: Expected shape of the dequantized weight
            dtype: Output dtype

        Returns:
            Dequantized weight tensor or None if not ready
        """
        if not self.can_dequantize(base_name):
            return None

        params = self._param_buffer[base_name]
        packed_weight = params["weight"]
        scale = params["scale"]
        zero_point = params["zero_point"]

        # Dequantize based on bit width
        if self.bits == 4:
            weight = _dequantize_uint4(
                packed_weight, scale, zero_point, target_shape, dtype
            )
        else:
            # For 8-bit, simpler dequantization
            weight = (packed_weight.to(dtype) - zero_point.to(dtype)) * scale.to(dtype)
            if weight.shape != target_shape:
                weight = weight.reshape(target_shape)

        # Apply SVD adapter if available
        if "svd_down" in params and "svd_up" in params:
            weight = _apply_svd_adapter(weight, params["svd_down"], params["svd_up"])

        self._dequantized.add(base_name)
        return weight

    def process_weight_iterator(
        self,
        weight_iterator: Generator[tuple[str, torch.Tensor], None, None],
        model_state_dict: dict[str, torch.Tensor],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Process a weight iterator, dequantizing SDNQ weights as needed.

        This wraps the original weight iterator and:
        1. Buffers SDNQ auxiliary params (scale, zero_point, svd_*)
        2. When a quantized weight is encountered, dequantizes it
        3. Skips auxiliary params (they're consumed by dequantization)

        Args:
            weight_iterator: Original weight iterator
            model_state_dict: Model's state dict for getting target shapes

        Yields:
            (param_name, tensor) pairs with dequantized weights
        """
        # First pass: collect all params and identify SDNQ layers
        all_params = list(weight_iterator)

        # Identify which base names have SDNQ auxiliary params
        sdnq_bases = set()
        for param_name, _ in all_params:
            if self.is_sdnq_param(param_name):
                base_name = self.get_base_name(param_name)
                sdnq_bases.add(base_name)

        # First pass: buffer all SDNQ-related params
        for param_name, tensor in all_params:
            base_name = self.get_base_name(param_name)
            if self.is_sdnq_param(param_name) or base_name in sdnq_bases:
                # Buffer both auxiliary params and main weights for SDNQ layers
                self.add_param(param_name, tensor)

        # Second pass: yield dequantized weights
        yielded_bases = set()
        for param_name, tensor in all_params:
            if self.is_sdnq_param(param_name):
                # Skip auxiliary params
                continue

            base_name = self.get_base_name(param_name)

            # Try to get target shape from model state dict
            # Map various possible names
            target_tensor = model_state_dict.get(param_name)
            if target_tensor is None:
                # Try with .weight suffix
                target_tensor = model_state_dict.get(param_name + ".weight")

            if self.can_dequantize(base_name) and base_name not in yielded_bases:
                if target_tensor is not None:
                    target_shape = target_tensor.shape
                    dequantized = self.dequantize(base_name, target_shape, tensor.dtype)
                    if dequantized is not None:
                        yield param_name, dequantized
                        yielded_bases.add(base_name)
                        continue

            # Not an SDNQ weight or couldn't dequantize, yield as-is
            yield param_name, tensor
