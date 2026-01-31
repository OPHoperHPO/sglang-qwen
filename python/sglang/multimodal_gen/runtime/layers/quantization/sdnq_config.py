# SPDX-License-Identifier: Apache-2.0
"""
SDNQ (Stable Diffusion Neural Quantization) quantization config for multimodal generation.

This module provides support for loading SDNQ-quantized models (e.g., from
Disty0/Qwen-Image-Layered-SDNQ-uint4-svd-r32) with optional INT8 MatMul acceleration.

Example usage with diffusers pipeline:
    >>> import torch
    >>> import diffusers
    >>> from PIL import Image
    >>> from sdnq import SDNQConfig  # import sdnq to register it
    >>> from sdnq.common import use_torch_compile as triton_is_available
    >>> from sdnq.loader import apply_sdnq_options_to_model
    >>>
    >>> pipe = diffusers.QwenImageLayeredPipeline.from_pretrained(
    ...     "Disty0/Qwen-Image-Layered-SDNQ-uint4-svd-r32",
    ...     torch_dtype=torch.bfloat16
    ... )
    >>>
    >>> # Enable INT8 MatMul for AMD, Intel ARC and Nvidia GPUs:
    >>> if triton_is_available and (torch.cuda.is_available() or torch.xpu.is_available()):
    ...     pipe.transformer = apply_sdnq_options_to_model(
    ...         pipe.transformer, use_quantized_matmul=True
    ...     )
    ...     pipe.text_encoder = apply_sdnq_options_to_model(
    ...         pipe.text_encoder, use_quantized_matmul=True
    ...     )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.quantization import (
    register_quantization_config,
)
from sglang.multimodal_gen.runtime.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.layers.quantization import QuantizationMethods

logger = init_logger(__name__)


def _check_sdnq_available() -> bool:
    """Check if the sdnq library is available."""
    try:
        import sdnq  # noqa: F401

        return True
    except ImportError:
        return False


def _check_triton_available() -> bool:
    """Check if triton-accelerated matmul is available."""
    try:
        from sdnq.common import use_torch_compile as triton_is_available

        return triton_is_available
    except ImportError:
        return False


@register_quantization_config("sdnq")
class SDNQQuantizationConfig(QuantizationConfig):
    """Config class for SDNQ (Stable Diffusion Neural Quantization).

    SDNQ is a quantization method optimized for diffusion models that supports:
    - uint4/int8 weight quantization with SVD low-rank adapters
    - Optional INT8 MatMul acceleration using Triton kernels
    - Compatible with AMD, Intel ARC, and NVIDIA GPUs

    Args:
        use_quantized_matmul: Whether to use INT8 quantized matrix multiplication.
            Requires Triton and CUDA/XPU support. Default is False.
        apply_to_transformer: Whether to apply SDNQ options to the transformer model.
            Default is True.
        apply_to_text_encoder: Whether to apply SDNQ options to the text encoder.
            Default is True.
    """

    def __init__(
        self,
        use_quantized_matmul: bool = False,
        apply_to_transformer: bool = True,
        apply_to_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.use_quantized_matmul = use_quantized_matmul
        self.apply_to_transformer = apply_to_transformer
        self.apply_to_text_encoder = apply_to_text_encoder

        # Validate SDNQ availability
        if not _check_sdnq_available():
            logger.warning(
                "SDNQ library is not installed. Install it with: pip install sdnq"
            )

        # Check triton availability if quantized matmul is requested
        if use_quantized_matmul:
            if not _check_triton_available():
                logger.warning(
                    "Triton-accelerated matmul is not available. "
                    "INT8 matmul will be disabled."
                )
                self.use_quantized_matmul = False
            elif not (torch.cuda.is_available() or (hasattr(torch, "xpu") and torch.xpu.is_available())):
                logger.warning(
                    "Neither CUDA nor XPU is available. "
                    "INT8 matmul will be disabled."
                )
                self.use_quantized_matmul = False

    def get_name(self) -> "QuantizationMethods":
        """Return the name of the quantization method."""
        return "sdnq"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """Return supported activation dtypes."""
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        """Return minimum GPU capability required.

        SDNQ supports a wide range of GPUs including:
        - NVIDIA (SM 70+): Volta, Turing, Ampere, Hopper
        - AMD ROCm GPUs
        - Intel ARC GPUs
        """
        return 70  # Volta and above for NVIDIA

    @staticmethod
    def get_config_filenames() -> List[str]:
        """Return config filenames to search for in model directory."""
        return ["quantization_config.json", "config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SDNQQuantizationConfig":
        """Create a config instance from a model's quantization config dict."""
        use_quantized_matmul = cls.get_from_keys_or(
            config, ["use_quantized_matmul", "quantized_matmul"], False
        )
        apply_to_transformer = cls.get_from_keys_or(
            config, ["apply_to_transformer"], True
        )
        apply_to_text_encoder = cls.get_from_keys_or(
            config, ["apply_to_text_encoder"], True
        )
        return cls(
            use_quantized_matmul=use_quantized_matmul,
            apply_to_transformer=apply_to_transformer,
            apply_to_text_encoder=apply_to_text_encoder,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """Get the quantization method for a layer.

        For SDNQ, quantization is applied at the model level via
        apply_sdnq_options_to_model(), not at individual layer level.
        This method returns None as SDNQ handles quantization differently.
        """
        # SDNQ uses a different approach - it applies quantization
        # at the model level, not layer level
        return None

    def apply_sdnq_to_model(self, model: nn.Module) -> nn.Module:
        """Apply SDNQ options to a model.

        This method wraps the sdnq library's apply_sdnq_options_to_model function
        to enable INT8 MatMul acceleration if configured.

        Args:
            model: The model to apply SDNQ options to.

        Returns:
            The model with SDNQ options applied.
        """
        if not _check_sdnq_available():
            logger.warning(
                "SDNQ library not available. Returning model unchanged."
            )
            return model

        try:
            from sdnq.loader import apply_sdnq_options_to_model

            if self.use_quantized_matmul:
                logger.info(
                    "Applying SDNQ with INT8 quantized matmul to model: %s",
                    model.__class__.__name__,
                )
                model = apply_sdnq_options_to_model(
                    model, use_quantized_matmul=True
                )
            else:
                logger.info(
                    "SDNQ model loaded without INT8 matmul: %s",
                    model.__class__.__name__,
                )
        except Exception as e:
            logger.error("Failed to apply SDNQ options to model: %s", str(e))

        return model


class SDNQLinearMethod(QuantizeMethodBase):
    """Linear method for SDNQ quantization.

    Note: SDNQ applies quantization at the model level during loading,
    not at individual layer creation time. This class is provided for
    compatibility with the quantization framework but delegates actual
    quantization to the sdnq library's model-level functions.
    """

    def __init__(self, quant_config: SDNQQuantizationConfig):
        self.quant_config = quant_config

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ) -> None:
        """Create weights for a layer.

        For SDNQ, weights are created by the sdnq library during model loading.
        This method is a no-op placeholder.
        """
        # SDNQ handles weight creation during model loading
        pass

    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Apply the quantized computation.

        For SDNQ, the forward pass is handled by the sdnq library's
        quantized modules. This method delegates to the layer's forward.
        """
        # SDNQ replaces modules with quantized versions during loading
        # The forward pass is handled by those quantized modules
        return layer.forward(*args, **kwargs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process weights after loading.

        For SDNQ, post-loading processing is handled by the sdnq library.
        """
        pass
