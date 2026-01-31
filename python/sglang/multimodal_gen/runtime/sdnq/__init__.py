# SDNQ: SD.Next Quantization Engine
# Integrated into sglang for multimodal quantization support

__version__ = "0.1.5"

# Lazy imports to avoid import errors when dependencies are missing
_sdnq_loaded = False
_load_error = None


def _ensure_loaded():
    """Ensure SDNQ modules are loaded."""
    global _sdnq_loaded, _load_error

    if _sdnq_loaded:
        return True

    if _load_error is not None:
        raise _load_error

    try:
        global QuantizationMethod, SDNQConfig, SDNQQuantizer
        global sdnq_post_load_quant, apply_sdnq_to_module, sdnq_quantize_layer
        global save_sdnq_model, load_sdnq_model, sdnq_version

        from .quantizer import (
            QuantizationMethod,
            SDNQConfig,
            SDNQQuantizer,
            sdnq_post_load_quant,
            apply_sdnq_to_module,
            sdnq_quantize_layer,
        )
        from .loader import save_sdnq_model, load_sdnq_model
        from .common import sdnq_version

        _sdnq_loaded = True
        return True

    except ImportError as e:
        _load_error = e
        raise


def __getattr__(name):
    """Lazy attribute access for SDNQ module components."""
    if name in (
        "QuantizationMethod",
        "SDNQConfig",
        "SDNQQuantizer",
        "sdnq_post_load_quant",
        "apply_sdnq_to_module",
        "sdnq_quantize_layer",
        "save_sdnq_model",
        "load_sdnq_model",
        "sdnq_version",
    ):
        _ensure_loaded()
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "QuantizationMethod",
    "SDNQConfig",
    "SDNQQuantizer",
    "apply_sdnq_to_module",
    "load_sdnq_model",
    "save_sdnq_model",
    "sdnq_post_load_quant",
    "sdnq_quantize_layer",
]
