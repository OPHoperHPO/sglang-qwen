import os
import json
import logging
import torch
from diffusers.models.modeling_utils import ModelMixin

from .common import dtype_dict, use_tensorwise_fp8_matmul, check_torch_compile, conv_types, linear_types
from .quantizer import SDNQConfig, sdnq_post_load_quant, prepare_weight_for_matmul, prepare_svd_for_matmul, get_quant_args_from_config
from .forward import get_forward_func
from .file_loader import load_files

logger = logging.getLogger("sdnq")


def get_module_names(model: ModelMixin) -> list:
    modules_names = model._internal_dict.keys() # pylint: disable=protected-access
    modules_names = [m for m in modules_names if not m.startswith("_")]
    modules_names = [m for m in modules_names if isinstance(getattr(model, m, None), torch.nn.Module)]
    modules_names = sorted(set(modules_names))
    return modules_names


def unset_config_on_save(quantization_config: SDNQConfig) -> SDNQConfig:
    quantization_config.quantization_device = None
    quantization_config.return_device = None
    quantization_config.non_blocking = False
    quantization_config.add_skip_keys = False
    return quantization_config


def save_sdnq_model(model: ModelMixin, model_path: str, max_shard_size: str = "5GB", is_pipeline: bool = False, sdnq_config: SDNQConfig = None) -> None:
    if is_pipeline:
        for module_name in get_module_names(model):
            module = getattr(model, module_name, None)
            if hasattr(module, "config") and hasattr(module.config, "quantization_config") and isinstance(module.config.quantization_config, SDNQConfig):
                module.config.quantization_config = unset_config_on_save(module.config.quantization_config)
            if hasattr(module, "quantization_config") and isinstance(module.quantization_config, SDNQConfig):
                module.quantization_config = unset_config_on_save(module.quantization_config)
    else:
        if hasattr(model, "config") and hasattr(model.config, "quantization_config") and isinstance(model.config.quantization_config, SDNQConfig):
            model.config.quantization_config = unset_config_on_save(model.config.quantization_config)
        if hasattr(model, "quantization_config") and isinstance(model.quantization_config, SDNQConfig):
            model.quantization_config = unset_config_on_save(model.quantization_config)

    model.save_pretrained(model_path, max_shard_size=max_shard_size) # actual save

    quantization_config_path = os.path.join(model_path, "quantization_config.json")
    if sdnq_config is not None: # if provided, save global config
        sdnq_config = unset_config_on_save(sdnq_config)
        sdnq_config.to_json_file(quantization_config_path)

    if is_pipeline:
        for module_name in get_module_names(model): # save per-module config if available
            module = getattr(model, module_name, None)
            if module is None:
                continue
            module_quantization_config_path = os.path.join(model_path, module_name, "quantization_config.json")
            if hasattr(module, "quantization_config") and isinstance(module.quantization_config, SDNQConfig):
                module.quantization_config.to_json_file(module_quantization_config_path)
            elif hasattr(module, "config") and hasattr(module.config, "quantization_config") and isinstance(module.config.quantization_config, SDNQConfig):
                module.config.quantization_config.to_json_file(module_quantization_config_path)
    elif sdnq_config is None:
        if hasattr(model, "quantization_config") and isinstance(model.quantization_config, SDNQConfig):
            model.quantization_config.to_json_file(quantization_config_path)
        elif hasattr(model, "config") and hasattr(model.config, "quantization_config") and isinstance(model.config.quantization_config, SDNQConfig):
            model.config.quantization_config.to_json_file(quantization_config_path)


def load_sdnq_model(model_path: str, model_cls: ModelMixin = None, file_name: str = None, dtype: torch.dtype = None, device: torch.device = "cpu", dequantize_fp32: bool = None, use_quantized_matmul: bool = None, model_config: dict = None, quantization_config: dict = None, load_method: str = "safetensors") -> ModelMixin:
    from accelerate import init_empty_weights

    # Keys that are metadata and should not be passed to model constructor
    METADATA_KEYS = {
        "_class_name", "_diffusers_version", "_name_or_path", "architectures",
        "quantization_config", "_transformers_version", "model_type"
    }

    def filter_model_config(config: dict) -> dict:
        """Filter out metadata keys from config before passing to model constructor."""
        return {k: v for k, v in config.items() if k not in METADATA_KEYS}

    with init_empty_weights():
        model_config_path = os.path.join(model_path, "config.json")
        quantization_config_path = os.path.join(model_path, "quantization_config.json")

        if model_config is None:
            if os.path.exists(model_config_path):
                with open(model_config_path, "r", encoding="utf-8") as f:
                    model_config = json.load(f)
            else:
                model_config = {}

        if quantization_config is None:
            if os.path.exists(quantization_config_path):
                with open(quantization_config_path, "r", encoding="utf-8") as f:
                    quantization_config = json.load(f)
            else:
                quantization_config = model_config.get("quantization_config", None)
                if quantization_config is None:
                    raise ValueError(f"Cannot determine quantization_config for {model_path}, please provide quantization_config argument")

        if model_cls is None:
            import transformers
            import diffusers
            class_name = model_config.get("_class_name", None) or model_config.get("architectures", None)
            if isinstance(class_name, list):
                class_name = class_name[0]
            if class_name is not None:
                # First try to resolve from sglang's ModelRegistry (for custom models like QwenImageTransformer2DModel)
                try:
                    from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
                    model_cls, _ = ModelRegistry.resolve_model_cls(class_name)
                    logger.info(f"SDNQ: Resolved model class {class_name} from sglang ModelRegistry")
                except Exception:
                    # Fall back to diffusers/transformers
                    model_cls = getattr(diffusers, class_name, None) or getattr(transformers, class_name, None)
                    if model_cls is not None:
                        logger.info(f"SDNQ: Resolved model class {class_name} from diffusers/transformers")
        if model_cls is None:
            raise ValueError(f"Cannot determine model class for {model_path}, please provide model_cls argument")

        # Check if we need to use sglang's initialization (for models with custom config handling)
        # sglang models use (config, hf_config) initialization pattern
        sglang_model_init = False
        try:
            import inspect
            init_sig = inspect.signature(model_cls.__init__)
            params = list(init_sig.parameters.keys())
            if "hf_config" in params:
                sglang_model_init = True
        except Exception:
            pass

        if sglang_model_init:
            # Use sglang's initialization pattern with config and hf_config
            from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import get_diffusers_component_config
            from copy import deepcopy
            try:
                config_dict = get_diffusers_component_config(model_path=model_path)
                hf_config = deepcopy(config_dict)

                # Get the appropriate config class from the model class
                config_cls = getattr(model_cls, "config_class", None)
                if config_cls is None:
                    # Try to find config class from model's module
                    model_module = model_cls.__module__
                    try:
                        mod = __import__(model_module, fromlist=[model_cls.__name__])
                        # Look for a config class with similar name
                        for attr_name in dir(mod):
                            if "Config" in attr_name and not attr_name.startswith("_"):
                                potential_config = getattr(mod, attr_name)
                                if hasattr(potential_config, "arch_config"):
                                    config_cls = potential_config
                                    break
                    except Exception:
                        pass

                if config_cls is not None:
                    dit_config = config_cls()
                    dit_config.update_model_arch(config_dict)
                    model = model_cls(config=dit_config, hf_config=hf_config)
                    logger.info(f"SDNQ: Initialized model using sglang config pattern")
                else:
                    # Fallback to standard initialization
                    if hasattr(model_cls, "load_config") and hasattr(model_cls, "from_config"):
                        config = model_cls.load_config(model_path)
                        model = model_cls.from_config(config)
                    else:
                        model = model_cls(**filter_model_config(model_config))
            except Exception as e:
                logger.warning(f"SDNQ: Failed to use sglang config pattern: {e}, falling back to standard init")
                if hasattr(model_cls, "load_config") and hasattr(model_cls, "from_config"):
                    config = model_cls.load_config(model_path)
                    model = model_cls.from_config(config)
                else:
                    model = model_cls(**filter_model_config(model_config))
        elif hasattr(model_cls, "load_config") and hasattr(model_cls, "from_config"):
            config = model_cls.load_config(model_path)
            model = model_cls.from_config(config)
        elif hasattr(model_cls, "_from_config"):
            config = transformers.AutoConfig.from_pretrained(model_path)
            model = model_cls(config)
        else:
            model = model_cls(**filter_model_config(model_config))

        model = sdnq_post_load_quant(model, torch_dtype=dtype, add_skip_keys=False, use_dynamic_quantization=False, **get_quant_args_from_config(quantization_config))

    key_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
    files = []

    if file_name:
        files.append(os.path.join(model_path, file_name))
    else:
        all_files = os.listdir(model_path)
        files = sorted([os.path.join(model_path, f) for f in all_files if f.endswith(".safetensors")])

    state_dict = load_files(files, key_mapping=key_mapping, device=device, method=load_method)

    if isinstance(getattr(model, "_tied_weights_keys", None), dict):
        for key, value in model._tied_weights_keys.items(): # pylint: disable=protected-access
            if value in state_dict.keys() and key not in state_dict.keys():
                state_dict[key] = state_dict[value]
    else:
        # older transformers case, handle known models manually
        if model.__class__.__name__ in {"T5EncoderModel", "UMT5EncoderModel"} and "encoder.embed_tokens.weight" not in state_dict.keys():
            state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]
        elif model.__class__.__name__ in {"Qwen3ForCausalLM"} and "lm_head.weight" not in state_dict.keys():
            if "model.embed_tokens.weight" in state_dict.keys():
                state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    # Load state dict with strict=False to handle extra/missing keys gracefully
    load_result = model.load_state_dict(state_dict, assign=True, strict=False)
    if load_result.missing_keys:
        logger.warning(f"SDNQ: Missing keys when loading state dict: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        logger.warning(f"SDNQ: Unexpected keys in state dict (ignored): {load_result.unexpected_keys}")
    del state_dict

    model = post_process_model(model)
    if (dtype is not None) or (dequantize_fp32 is not None) or (use_quantized_matmul is not None):
        model = apply_sdnq_options_to_model(model, dtype=dtype, dequantize_fp32=dequantize_fp32, use_quantized_matmul=use_quantized_matmul)
    return model


def post_process_model(model):
    has_children = list(model.children())
    if not has_children:
        return model
    for module_name, module in model.named_children():
        if hasattr(module, "sdnq_dequantizer"):
            module.weight.requires_grad_(False)
            module.scale.requires_grad_(False)
            if module.zero_point is not None:
                module.zero_point.requires_grad_(False)
            if module.sdnq_dequantizer.use_quantized_matmul and not module.sdnq_dequantizer.re_quantize_for_matmul:
                module.weight.data = prepare_weight_for_matmul(module.weight)
            if module.svd_up is not None:
                module.svd_up.requires_grad_(False)
                module.svd_down.requires_grad_(False)
                module.svd_up.data, module.svd_down.data = prepare_svd_for_matmul(module.svd_up, module.svd_down, module.sdnq_dequantizer.use_quantized_matmul)
            setattr(model, module_name, module)
        else:
            setattr(model, module_name, post_process_model(module))
    return model


def apply_sdnq_options_to_module(model, dtype: torch.dtype = None, dequantize_fp32: bool = None, use_quantized_matmul: bool = None):
    has_children = list(model.children())
    if not has_children:
        if dtype is not None and getattr(model, "dtype", torch.float32) != torch.float32:
            model = model.to(dtype=dtype)
        return model
    for module_name, module in model.named_children():
        if hasattr(module, "sdnq_dequantizer"):
            layer_class_name = module.original_class.__name__
            current_use_quantized_matmul = use_quantized_matmul
            if current_use_quantized_matmul:
                if layer_class_name in conv_types:
                    output_channel_size, channel_size = module.sdnq_dequantizer.original_shape[:2]
                elif layer_class_name in linear_types:
                    output_channel_size, channel_size = module.sdnq_dequantizer.original_shape
                else:
                    current_use_quantized_matmul = False
                current_use_quantized_matmul = current_use_quantized_matmul and channel_size >= 32 and output_channel_size >= 32 # pylint: disable=possibly-used-before-assignment
                current_use_quantized_matmul = current_use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0 # pylint: disable=possibly-used-before-assignment

            if dtype is not None and module.sdnq_dequantizer.result_dtype != torch.float32:
                module.sdnq_dequantizer.result_dtype = dtype

            upcast_scale = bool(
                dequantize_fp32
                or dtype_dict[module.sdnq_dequantizer.weights_dtype]["num_bits"] > 8
                or (
                    (current_use_quantized_matmul or (current_use_quantized_matmul is None and module.sdnq_dequantizer.use_quantized_matmul))
                    and not dtype_dict[module.sdnq_dequantizer.quantized_matmul_dtype]["is_integer"]
                    and (not use_tensorwise_fp8_matmul or dtype_dict[module.sdnq_dequantizer.quantized_matmul_dtype]["num_bits"] == 16)
                )
            )
            scale_dtype = torch.float32 if upcast_scale or dequantize_fp32 or (dequantize_fp32 is None and module.scale.dtype == torch.float32) else module.sdnq_dequantizer.result_dtype

            module.scale.data = module.scale.to(dtype=scale_dtype)
            if module.zero_point is not None:
                module.zero_point.data = module.zero_point.to(dtype=scale_dtype)
            if module.svd_up is not None:
                module.svd_up.data = module.svd_up.to(dtype=scale_dtype)
                module.svd_down.data = module.svd_down.to(dtype=scale_dtype)

            if current_use_quantized_matmul is not None and current_use_quantized_matmul != module.sdnq_dequantizer.use_quantized_matmul:
                if not module.sdnq_dequantizer.re_quantize_for_matmul and not dtype_dict[module.sdnq_dequantizer.weights_dtype]["is_packed"]:
                    module.scale.t_()
                    module.weight.t_()
                    if current_use_quantized_matmul:
                        module.weight.data = prepare_weight_for_matmul(module.weight)
                    else:
                        module.scale.data = module.scale.contiguous()
                        module.weight.data = module.weight.contiguous()
                if module.svd_up is not None:
                    module.svd_up.data, module.svd_down.data = prepare_svd_for_matmul(module.svd_up.t_(), module.svd_down.t_(), current_use_quantized_matmul)
                module.sdnq_dequantizer.use_quantized_matmul = current_use_quantized_matmul
                module.forward_func = get_forward_func(module.original_class.__name__, module.sdnq_dequantizer.quantized_matmul_dtype, current_use_quantized_matmul)
            setattr(model, module_name, module)
        else:
            setattr(model, module_name, apply_sdnq_options_to_module(module, dtype=dtype, dequantize_fp32=dequantize_fp32, use_quantized_matmul=use_quantized_matmul))
    return model


def apply_sdnq_options_to_model(model, dtype: torch.dtype = None, dequantize_fp32: bool = None, use_quantized_matmul: bool = None):
    if use_quantized_matmul and not check_torch_compile():
        raise RuntimeError("SDNQ Quantized MatMul requires a working Triton install.")
    model = apply_sdnq_options_to_module(model, dtype=dtype, dequantize_fp32=dequantize_fp32, use_quantized_matmul=use_quantized_matmul)
    if hasattr(model, "quantization_config"):
        if use_quantized_matmul is not None:
            model.quantization_config.use_quantized_matmul = use_quantized_matmul
        if dequantize_fp32 is not None:
            model.quantization_config.dequantize_fp32 = dequantize_fp32
    if hasattr(model, "config"):
        try:
            if hasattr(model.config, "quantization_config"):
                if use_quantized_matmul is not None:
                    model.config.quantization_config.use_quantized_matmul = use_quantized_matmul
                if dequantize_fp32 is not None:
                    model.config.quantization_config.dequantize_fp32 = dequantize_fp32
        except Exception:
            pass
        try:
            if hasattr(model.config, "get") and model.config.get("quantization_config", None) is not None:
                if use_quantized_matmul is not None:
                    model.config["quantization_config"].use_quantized_matmul = use_quantized_matmul
                if dequantize_fp32 is not None:
                    model.config["quantization_config"].dequantize_fp32 = dequantize_fp32
        except Exception:
            pass
    return model
