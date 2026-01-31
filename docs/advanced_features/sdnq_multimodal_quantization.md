# SDNQ Quantization for Multimodal Models

SGLang's multimodal generation module supports SDNQ (SD.Next Quantization) for running multimodal models like Qwen-Image-Layered on low-end GPUs. SDNQ provides flexible weight quantization with options for INT8, INT4, and other low-bit formats.

## Overview

SDNQ is designed specifically for diffusion and multimodal models, offering:

- **Multiple precision formats**: INT8, INT4, UINT4, FP8, and other low-bit formats
- **Low-end GPU support**: Significantly reduces VRAM usage for inference on consumer GPUs
- **Quantized matrix multiplication**: Optional INT8/FP8 matmul for faster inference
- **SVDQuant support**: Optional SVD-based low-rank approximation for improved accuracy
- **Module-level offloading**: RAM offloading for extremely limited VRAM scenarios

## Quick Start

### Basic Usage

Enable SDNQ quantization with the `--sdnq-enabled` flag:

```bash
python -m sglang.launch_server \
    --model-path /path/to/qwen-image-layered \
    --sdnq-enabled \
    --port 30000
```

### With Quantized MatMul (Faster Inference)

```bash
python -m sglang.launch_server \
    --model-path /path/to/qwen-image-layered \
    --sdnq-enabled \
    --sdnq-use-quantized-matmul \
    --port 30000
```

### Low-Bit Quantization (INT4)

```bash
python -m sglang.launch_server \
    --model-path /path/to/qwen-image-layered \
    --sdnq-enabled \
    --sdnq-weights-dtype int4 \
    --port 30000
```

## Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--sdnq-enabled` | `false` | Enable SDNQ quantization |
| `--sdnq-weights-dtype` | `int8` | Target dtype for weights (int8, int4, uint4, fp8, etc.) |
| `--sdnq-use-quantized-matmul` | `false` | Use INT8/FP8 quantized matrix multiplication |
| `--sdnq-quant-conv` | `false` | Also quantize convolutional layers |
| `--sdnq-group-size` | `0` | Group size for quantization (0 = auto) |
| `--sdnq-use-svd` | `false` | Enable SVDQuant algorithm for better accuracy |
| `--sdnq-svd-rank` | `32` | Rank size for SVDQuant |
| `--sdnq-dequantize-fp32` | `false` | Use FP32 for dequantization (higher precision) |

## Module-Level Offloading

For extremely limited VRAM scenarios, you can enable module-level offloading to RAM:

```bash
python -m sglang.launch_server \
    --model-path /path/to/qwen-image-layered \
    --sdnq-enabled \
    --multimodal-module-offload \
    --port 30000
```

### Sequential Offloading

For even more VRAM savings, use sequential offloading which loads only one module at a time:

```bash
python -m sglang.launch_server \
    --model-path /path/to/qwen-image-layered \
    --sdnq-enabled \
    --multimodal-module-offload \
    --multimodal-sequential-offload \
    --port 30000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--multimodal-module-offload` | `false` | Enable module-level offloading to RAM |
| `--multimodal-sequential-offload` | `false` | Enable sequential offloading (one module at a time) |

## Supported Models

SDNQ quantization is currently supported for:

- **Qwen-Image-Layered**: Full support with all SDNQ options
- **QwenImage**: Text encoder, transformer, and VAE quantization
- Other multimodal pipelines with transformer/VAE components

## Best Practices

### For Low-End GPUs (4-8GB VRAM)

```bash
python -m sglang.launch_server \
    --model-path /path/to/qwen-image-layered \
    --sdnq-enabled \
    --sdnq-weights-dtype int4 \
    --sdnq-use-quantized-matmul \
    --multimodal-module-offload \
    --multimodal-sequential-offload \
    --port 30000
```

### For Mid-Range GPUs (8-12GB VRAM)

```bash
python -m sglang.launch_server \
    --model-path /path/to/qwen-image-layered \
    --sdnq-enabled \
    --sdnq-weights-dtype int8 \
    --sdnq-use-quantized-matmul \
    --port 30000
```

### For Higher Quality (with SVDQuant)

```bash
python -m sglang.launch_server \
    --model-path /path/to/qwen-image-layered \
    --sdnq-enabled \
    --sdnq-weights-dtype int4 \
    --sdnq-use-svd \
    --sdnq-svd-rank 64 \
    --port 30000
```

## Memory Savings

Typical memory savings with SDNQ quantization:

| Quantization | Approximate Memory Reduction |
|--------------|------------------------------|
| INT8 | ~50% |
| INT4 | ~75% |
| INT4 + Module Offload | ~85-90% (varies by usage) |

## Supported Weight Dtypes

SDNQ supports the following weight data types:

### Integer Types
- `int8`, `int7`, `int6`, `int5`, `int4`, `int3`, `int2`
- `uint8`, `uint7`, `uint6`, `uint5`, `uint4`, `uint3`, `uint2`, `uint1`

### Float Types
- `float8_e4m3fn`, `float8_e5m2`
- Custom low-bit floats: `float7_*`, `float6_*`, `float5_*`, `float4_*`, `float3_*`, `float2_*`

## Programmatic Usage

You can also use SDNQ programmatically:

```python
from sglang.multimodal_gen.runtime.sdnq import sdnq_post_load_quant, SDNQConfig

# Apply SDNQ quantization to a model
quantized_model = sdnq_post_load_quant(
    model,
    weights_dtype="int8",
    use_quantized_matmul=True,
    quant_conv=False,
)
```

## References

- [SDNQ Wiki](https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization)
- [Pre-quantized models on HuggingFace](https://huggingface.co/collections/Disty0/sdnq)
