# vLLM GPU Memory Calculator API Documentation

## Overview

The vLLM GPU Memory Calculator API is a specialized tool designed to help you estimate the GPU memory requirements for serving large language models (LLMs) with the vLLM framework. This API provides detailed memory breakdown calculations based on model architecture, batch size, sequence length, and other parameters to help you plan and optimize your GPU resources.

## Core Concepts

Before using the API, it helps to understand the components that contribute to GPU memory usage when serving LLMs:

1. **Model Parameters Memory**: The memory required to store the model weights themselves.
2. **Activation Memory**: Memory needed for intermediate tensors during the forward pass.
3. **KV Cache Memory**: Memory used by the key-value cache for efficient inference, which grows with context length.
4. **CUDA Overhead**: Memory consumed by CUDA context and additional libraries.

The API follows the general equation:
```
total_memory = model_memory + activation_memory + kv_cache_memory + cuda_overhead
```

## API Endpoints

### 1. Estimate GPU Memory

**Endpoint**: `POST /estimate-gpu-memory`

This is the primary endpoint that calculates the estimated GPU memory required for a specific model configuration.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_name` | string | Yes | Name of the model (e.g., "meta-llama/Llama-2-7b") |
| `max_seq_len` | integer | Yes | Maximum sequence length for inference |
| `dtype` | string | No | Data type for model weights (default: "float16") |
| `kv_cache_dtype` | string | No | Data type for KV cache (defaults to `dtype` if not specified) |
| `max_batch_size` | integer | No | Maximum batch size (default: 32) |
| `max_model_len` | integer | No | Maximum model length (defaults to `max_seq_len` if not specified) |
| `gpu_memory_utilization` | float | No | GPU memory utilization ratio (default: 0.9) |
| `quantization` | string | No | Quantization method ("Q4", "Q8", or null for full precision) |
| `params_billions` | float | No | Override for parameter count in billions |
| `architecture` | object | No | Optional model architecture details |

The `architecture` object can include:

- `num_layers`: Number of transformer layers
- `hidden_size`: Hidden dimension size
- `num_heads`: Number of attention heads
- `head_dim`: Dimension per attention head
- `intermediate_size`: Size of feed-forward network layers

#### Example Request

```json
{
  "model_name": "meta-llama/Llama-2-7b",
  "max_seq_len": 2048,
  "dtype": "float16",
  "max_batch_size": 32,
  "gpu_memory_utilization": 0.9,
  "quantization": null,
  "params_billions": null,
  "architecture": {
    "num_layers": 32,
    "hidden_size": 4096,
    "num_heads": 32,
    "head_dim": 128,
    "intermediate_size": 11008
  }
}
```

#### Response

The API returns a detailed breakdown of memory requirements:

| Field | Description |
|-------|-------------|
| `model_name` | Name of the model |
| `model_params_memory_gb` | Memory required for model parameters in GB |
| `activation_memory_gb` | Memory required for activations in GB |
| `min_kv_cache_memory_gb` | Minimum KV cache memory with PagedAttention in GB |
| `max_kv_cache_memory_gb` | Maximum KV cache memory with PagedAttention in GB |
| `cuda_overhead_gb` | Memory required for CUDA context in GB |
| `total_min_memory_gb` | Minimum total memory requirement in GB |
| `total_max_memory_gb` | Maximum total memory requirement in GB |
| `recommended_memory_gb` | Recommended GPU memory based on utilization ratio |
| `warning` | Optional warning message for very large models |
| `components_breakdown` | Detailed breakdown of memory components |
| `architecture_used` | The architecture values used in the calculation |

#### Example Response

```json
{
  "model_name": "meta-llama/Llama-2-7b",
  "model_params_memory_gb": 14.98,
  "activation_memory_gb": 3.25,
  "min_kv_cache_memory_gb": 0.42,
  "max_kv_cache_memory_gb": 53.69,
  "cuda_overhead_gb": 0.75,
  "total_min_memory_gb": 19.4,
  "total_max_memory_gb": 72.67,
  "recommended_memory_gb": 80.74,
  "warning": null,
  "components_breakdown": {
    "model_params": 14.98,
    "activation": 3.25,
    "min_kv_cache": 0.42,
    "max_kv_cache": 53.69,
    "cuda_overhead": 0.75
  },
  "architecture_used": {
    "num_layers": 32,
    "hidden_size": 4096,
    "num_heads": 32,
    "head_dim": 128,
    "intermediate_size": 11008
  }
}
```

### 2. Get Known Architectures

**Endpoint**: `GET /known-architectures`

Returns a list of known model architectures in the system's database.

#### Response

A JSON object containing model architectures for known models.

### 3. Root Endpoint

**Endpoint**: `GET /`

Returns basic information about the API, including an example request.

## Supported Models

The API includes pre-configured information for many popular model families:

- **LLaMA family**: Llama 2 (7B, 13B, 70B), Llama 3 (8B, 70B), Llama 3.1 (8B, 70B, 405B), Llama 3.2 (1B, 3B)
- **Mistral family**: Mistral 7B, Mixtral 8x7B, Mistral Large/Medium, Mixtral 8x22B
- **OpenAI models**: GPT-3.5-Turbo, GPT-4, GPT-4o
- **Qwen family**: Qwen 1.5 (0.5B to 110B)
- **Gemma family**: Gemma 2B, 7B
- **Claude family**: Claude 3 (Haiku, Sonnet, Opus)
- **Gemini family**: Gemini Nano, Pro, Ultra
- **Yi family**: Yi (6B, 9B, 34B)
- **Falcon family**: Falcon (7B, 40B, 180B)
- And others (StableLM, MPT, Pythia, OpenChat)

For models not in the database, the API attempts to estimate parameters based on the model name or uses defaults.

## Understanding the Results

### Memory Components

1. **Model Parameters Memory**
   - The memory required to store the model weights
   - Affected by parameter count, data type, and quantization

2. **Activation Memory**
   - Memory used for intermediate tensors during inference
   - Scales with batch size, sequence length, and model dimensions

3. **KV Cache Memory**
   - Memory for key and value tensors cached during attention
   - Grows with sequence length and batch size
   - The API calculates minimum and maximum estimates with PagedAttention

4. **CUDA Overhead**
   - Fixed memory consumption for CUDA context
   - Additional overhead for quantization libraries

### PagedAttention

The API incorporates vLLM's PagedAttention mechanism, which allocates memory in blocks for more efficient memory usage:

- **Minimum KV Cache**: Represents the initial allocation (one block per sequence)
- **Maximum KV Cache**: Represents the worst-case scenario where all blocks are used

### Recommended Memory

The API provides a recommended GPU memory value based on:
- Total maximum memory requirement
- User-specified GPU memory utilization ratio (default: 0.9)

This helps account for memory fragmentation and other inefficiencies in GPU memory allocation.

## Best Practices

1. **Model Selection**
   - For accurate results, use the full model name (e.g., "meta-llama/Llama-2-7b")
   - For unknown models, provide the parameter count if available

2. **Architecture Details**
   - For best accuracy, provide known architecture details
   - The API will use estimates based on parameter count if not provided

3. **Memory Optimization**
   - Consider quantization (Q4, Q8) for larger models
   - Adjust batch size based on your throughput requirements
   - Balance sequence length with memory constraints

4. **Interpretation**
   - Use the recommended memory as a guideline when selecting GPU hardware
   - Consider the warning messages for very large models

## Error Handling

The API returns HTTP 400 errors with detailed messages for invalid inputs:

- Non-positive sequence length or batch size
- Invalid GPU memory utilization ratio
- Unsupported data types or quantization methods

## Technical Details

### Supported Data Types

- `float32`: 4 bytes per parameter
- `float16`: 2 bytes per parameter
- `bfloat16`: 2 bytes per parameter
- `int8`: 1 byte per parameter
- `int4`: 0.5 bytes per parameter

### Quantization Methods

- `Q8`: Reduces model size by a factor of 2
- `Q4`: Reduces model size by a factor of 4
- `null`: No quantization (full precision)

### Architecture Estimation

For models with unknown architectures, the API uses a sophisticated estimation approach:

1. Range-based estimation for number of layers and heads
2. Quadratic equation solving for hidden dimensions
3. Scaling factors for intermediate dimensions

This produces realistic architecture estimates based on parameter count.

## Limitations

- Estimates are approximations and actual memory usage may vary
- The API does not account for specific optimizations in model implementations
- Custom model architectures may require manual parameter specification
- The API focuses on transformer-based models and may not be accurate for other architectures

## Example Use Cases

1. **Hardware Planning**
   - Determine if a specific model can run on available GPU hardware
   - Plan GPU resource allocation for multi-model deployment

2. **Optimization Strategy**
   - Evaluate the impact of quantization on memory requirements
   - Analyze memory-sequence length tradeoffs
   - Optimize batch size for maximum throughput

3. **Cost Estimation**
   - Estimate cloud GPU instance requirements
   - Calculate hosting costs for model serving