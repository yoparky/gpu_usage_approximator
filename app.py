from fastapi import FastAPI, HTTPException
import math
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI(title="vLLM GPU Memory Calculator API")
# General equation:
# total_memory = model_memory + activation_memory + kv_cache_memory + cuda_overhead
# https://github.com/RahulSChand/gpu_poor 참조

class ModelArchitecture(BaseModel):
    """Optional model architecture details if known by the user"""
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None  # For FFN layers


class ModelRequest(BaseModel):
    model_name: str
    max_seq_len: int
    dtype: str = "float16"  # Default to float16
    kv_cache_dtype: Optional[str] = None  # If None, will use the same as dtype
    max_batch_size: int = 32
    max_model_len: Optional[int] = None  # If None, will use max_seq_len
    gpu_memory_utilization: float = 0.9  # Default GPU memory utilization ratio
    quantization: Optional[str] = None  # Q4, Q8, or None for full precision
    params_billions: Optional[float] = None  # Optional override for parameter count
    architecture: Optional[ModelArchitecture] = None  # Optional model architecture details


class MemoryEstimation(BaseModel):
    model_name: str
    model_params_memory_gb: float
    activation_memory_gb: float
    min_kv_cache_memory_gb: float
    max_kv_cache_memory_gb: float
    cuda_overhead_gb: float
    total_min_memory_gb: float
    total_max_memory_gb: float
    recommended_memory_gb: float
    warning: Optional[str] = None
    components_breakdown: Dict[str, float]
    architecture_used: Dict[str, int]  # The architecture values used in the calculation


# Model parameter counts for common models (in billions)
MODEL_PARAMS = {
    # LLaMA family (existing entries, plus missing ones)
    "meta-llama/Llama-2-7b": 7,
    "meta-llama/Llama-2-13b": 13,
    "meta-llama/Llama-2-70b": 70,
    "meta-llama/Llama-2-7b-chat": 7,
    "meta-llama/Llama-2-13b-chat": 13,
    "meta-llama/Llama-2-70b-chat": 70,
    "meta-llama/Llama-3-8b": 8,
    "meta-llama/Llama-3-70b": 70,
    "meta-llama/Llama-3-8b-instruct": 8,
    "meta-llama/Llama-3-70b-instruct": 70,

    # Add Llama 3.2 family (new)
    "meta-llama/Llama-3.2-1B": 1.23,       # Actual parameter count from model card
    "meta-llama/Llama-3.2-3B": 3.21,       # Actual parameter count from model card
    "meta-llama/Llama-3.2-1B-instruct": 1.23,
    "meta-llama/Llama-3.2-3B-instruct": 3.21,
    
    # Adding more LLaMA models
    "meta-llama/Llama-3.1-8b": 8,
    "meta-llama/Llama-3.1-70b": 70,
    "meta-llama/Llama-3.1-405b": 405,
    "meta-llama/Llama-3.1-8b-instruct": 8,
    "meta-llama/Llama-3.1-70b-instruct": 70,
    "meta-llama/Llama-3.1-405b-instruct": 405,
    
    # Mistral family (existing entries)
    "mistralai/Mistral-7B-v0.1": 7,
    "mistralai/Mixtral-8x7B-v0.1": 47,  # 8 experts of 7B each, but not all loaded at once
    "mistralai/Mistral-7B-Instruct-v0.1": 7,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 47,
    
    # Adding more Mistral models
    "mistralai/Mistral-Large-2-128k": 32,
    "mistralai/Mistral-Large-Instruct-2-128k": 32,
    "mistralai/Mistral-Medium-2-128k": 7,
    "mistralai/Mistral-Medium-Instruct-2-128k": 7,
    "mistralai/Mixtral-8x22B-v0.1": 176,  # 8 experts of 22B each
    
    # OpenAI models
    "openai/gpt-3.5-turbo": 20,  # Estimated
    "openai/gpt-3.5-turbo-16k": 20,  # Estimated
    "openai/gpt-4": 1500,  # Estimated
    "openai/gpt-4-turbo": 1700,  # Estimated
    "openai/gpt-4o": 1800,  # Estimated
    "openai/text-davinci-003": 175,  # Estimated
    
    # Qwen family
    "Qwen/Qwen-1.5-0.5B": 0.5,
    "Qwen/Qwen-1.5-1.8B": 1.8,
    "Qwen/Qwen-1.5-4B": 4,
    "Qwen/Qwen-1.5-7B": 7,
    "Qwen/Qwen-1.5-14B": 14,
    "Qwen/Qwen-1.5-32B": 32,
    "Qwen/Qwen-1.5-72B": 72,
    "Qwen/Qwen-1.5-110B": 110,
    
    # Gemma family
    "google/gemma-2b": 2,
    "google/gemma-7b": 7,
    "google/gemma-2b-instruct": 2,
    "google/gemma-7b-instruct": 7,
    
    # Claude family (Anthropic)
    "anthropic/claude-3-opus": 180,  # Estimated
    "anthropic/claude-3-sonnet": 100,  # Estimated
    "anthropic/claude-3-haiku": 40,  # Estimated
    
    # Gemini family
    "google/gemini-nano": 3.25,  # Estimated
    "google/gemini-pro": 50,  # Estimated
    "google/gemini-ultra": 500,  # Estimated
    
    # Yi family
    "01-ai/Yi-6B": 6,
    "01-ai/Yi-9B": 9,
    "01-ai/Yi-34B": 34,
    
    # Falcon family
    "tiiuae/falcon-7b": 7,
    "tiiuae/falcon-40b": 40,
    "tiiuae/falcon-180b": 180,
    
    # Other existing models
    "stabilityai/stablelm-tuned-alpha-7b": 7,
    "mosaicml/mpt-7b": 7,
    "mosaicml/mpt-30b": 30,
    "EleutherAI/pythia-12b": 12,
    "Together/falcon-40b-instruct": 40,
    "openchat/openchat-3.5": 7,
    
    # Default fallback
    "unknown": 0  # Will be estimated based on name or set by user
}

# Architecture details for known models
MODEL_ARCHITECTURES = {
    # LLaMA-2 family
    "meta-llama/Llama-2-7b": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "intermediate_size": 11008
    },
    "meta-llama/Llama-2-13b": {
        "num_layers": 40,
        "hidden_size": 5120,
        "num_heads": 40,
        "head_dim": 128,
        "intermediate_size": 13824
    },
    "meta-llama/Llama-2-70b": {
        "num_layers": 80,
        "hidden_size": 8192,
        "num_heads": 64,
        "head_dim": 128,
        "intermediate_size": 28672
    },
    # llama 계통 3
    # Add Llama 3.2 family (new)
    "meta-llama/Llama-3.2-1B": {
        "num_layers": 24,           # Based on typical architecture scaling
        "hidden_size": 2048,        # Based on typical architecture scaling
        "num_heads": 16,            # Common head configuration for this size
        "head_dim": 128,            # Maintained from Llama family
        "intermediate_size": 5632   # Based on ~2.75x hidden_size ratio
    },
    "meta-llama/Llama-3.2-3B": {
        "num_layers": 26,           # Based on typical architecture scaling
        "hidden_size": 3072,        # Based on typical architecture scaling
        "num_heads": 24,            # Common head configuration for this size
        "head_dim": 128,            # Maintained from Llama family
        "intermediate_size": 8448   # Based on ~2.75x hidden_size ratio
    },
    
    # openapi 계통
    # Qwen 계통 1.5
    # Gemma 계통

    # Mistral-7B
    "mistralai/Mistral-7B-v0.1": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "intermediate_size": 14336
    }
    
    # Add more known model architectures as needed
}


# Memory sizes for different dtypes in bytes
DTYPE_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,  # 4 bits = 0.5 bytes
}


# Quantization factors (how much smaller the model becomes)
QUANTIZATION_FACTORS = {
    "Q8": 2,  # Model size is divided by 2 for Q8 quantization
    "Q4": 4,  # Model size is divided by 4 for Q4 quantization
    None: 1   # No quantization, keep original size
}

def estimate_model_architecture(params_billions: float) -> Dict[str, int]:
    """
    Estimate model architecture details based on parameter count using a more accurate
    approach that combines range-based estimation with quadratic equation solving.
    
    This matches how models are actually designed in practice, producing more realistic
    architecture parameters, especially for smaller models.
    
    The approach to this estimation was based on 
    # https://github.com/RahulSChand/gpu_poor

    Args:
        params_billions: Model size in billions of parameters
        
    Returns:
        Dictionary with estimated architecture details
    """
    # Step 1: Determine baseline architecture parameters based on model size ranges
    # These values come from analyzing real-world model architectures
    if params_billions < 3:
        num_layers = 24
        num_heads = 16
        vocab = 32000
    elif params_billions < 7:
        num_layers = 26      # Appropriate for models like Llama-3.2-3B
        num_heads = 24
        vocab = 32000
    elif params_billions < 10:
        num_layers = 32
        num_heads = 32
        vocab = 32000
    elif params_billions < 20:
        num_layers = 40
        num_heads = 40
        vocab = 32000
    elif params_billions < 50:
        num_layers = 60
        num_heads = 52
        vocab = 32000
    else:
        num_layers = 80
        num_heads = 64
        vocab = 32000
    
    # Step 2: Solve quadratic equation to find the hidden dimension
    # The equation models how parameters are distributed in transformer architectures:
    # vocab*h*2 + numLayers*4*h*h + numLayers*3*4*h*h = params_billions*10^9
    
    # Rearrange to standard form: Ah² + Bh + C = 0
    A = num_layers * 4 + num_layers * 12  # Coefficient of h²
    B = 2 * vocab                        # Coefficient of h (input & output embeddings)
    C = -1 * params_billions * 1e9       # Constant term
    
    # Apply quadratic formula to solve for h (hidden size)
    discriminant = B**2 - 4*A*C
    if discriminant < 0:
        # Fallback in case the equation doesn't have a real solution
        hidden_size = int(math.sqrt(params_billions * 1e9 / (num_layers * 16)))
    else:
        # Solve using quadratic formula
        hidden_size = int((-B + math.sqrt(discriminant)) / (2*A))
    
    # Step 3: Ensure hidden_size is divisible by num_heads for clean head dimensions
    head_dim = max(64, hidden_size // num_heads)
    hidden_size = head_dim * num_heads
    
    # Step 4: Calculate intermediate size based on hidden size
    # Most modern models use a multiplier between 2.5-4×
    # Llama-2 uses ~2.7×, Llama-3.2 uses ~2.75×, Mistral uses ~3.5×
    intermediate_size = int(hidden_size * 2.75)
    
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size
    }


def estimate_params_from_name(model_name: str) -> float:
    """Estimate parameter count from model name if not in our database."""
    # Try to find numbers in the model name (like 7b, 13b, etc.)
    import re
    numbers = re.findall(r'(\d+)b', model_name.lower())
    if numbers:
        # Take the largest number followed by 'b' as parameter count in billions
        return max([float(num) for num in numbers])
    return 7  # Default to 7B if we can't determine


def get_model_params_count(model_name: str, params_billions: Optional[float] = None) -> float:
    """Get parameter count for a model in billions."""
    # If user provided parameter count, use it
    if params_billions is not None:
        return params_billions
    
    # Otherwise check our database
    if model_name in MODEL_PARAMS:
        return MODEL_PARAMS[model_name]
    
    # Try to find the model by its short name
    short_name = model_name.split('/')[-1].lower()
    for known_model, params in MODEL_PARAMS.items():
        if short_name in known_model.lower():
            return params
    
    # If still not found, try to estimate from the name
    return estimate_params_from_name(model_name)


def get_model_architecture(model_name: str, params_billions: float, 
                          user_arch: Optional[ModelArchitecture] = None) -> Dict[str, int]:
    """
    Get model architecture details, with priority:
    1. User-provided values
    2. Known architecture from database
    3. Estimated values based on parameter count
    """
    # Start with estimated architecture
    arch = estimate_model_architecture(params_billions)
    
    # Update with known architecture if available
    if model_name in MODEL_ARCHITECTURES:
        for key, value in MODEL_ARCHITECTURES[model_name].items():
            arch[key] = value
    
    # Update with any user-provided architecture details
    if user_arch:
        user_arch_dict = user_arch.dict(exclude_unset=True, exclude_none=True)
        for key, value in user_arch_dict.items():
            if value is not None:
                arch[key] = value
    
    return arch


def calculate_model_memory(model_name: str, dtype: str, 
                          quantization: Optional[str] = None,
                          params_billions: Optional[float] = None) -> Dict[str, float]:
    """Calculate memory needed for model parameters in GB."""
    # Get parameter count (using user override if provided)
    params_billions = get_model_params_count(model_name, params_billions)
    bytes_per_param = DTYPE_SIZES[dtype]
    
    # Apply quantization factor if specified
    quant_factor = QUANTIZATION_FACTORS.get(quantization, 1)
    
    # Model size in GB (with slight overhead)
    model_size_bytes = params_billions * 1e9 * bytes_per_param
    model_size_gb = model_size_bytes / (1024**3) / quant_factor * 1.05  # 5% overhead for model tensors
    
    return {
        "params_billions": params_billions,
        "model_size_gb": model_size_gb
    }


def calculate_activation_memory(arch: Dict[str, int],
                               max_batch_size: int, 
                               max_seq_len: int, 
                               dtype: str) -> float:
    """
    Calculate activation memory for inference.
    This accounts for both hidden_size and intermediate_size activations.
    """
    num_layers = arch["num_layers"]
    hidden_size = arch["hidden_size"]
    intermediate_size = arch["intermediate_size"]
    
    # Bytes per value based on dtype
    bytes_per_value = DTYPE_SIZES[dtype]
    # Assume 5 activations per layer
    # For hidden_size activations (query, key, value, attention output, etc.)
    # Typically 3 activations per layer use hidden_size
    hidden_activation_bytes = 3 * max_batch_size * max_seq_len * hidden_size * bytes_per_value
    
    # For intermediate_size activations (FFN intermediate outputs)
    # Typically 2 activations per layer use intermediate_size
    intermediate_activation_bytes = 2 * max_batch_size * max_seq_len * intermediate_size * bytes_per_value
    
    # Total activation memory across all layers
    total_activation_bytes = num_layers * (hidden_activation_bytes + intermediate_activation_bytes)
    
    # Convert to GB
    activation_memory_gb = total_activation_bytes / (1024**3)
    
    return activation_memory_gb


def calculate_kv_cache_memory(arch: Dict[str, int], 
                             max_seq_len: int, 
                             max_batch_size: int, 
                             kv_cache_dtype: str,
                             max_model_len: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate the KV cache memory requirements with PagedAttention.
    """
    if max_model_len is None:
        max_model_len = max_seq_len
    
    num_layers = arch["num_layers"]
    hidden_size = arch["hidden_size"]
    
    # Calculate KV cache size
    bytes_per_token = DTYPE_SIZES[kv_cache_dtype]
    
    # In PagedAttention, memory is allocated in blocks
    # Each block typically holds 16 tokens for efficient memory management
    block_size = 16
    num_blocks_per_seq = math.ceil(max_model_len / block_size)
    
    # Key + Value cache per layer per token
    # vLLM only needs to store K and V, not Q (query) vectors
    kv_cache_per_token = 2 * hidden_size * bytes_per_token  # 2 for K and V
    
    # Total KV cache size for minimum allocation (one block per sequence)
    min_kv_cache_size_bytes = num_layers * max_batch_size * kv_cache_per_token * block_size
    
    # Total KV cache size for maximum allocation (all blocks per sequence)
    max_kv_cache_size_bytes = num_layers * max_batch_size * kv_cache_per_token * max_model_len
    
    # Convert to GB
    min_kv_cache_size_gb = min_kv_cache_size_bytes / (1024**3)
    max_kv_cache_size_gb = max_kv_cache_size_bytes / (1024**3)
    
    return {
        "min_kv_cache_memory_gb": min_kv_cache_size_gb,
        "max_kv_cache_memory_gb": max_kv_cache_size_gb
    }


@app.post("/estimate-gpu-memory", response_model=MemoryEstimation)
def estimate_gpu_memory(request: ModelRequest) -> MemoryEstimation:
    """
    Estimate GPU memory usage for serving a specific model with vLLM.
    
    This takes into account:
    1. Model parameters memory
    2. Activation memory for inference
    3. KV cache memory (with PagedAttention)
    4. CUDA context and other overhead
    """
    # Add at start of function
    if request.max_seq_len <= 0:
        raise HTTPException(status_code=400, detail="max_seq_len must be positive")
    if request.max_batch_size <= 0:
        raise HTTPException(status_code=400, detail="max_batch_size must be positive")
    if not (0 < request.gpu_memory_utilization <= 1):
        raise HTTPException(status_code=400, detail="gpu_memory_utilization must be between 0 and 1")
    
    # Validate and set default values
    if request.kv_cache_dtype is None:
        request.kv_cache_dtype = request.dtype
    
    if request.max_model_len is None:
        request.max_model_len = request.max_seq_len
    
    # Ensure dtype values are valid
    if request.dtype not in DTYPE_SIZES:
        raise HTTPException(status_code=400, detail=f"Unsupported dtype: {request.dtype}. Supported types: {list(DTYPE_SIZES.keys())}")
    
    if request.kv_cache_dtype not in DTYPE_SIZES:
        raise HTTPException(status_code=400, detail=f"Unsupported KV cache dtype: {request.kv_cache_dtype}. Supported types: {list(DTYPE_SIZES.keys())}")
    
    if request.quantization is not None and request.quantization not in QUANTIZATION_FACTORS:
        raise HTTPException(status_code=400, detail=f"Unsupported quantization: {request.quantization}. Supported values: Q4, Q8, None")
    
    # Calculate model parameters memory
    model_info = calculate_model_memory(
        request.model_name, 
        request.dtype, 
        request.quantization,
        request.params_billions
    )
    
    model_params_memory_gb = model_info["model_size_gb"]
    params_billions = model_info["params_billions"]
    
    # Get model architecture (prioritizing user-provided values)
    architecture = get_model_architecture(
        request.model_name,
        params_billions,
        request.architecture
    )
    
    # Calculate activation memory
    activation_memory_gb = calculate_activation_memory(
        architecture,
        request.max_batch_size,
        request.max_seq_len,
        request.dtype
    )
    
    # Calculate KV cache memory (min and max)
    kv_cache_memory = calculate_kv_cache_memory(
        architecture,
        request.max_seq_len,
        request.max_batch_size,
        request.kv_cache_dtype,
        request.max_model_len
    )
    
    # CUDA context and other overhead
    cuda_overhead_gb = 0.75  # 750 MB as suggested
    
    # Additional overhead for quantization libraries if using quantization
    if request.quantization:
        cuda_overhead_gb += 0.25  # Add 250 MB more for quantization libraries
    
    # Calculate total memory requirements
    total_min_memory_gb = (
        model_params_memory_gb + 
        activation_memory_gb + 
        kv_cache_memory["min_kv_cache_memory_gb"] + 
        cuda_overhead_gb
    )
    
    total_max_memory_gb = (
        model_params_memory_gb + 
        activation_memory_gb + 
        kv_cache_memory["max_kv_cache_memory_gb"] + 
        cuda_overhead_gb
    )
    
    # Calculate recommended memory with the specified utilization ratio
    recommended_memory_gb = total_max_memory_gb / request.gpu_memory_utilization
    
    # Round values for better readability while preserving 2 decimal places for precision
    model_params_memory_gb = round(model_params_memory_gb, 2)
    activation_memory_gb = round(activation_memory_gb, 2)
    min_kv_cache_memory_gb = round(kv_cache_memory["min_kv_cache_memory_gb"], 2)
    max_kv_cache_memory_gb = round(kv_cache_memory["max_kv_cache_memory_gb"], 2)
    total_min_memory_gb = round(total_min_memory_gb, 2)
    total_max_memory_gb = round(total_max_memory_gb, 2)
    recommended_memory_gb = round(recommended_memory_gb, 2)
    
    # Detailed breakdown of memory components
    components_breakdown = {
        "model_params": model_params_memory_gb,
        "activation": activation_memory_gb,
        "min_kv_cache": min_kv_cache_memory_gb,
        "max_kv_cache": max_kv_cache_memory_gb,
        "cuda_overhead": cuda_overhead_gb
    }
    
    # Create a warning if the model seems unusually large
    warning = None
    if recommended_memory_gb > 80:
        warning = "This model may require multiple GPUs or a high-end GPU with sufficient memory."
    
    return MemoryEstimation(
        model_name=request.model_name,
        model_params_memory_gb=model_params_memory_gb,
        activation_memory_gb=activation_memory_gb,
        min_kv_cache_memory_gb=min_kv_cache_memory_gb,
        max_kv_cache_memory_gb=max_kv_cache_memory_gb,
        cuda_overhead_gb=cuda_overhead_gb,
        total_min_memory_gb=total_min_memory_gb,
        total_max_memory_gb=total_max_memory_gb,
        recommended_memory_gb=recommended_memory_gb,
        warning=warning,
        components_breakdown=components_breakdown,
        architecture_used=architecture
    )


# Additional endpoint to get known model architectures
@app.get("/known-architectures")
def get_known_architectures():
    return {
        "models": MODEL_ARCHITECTURES
    }


# Example usage documentation endpoint
@app.get("/")
def read_root():
    return {
        "info": "vLLM GPU Memory Calculator API",
        "usage": "POST to /estimate-gpu-memory with model details",
        "example_request": {
            "model_name": "meta-llama/Llama-2-7b",
            "max_seq_len": 2048,
            "dtype": "float16",
            "max_batch_size": 32,
            "gpu_memory_utilization": 0.9,
            "quantization": None,  # Optional: "Q4" or "Q8" for quantized models
            "params_billions": None,  # Optional: Override parameter count if known
            "architecture": {  # Optional: Provide known architecture details
                "num_layers": 32,
                "hidden_size": 4096,
                "num_heads": 32,
                "head_dim": 128,
                "intermediate_size": 11008
            }
        },
        "memory_components": {
            "model_params": "Memory for model weights",
            "activation": "Memory for intermediate activations during inference",
            "kv_cache": "Memory for key-value cache (min and max with PagedAttention)",
            "cuda_overhead": "Memory for CUDA context and additional libraries"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)