from fastapi import FastAPI, HTTPException
import math
import re
from pydantic import BaseModel
from typing import Dict, Optional, List
from huggingface_hub import hf_hub_download
import json
import requests
from functools import lru_cache

app = FastAPI(title="vLLM GPU Memory Calculator API")

class ModelArchitecture(BaseModel):
    """Model architecture details required for calculations"""
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None


class ModelRequest(BaseModel):
    model_name: str
    max_seq_len: Optional[int] = None
    dtype: str = "float16"
    kv_cache_dtype: Optional[str] = None
    max_batch_size: int = 1
    max_model_len: Optional[int] = None
    quantization: Optional[str] = None
    params_billions: Optional[float] = None
    params_millions: Optional[float] = None
    architecture: Optional[ModelArchitecture] = None
    current_gpu_memory_size_gb: Optional[float] = None
    tensor_parallel_size: int = 1


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
    architecture_used: Dict[str, int]
    recommended_gpu_memory_utilization: Optional[float] = None
    context_length_source: str = "user"
    context_length: int
    dtype: str
    kv_cache_dtype: str
    quantization: Optional[str] = None
    batch_size: int
    estimated_values: List[str] = []
    missing_values: List[str] = []
    tensor_parallel_size: int = 1
    per_gpu_memory_breakdown: Dict[str, float] = {}
    per_gpu_total_min_memory_gb: float = 0.0
    per_gpu_total_max_memory_gb: float = 0.0
    recommended_gpu_memory_utilization: Optional[float] = None

DTYPE_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}

QUANTIZATION_FACTORS = {
    "Q8": 2,
    "Q4": 4,
    None: 1
}

@lru_cache(maxsize=100)
def fetch_model_config_from_hf(model_name: str) -> Optional[Dict]:
    """
    Fetch model configuration from Hugging Face Hub.
    Uses caching to avoid repeated requests for the same model.
    
    Args:
        model_name: The name of the model on Hugging Face Hub
        
    Returns:
        Dictionary containing model configuration or None if not found
    """
    try:
        try:
            config_file = hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                repo_type="model"
            )
            with open(config_file, 'r') as f:
                config = json.load(f)
            return config
        except Exception:
            api_url = f"https://huggingface.co/{model_name}/raw/main/config.json"
            response = requests.get(api_url)
            if response.status_code == 200:
                return response.json()
            return None
    except Exception as e:
        print(f"Error fetching config for {model_name}: {str(e)}")
        return None

def get_params_from_hf_config(config: Dict) -> Optional[float]:
    """
    Extract parameter count from Hugging Face config if available.
    
    Args:
        config: Dictionary containing model configuration from Hugging Face
        
    Returns:
        Float representing parameter count in billions or None if not found directly
    """
    param_fields = [
        "num_parameters", 
        "n_params", 
        "num_params", 
        "total_params", 
        "parameter_count", 
        "model_parameters", 
        "total_parameter_count", 
        "params", 
        "n_parameters", 
        "param_count", 
        "parameters"
    ]
    
    for field in param_fields:
        if field in config:
            print(f"Found parameter count in field '{field}': {config[field]}")
            return config[field] / 1e9
    
    return None

def parse_hf_config_to_architecture(config: Dict) -> Dict[str, int]:
    """
    Parse Hugging Face config.json into our architecture format.
    Handles different model architectures and their specific naming conventions.
    
    Args:
        config: Dictionary containing model configuration from Hugging Face
        
    Returns:
        Dictionary containing standardized architecture parameters
    """
    arch = {}
    
    param_mappings = {
        "num_layers": ["num_layers", "n_layer", "num_hidden_layers", "n_layers"],
        "hidden_size": ["hidden_size", "n_embd", "d_model"],
        "num_heads": ["num_attention_heads", "n_head"],
        "head_dim": ["head_dim"],
        "intermediate_size": ["intermediate_size", "ffn_dim", "n_inner"]
    }
    
    for our_key, possible_keys in param_mappings.items():
        for key in possible_keys:
            if key in config:
                arch[our_key] = config[key]
                break
    
    if "architectures" in config:
        model_arch = config["architectures"][0].lower() if config["architectures"] else ""
        model_type = config.get("model_type", "").lower()
        
        if ("llama" in model_type or "llama" in model_arch) and "8b" in config.get("_name_or_path", "").lower():
            arch["num_layers"] = 32
            arch["hidden_size"] = 4096
            arch["num_heads"] = 32
            arch["head_dim"] = 128
            arch["intermediate_size"] = 11008
    
    if "head_dim" not in arch and "hidden_size" in arch and "num_heads" in arch:
        head_dim = arch["hidden_size"] // arch["num_heads"]
        if head_dim > 96:
            head_dim = round(head_dim / 32) * 32
        else:
            head_dim = 2 ** round(math.log2(head_dim))
        arch["head_dim"] = head_dim
    
    if "intermediate_size" not in arch and "hidden_size" in arch:
        if "model_type" in config:
            model_type = config["model_type"].lower()
            if "llama" in model_type:
                arch["intermediate_size"] = int(arch["hidden_size"] * 2.75)
            elif "mistral" in model_type:
                arch["intermediate_size"] = int(arch["hidden_size"] * 3.5)
            else:
                arch["intermediate_size"] = int(arch["hidden_size"] * 4)
        else:
            intermediate_size = int(arch["hidden_size"] * 2.75)
            intermediate_size = round(intermediate_size / 256) * 256
            arch["intermediate_size"] = intermediate_size
    
    return arch

def estimate_params_from_name(model_name: str) -> float:
    """
    Estimate parameter count from model name if not found in Hugging Face.
    Handles both billion (B, b) and million (M, m) parameter specifications.
    """
    billions_match = re.findall(r'(\d+\.?\d*)b', model_name.lower())
    if billions_match:
        return max([float(num) for num in billions_match])
    
    millions_match = re.findall(r'(\d+\.?\d*)m', model_name.lower())
    if millions_match:
        max_millions = max([float(num) for num in millions_match])
        return max_millions / 1000.0
    
    return 7.0
def estimate_model_architecture(params_billions: float) -> Dict[str, int]:
    """
    Estimate model architecture details based on parameter count using scaling laws.
    This matches how models are actually designed in practice, producing more realistic
    architecture parameters, especially for smaller models.
    
    Args:
        params_billions: Model size in billions of parameters
    Returns:
        Dictionary with estimated architecture details
    """
    # Handle special cases for well-known model sizes (e.g., 7B, 8B, 13B, 70B models)
    # These are hardcoded based on common architectures in the industry
    if abs(params_billions - 7) < 0.5 or abs(params_billions - 8) < 0.5:
        return {
            "num_layers": 32,
            "hidden_size": 4096,
            "num_heads": 32,
            "head_dim": 128,
            "intermediate_size": 11008
        }
    elif abs(params_billions - 13) < 0.5:
        return {
            "num_layers": 40,
            "hidden_size": 5120,
            "num_heads": 40,
            "head_dim": 128,
            "intermediate_size": 13824
        }
    elif abs(params_billions - 70) < 1.0:
        return {
            "num_layers": 80,
            "hidden_size": 8192,
            "num_heads": 64,
            "head_dim": 128,
            "intermediate_size": 22016
        }
        
    # For other model sizes, estimate parameters based on size ranges
    # Assign number of layers, heads and vocabulary size based on parameter count
    if params_billions < 0.2:
        num_layers = 12
        num_heads = 8
        vocab = 30000
    elif params_billions < 0.5:
        num_layers = 16
        num_heads = 12
        vocab = 30000
    elif params_billions < 1:
        num_layers = 20
        num_heads = 16
        vocab = 30000
    elif params_billions < 3:
        num_layers = 24
        num_heads = 16
        vocab = 32000
    elif params_billions < 7:
        num_layers = 26
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
        
    # Calculate hidden_size using a quadratic equation derived from parameter count
    # A*hidden_size^2 + B*hidden_size + C = 0 where:
    # A = num_layers * 16 (accounts for weights in attention and MLP blocks)
    # B = 2 * vocab (accounts for embedding and unembedding matrices)
    # C = -params_billions * 1e9 (desired parameter count converted to raw count)
    A = num_layers * 4 + num_layers * 12  # Simplified from num_layers * 16
    B = 2 * vocab
    C = -1 * params_billions * 1e9
    discriminant = B**2 - 4*A*C
    
    # Solve quadratic equation for hidden_size
    if discriminant < 0:
        # Fallback calculation if discriminant is negative (shouldn't happen with valid inputs)
        hidden_size = int(math.sqrt(params_billions * 1e9 / (num_layers * 16)))
    else:
        hidden_size = int((-B + math.sqrt(discriminant)) / (2*A))
    
    # Round hidden_size up to nearest multiple of 128 for hardware efficiency
    hidden_size = math.ceil(hidden_size / 128) * 128
    
    # Determine head dimension and adjust number of heads
    if hidden_size >= 2048:
        head_dim = 128  # Larger models use 128-dimensional heads
    else:
        head_dim = 64   # Smaller models use 64-dimensional heads
    num_heads = hidden_size // head_dim
    
    # Calculate intermediate size (MLP width) based on hidden size
    # Larger models use a higher multiplier for wider MLPs
    if params_billions >= 20:
        multiplier = 3.0  # 3x multiplier for very large models
    else:
        multiplier = 2.75  # 2.75x multiplier for smaller models
        
    # Calculate and round intermediate_size to nearest multiple of 256
    intermediate_size = int(hidden_size * multiplier)
    intermediate_size = round(intermediate_size / 256) * 256
    
    # Return the complete architecture specification
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size
    }

def get_model_params_count(model_name: str, 
                          params_billions: Optional[float] = None,
                          params_millions: Optional[float] = None) -> float:
    """
    Get parameter count for a model in billions, prioritizing:
    1. User-provided values (highest priority)
    2. Hugging Face config (fallback)
    3. Estimation from model name (last resort)
    """
    if params_billions is not None:
        return params_billions
    
    if params_millions is not None:
        return params_millions / 1000.0
    
    hf_config = fetch_model_config_from_hf(model_name)
    if hf_config:
        params_from_config = get_params_from_hf_config(hf_config)
        if params_from_config:
            return params_from_config
    
    return estimate_params_from_name(model_name)

def get_model_architecture(model_name: str, params_billions: float, 
                          user_arch: Optional[ModelArchitecture] = None) -> Dict[str, int]:
    """
    Get model architecture details, prioritizing:
    1. User-provided architecture details (highest priority)
    2. Hugging Face config (fallback)
    3. Estimation based on parameter count (last resort)
    
    Note: User values always override any values from HuggingFace or estimation
    """
    arch = estimate_model_architecture(params_billions)
    
    hf_config = fetch_model_config_from_hf(model_name)
    if hf_config:
        hf_arch = parse_hf_config_to_architecture(hf_config)
        for key, value in hf_arch.items():
            if value is not None:
                arch[key] = value
    
    if user_arch:
        user_arch_dict = user_arch.dict(exclude_unset=True, exclude_none=True)
        for key, value in user_arch_dict.items():
            if value is not None:
                arch[key] = value
    
    return arch

def calculate_model_memory(model_name: str, dtype: str, 
                          quantization: Optional[str] = None,
                          params_billions: Optional[float] = None,
                          params_millions: Optional[float] = None) -> Dict[str, float]:
    """Calculate memory needed for model parameters in GB."""
    params_billions = get_model_params_count(
        model_name, 
        params_billions,
        params_millions
    )
    print(params_billions)
    
    bytes_per_param = DTYPE_SIZES[dtype]
    
    quant_factor = QUANTIZATION_FACTORS.get(quantization, 1)
    
    model_size_bytes = params_billions * 1e9 * bytes_per_param
    model_size_gb = model_size_bytes / (1024**3) / quant_factor * 1.05
    
    return {
        "params_billions": params_billions,
        "model_size_gb": model_size_gb
    }

def calculate_activation_memory_inference(
    arch: Dict[str, int],
    max_batch_size: int,
    max_seq_len: int,
    dtype: str
) -> float:
    """
    Calculate activation memory specifically for vLLM inference, which is much more
    memory-efficient than training or naive inference.
    This calculates for a single GPU without parallelism considerations.
    
    Args:
        arch: Dictionary containing model architecture parameters
            - hidden_size: Model's hidden dimension size
            - num_heads: Number of attention heads
            - num_layers: Number of transformer layers
            - head_dim (optional): Dimension of each attention head
            - intermediate_size (optional): Size of feedforward intermediate layer
        max_batch_size: Maximum number of sequences processed simultaneously
        max_seq_len: Maximum sequence length
        dtype: Data type used for computation (e.g., "float16", "float32")
        
    Returns:
        Estimated activation memory in gigabytes
    """
    # Extract architecture parameters
    hidden_size = arch["hidden_size"]
    num_heads = arch["num_heads"]
    num_layers = arch["num_layers"]
    head_dim = arch.get("head_dim", hidden_size // num_heads)
    intermediate_size = arch.get("intermediate_size", 4 * hidden_size)
    
    # Define dtype sizes in bytes
    DTYPE_SIZES = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "int8": 1
    }
    bytes_per_value = DTYPE_SIZES[dtype]
    
    # For inference, the KV cache is the dominant activation memory component
    # Each layer stores K and V tensors for each token in the sequence
    kv_cache_bytes = (
        2 *  # K and V
        max_batch_size *
        max_seq_len *
        num_layers *
        num_heads *
        head_dim *
        bytes_per_value
    )
    
    # During decoding, we still need some activation memory for the current token
    # This includes attention computations and FFN activations
    # For flash attention, this is much smaller than the KV cache for long sequences
    
    # Attention activations for the current token (Q * K for all heads)
    attn_act_bytes = (
        max_batch_size *
        1 *  # only computing for the new token
        num_heads *
        head_dim *
        bytes_per_value
    )
    
    # FFN activations
    ffn_act_bytes = (
        max_batch_size *
        1 *  # only computing for the new token
        intermediate_size *
        bytes_per_value
    )
    
    # Multiply by number of layers
    per_layer_act_bytes = attn_act_bytes + ffn_act_bytes
    decode_act_bytes = per_layer_act_bytes * num_layers
    
    # Add a small buffer for miscellaneous activations (10%)
    misc_buffer = 0.1 * (kv_cache_bytes + decode_act_bytes)
    
    # Total activation memory
    total_act_bytes = kv_cache_bytes + decode_act_bytes + misc_buffer
    
    # Convert to gigabytes
    total_act_gb = total_act_bytes / (1024**3) + 0.7 # empircal extra overhead
    
    return total_act_gb

def calculate_kv_cache_memory(arch: Dict[str, int], 
                             max_seq_len: int, 
                             max_batch_size: int, 
                             kv_cache_dtype: str,
                             max_model_len: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate the KV cache memory requirements with PagedAttention.
    
    Note: vLLM allocates substantially more memory to KV cache than theoretical
    calculations would suggest - this is by design for performance optimization.
    """
    if max_model_len is None:
        max_model_len = max_seq_len
    
    num_layers = arch["num_layers"]
    hidden_size = arch["hidden_size"]
    
    bytes_per_token = DTYPE_SIZES[kv_cache_dtype]
    
    block_size = 16
    if max_model_len > 8192:
        block_size = 32
    
    num_blocks_per_seq = math.ceil(max_model_len / block_size)
    
    kv_cache_per_token = 2 * hidden_size * bytes_per_token
    
    min_kv_cache_size_bytes = num_layers * max_batch_size * kv_cache_per_token * max_model_len * 0.2
    
    max_kv_cache_size_bytes = num_layers * max_batch_size * kv_cache_per_token * max_model_len
    
    min_kv_cache_size_gb = min_kv_cache_size_bytes / (1024**3)
    max_kv_cache_size_gb = max_kv_cache_size_bytes / (1024**3)
    
    return {
        "min_kv_cache_memory_gb": min_kv_cache_size_gb,
        "max_kv_cache_memory_gb": max_kv_cache_size_gb
    }

def calculate_per_gpu_memory(
    tensor_parallel_size: int,
    model_params_memory_gb: float,
    activation_memory_gb: float,
    min_kv_cache_memory_gb: float,
    max_kv_cache_memory_gb: float,
    cuda_overhead_gb: float,
    params_billions: float
) -> Dict[str, float]:
    """
    Calculate memory requirements per GPU when using tensor parallelism.
    
    Improved based on actual vLLM logs to better reflect real-world distributions:
    - Model parameters are not perfectly distributed (added overhead)
    - Activation memory varies between primary and worker GPUs
    - KV cache is based on the actual model requirements
    - CUDA overhead varies with model size
    
    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism
        model_params_memory_gb: Total model parameters memory in GB
        activation_memory_gb: Total activation memory in GB
        min_kv_cache_memory_gb: Minimum KV cache memory in GB
        max_kv_cache_memory_gb: Maximum KV cache memory in GB
        cuda_overhead_gb: CUDA overhead in GB
        params_billions: Model size in billions of parameters
        
    Returns:
        Dictionary with per-GPU memory requirements
    """
    if tensor_parallel_size <= 1:        
        return {
            "per_gpu_model_params": model_params_memory_gb,
            "per_gpu_activation": activation_memory_gb,
            "per_gpu_min_kv_cache": min_kv_cache_memory_gb,
            "per_gpu_max_kv_cache": max_kv_cache_memory_gb,
            "per_gpu_cuda_overhead": cuda_overhead_gb,
            "per_gpu_primary_activation": activation_memory_gb,
        }
    
    per_gpu_model_params = (model_params_memory_gb / tensor_parallel_size)
    
    communication_overhead = 0.20
    sync_buffer_overhead = 0.15
    staging_overhead = 0.07
    metadata_overhead = 0.05
    total_additional_overhead = communication_overhead + sync_buffer_overhead + staging_overhead + metadata_overhead

    primary_gpu_activation = activation_memory_gb / (5 + tensor_parallel_size - 1) * 5 * (1 + total_additional_overhead) * 1.1 # 1.1 is a safety factor
    worker_gpu_activation = activation_memory_gb / (5 + tensor_parallel_size - 1) * (1 + total_additional_overhead) * 1.1 # 1.1 is a safety factor
   
    per_gpu_activation = primary_gpu_activation
    
    per_gpu_min_kv_cache = min_kv_cache_memory_gb / tensor_parallel_size
    per_gpu_max_kv_cache = max_kv_cache_memory_gb / tensor_parallel_size
    
    per_gpu_cuda_overhead = cuda_overhead_gb / tensor_parallel_size * 1.5
    
    return {
        "per_gpu_model_params": per_gpu_model_params,
        "per_gpu_activation": per_gpu_activation,
        "per_gpu_min_kv_cache": per_gpu_min_kv_cache,
        "per_gpu_max_kv_cache": per_gpu_max_kv_cache,
        "per_gpu_cuda_overhead": per_gpu_cuda_overhead,
    }

def get_context_length_from_config(config: Dict) -> Optional[int]:
    """
    Extract the context window size from a model config.
    
    Args:
        config: Dictionary containing model configuration from Hugging Face
        
    Returns:
        Integer representing max sequence length or None if not found
    """
    context_length_fields = [
        "max_position_embeddings",
        "max_sequence_length", 
        "context_length",
        "max_seq_len",
        "window_size",
        "n_positions"
    ]
    
    for field in context_length_fields:
        if field in config:
            return config[field]
    
    if "rope_scaling" in config:
        base_length = config.get("max_position_embeddings", 0)
        scaling_factor = config["rope_scaling"].get("factor", 1)
        if base_length and scaling_factor > 1:
            return int(base_length * scaling_factor)
    
    return None

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
    estimated_values = []
    missing_values = []
    
    if request.tensor_parallel_size <= 0:
        raise HTTPException(status_code=400, detail="tensor_parallel_size must be positive")
    
    hf_config = fetch_model_config_from_hf(request.model_name)
    if hf_config is None:
        estimated_values.append("model_configuration")
    
    config_context_length = None
    if hf_config:
        config_context_length = get_context_length_from_config(hf_config)
        if config_context_length is None:
            estimated_values.append("context_length_from_config")
    
    if request.max_seq_len is not None:
        effective_max_seq_len = request.max_seq_len
    elif config_context_length:
        effective_max_seq_len = config_context_length
        estimated_values.append("context_length_from_config")
    else:
        raise HTTPException(
            status_code=400, 
            detail="No sequence length available. Please provide max_seq_len as this model's config does not contain context length information."
        )
    
    if effective_max_seq_len <= 0:
        raise HTTPException(status_code=400, detail="max_seq_len must be positive")
    
    if request.max_model_len is not None:
        max_model_len = request.max_model_len
    else:
        max_model_len = effective_max_seq_len
        
    if max_model_len is None:
        max_model_len = effective_max_seq_len
    
    if request.max_batch_size <= 0:
        raise HTTPException(status_code=400, detail="max_batch_size must be positive")
    if request.params_millions is not None and request.params_billions is None:
        request.params_billions = request.params_millions / 1000.0
    
    if request.kv_cache_dtype is None:
        request.kv_cache_dtype = request.dtype
    
    if request.dtype not in DTYPE_SIZES:
        raise HTTPException(status_code=400, detail=f"Unsupported dtype: {request.dtype}. Supported types: {list(DTYPE_SIZES.keys())}")
    
    if request.kv_cache_dtype not in DTYPE_SIZES:
        raise HTTPException(status_code=400, detail=f"Unsupported KV cache dtype: {request.kv_cache_dtype}. Supported types: {list(DTYPE_SIZES.keys())}")
    
    if request.quantization is not None and request.quantization not in QUANTIZATION_FACTORS:
        raise HTTPException(status_code=400, detail=f"Unsupported quantization: {request.quantization}. Supported values: Q4, Q8, None")
    
    model_info = calculate_model_memory(
        request.model_name, 
        request.dtype, 
        request.quantization,
        request.params_billions,
        request.params_millions
    )
    
    model_params_memory_gb = model_info["model_size_gb"]
    params_billions = model_info["params_billions"]
    
    if request.params_billions is None and request.params_millions is None:
        if hf_config:
            params_from_config = get_params_from_hf_config(hf_config)
            if params_from_config is None:
                estimated_values.append("parameter_count")
        else:
            estimated_values.append("parameter_count")
    
    architecture = get_model_architecture(
        request.model_name,
        params_billions,
        request.architecture
    )
    
    if request.architecture is None:
        hf_config = fetch_model_config_from_hf(request.model_name)
        if hf_config:
            hf_arch = parse_hf_config_to_architecture(hf_config)
            if not hf_arch or len(hf_arch) < 3:
                estimated_values.append("model_architecture")
        else:
            estimated_values.append("model_architecture")
    
    activation_memory_gb = calculate_activation_memory_inference(
        architecture,
        request.max_batch_size,
        effective_max_seq_len,
        request.dtype
    )
    
    kv_cache_memory = calculate_kv_cache_memory(
        architecture,
        effective_max_seq_len,
        request.max_batch_size,
        request.kv_cache_dtype,
        max_model_len
    )
    
    min_kv_cache_memory_gb = kv_cache_memory["min_kv_cache_memory_gb"]
    max_kv_cache_memory_gb = kv_cache_memory["max_kv_cache_memory_gb"]
    
    cuda_overhead_gb = 0.75
    
    if request.quantization:
        cuda_overhead_gb += 0.25
    
    total_min_memory_gb = (
        model_params_memory_gb + 
        activation_memory_gb + 
        min_kv_cache_memory_gb + 
        cuda_overhead_gb
    )
    
    total_max_memory_gb = (
        model_params_memory_gb + 
        activation_memory_gb + 
        max_kv_cache_memory_gb + 
        cuda_overhead_gb
    )
    
    recommended_memory_gb = total_min_memory_gb
    
    per_gpu_memory = calculate_per_gpu_memory(
        request.tensor_parallel_size,
        model_params_memory_gb,
        activation_memory_gb,
        min_kv_cache_memory_gb,
        max_kv_cache_memory_gb,
        cuda_overhead_gb,
        params_billions 
    )
    
    per_gpu_total_min_memory_gb = round(
        per_gpu_memory["per_gpu_model_params"] +
        per_gpu_memory["per_gpu_activation"] +
        per_gpu_memory["per_gpu_min_kv_cache"] +
        per_gpu_memory["per_gpu_cuda_overhead"],
        2
    )
    
    per_gpu_total_max_memory_gb = round(
        per_gpu_memory["per_gpu_model_params"] +
        per_gpu_memory["per_gpu_activation"] +
        per_gpu_memory["per_gpu_max_kv_cache"] +
        per_gpu_memory["per_gpu_cuda_overhead"],
        2
    )
    
    model_params_memory_gb = round(model_params_memory_gb, 2)
    activation_memory_gb = round(activation_memory_gb, 2)
    min_kv_cache_memory_gb = round(min_kv_cache_memory_gb, 2)
    max_kv_cache_memory_gb = round(max_kv_cache_memory_gb, 2)
    total_min_memory_gb = round(total_min_memory_gb, 2)
    total_max_memory_gb = round(total_max_memory_gb, 2)
    recommended_memory_gb = round(recommended_memory_gb, 2)
    
    for key in per_gpu_memory:
        per_gpu_memory[key] = round(per_gpu_memory[key], 2)
    
    context_length_source = "config"
    if request.max_seq_len is not None:
        context_length_source = "user"
    
    components_breakdown = {
        "model_params": model_params_memory_gb,
        "activation": activation_memory_gb,
        "min_kv_cache": min_kv_cache_memory_gb,
        "max_kv_cache": max_kv_cache_memory_gb,
        "cuda_overhead": cuda_overhead_gb
    }
    
    warning = None
    if estimated_values:
        if "parameter_count" in estimated_values and "model_architecture" in estimated_values:
            warning = "Important model details (parameter count and architecture) were estimated and may be inaccurate. Consider providing these values manually for more precise calculations."
        elif "parameter_count" in estimated_values:
            warning = "Model parameter count was estimated from the model name and may be inaccurate. Consider providing params_billions or params_millions for more precise calculations."
        elif "model_architecture" in estimated_values:
            warning = "Model architecture details were estimated and may be inaccurate. Consider providing architecture details for more precise calculations."
    
    if missing_values:
        if warning:
            warning += f" Missing values that could not be determined: {', '.join(missing_values)}."
        else:
            warning = f"Missing values that could not be determined: {', '.join(missing_values)}."
    
    if recommended_memory_gb > 80 and not warning:
        warning = "This model may require multiple GPUs or a high-end GPU with sufficient memory."
    elif recommended_memory_gb > 80:
        warning += " This model may require multiple GPUs or a high-end GPU with sufficient memory."
    
    if request.tensor_parallel_size > 1:
        multi_gpu_warning = f"Memory requirements calculated for tensor parallelism across {request.tensor_parallel_size} GPUs."
        if warning:
            warning += " " + multi_gpu_warning
        else:
            warning = multi_gpu_warning
    
    recommended_gpu_memory_utilization = None
    if request.current_gpu_memory_size_gb:
        if request.current_gpu_memory_size_gb <= 0:
            raise HTTPException(status_code=400, detail="current_gpu_memory_size_gb must be positive")
        
        if request.tensor_parallel_size > 1:
            util_ratio = per_gpu_total_min_memory_gb / request.current_gpu_memory_size_gb
        else:
            util_ratio = total_min_memory_gb / request.current_gpu_memory_size_gb
        
        if util_ratio > 0.97:
            if warning:
                warning += " Model requires more memory than available on your GPU. Consider using a larger GPU, reducing batch size, or applying quantization."
            else:
                warning = "Model requires more memory than available on your GPU. Consider using a larger GPU, reducing batch size, or applying quantization."
            recommended_gpu_memory_utilization = None
        else:
            recommended_gpu_memory_utilization = round(util_ratio + 0.02, 2)
            recommended_gpu_memory_utilization = min(recommended_gpu_memory_utilization, 0.95)

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
        recommended_gpu_memory_utilization=recommended_gpu_memory_utilization,
        warning=warning,
        components_breakdown=components_breakdown,
        architecture_used=architecture,
        context_length_source=context_length_source,
        context_length=effective_max_seq_len,
        dtype=request.dtype,
        kv_cache_dtype=request.kv_cache_dtype,
        quantization=request.quantization,
        batch_size=request.max_batch_size,
        estimated_values=estimated_values,
        missing_values=missing_values,
        tensor_parallel_size=request.tensor_parallel_size,
        per_gpu_memory_breakdown=per_gpu_memory,
        per_gpu_total_min_memory_gb=per_gpu_total_min_memory_gb,
        per_gpu_total_max_memory_gb=per_gpu_total_max_memory_gb
    )

@app.get("/search-models")
def search_models(query: str = "", limit: int = 10):
    """
    Search for models on Hugging Face based on a query.
    Returns model names matching the search criteria.
    """
    try:
        url = f"https://huggingface.co/api/models?search={query}&limit={limit}&filter=text-generation"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            models = response.json()
            model_names = [model['modelId'] for model in models]
            return {
                "models": model_names
            }
        else:
            return {
                "error": f"API request failed with status code: {response.status_code}",
                "models": []
            }
    except Exception as e:
        return {
            "error": str(e),
            "models": []
        }

@app.get("/")
def read_root():
    return {
        "info": "vLLM GPU Memory Calculator API",
        "usage": "POST to /estimate-gpu-memory with model details",
        "example_request": {
            "model_name": "meta-llama/Llama-3-8b",
            "dtype": "float16",
            "tensor_parallel_size": 2
        },
        "features": [
            "Auto-fetches model configurations from Hugging Face Hub",
            "Auto-detects context length from model config when available",
            "Supports user-provided architecture details and overrides",
            "Intelligently estimates model parameters if not provided",
            "Calculates memory requirements for vLLM deployments",
            "Provides detailed feedback on estimated or missing values",
            "Supports tensor parallelism across multiple GPUs"
        ],
        "endpoints": {
            "/estimate-gpu-memory": "POST - Calculate GPU memory requirements",
            "/search-models": "GET - Search for models on Hugging Face"
        },
        "notes": [
            "The sequence length (max_seq_len) parameter is optional - if not provided, the API will try to get it from the model's config",
            "If the model's config doesn't have context length information, you must provide max_seq_len",
            "For multi-GPU setups, specify the 'tensor_parallel_size' parameter to see per-GPU memory requirements"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)