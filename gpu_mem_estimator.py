from fastapi import FastAPI, HTTPException
import math
import re
from pydantic import BaseModel
from typing import Dict, Optional, List
from huggingface_hub import hf_hub_download
import json
import requests
from functools import lru_cache

app = FastAPI(title="vLLM GPU 메모리 계산기 API")

class ModelArchitecture(BaseModel):
    """계산에 필요한 모델 아키텍처 세부 정보"""
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
    context_length_source: str = "사용자"
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
    Hugging Face Hub에서 모델 구성을 가져옵니다.
    
    Args:
        model_name: Hugging Face Hub의 모델 이름
        
    Returns:
        모델 구성을 포함하는 딕셔너리 또는 찾을 수 없는 경우 None
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
        print(f"{model_name}에 대한 구성을 가져오는 중 오류 발생: {str(e)}")
        return None

def get_params_from_hf_config(config: Dict) -> Optional[float]:
    """
    가능한 경우 Hugging Face 구성에서 매개변수 수를 추출합니다.
    
    Args:
        config: Hugging Face에서 가져온 모델 구성을 포함하는 딕셔너리
        
    Returns:
        매개변수 수를 십억 단위로 나타내는 부동 소수점 또는 직접 찾을 수 없는 경우 None
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
            print(f"필드 '{field}'에서 매개변수 수 발견: {config[field]}")
            return config[field] / 1e9
    
    return None

def parse_hf_config_to_architecture(config: Dict) -> Dict[str, int]:
    """
    Hugging Face config.json을 아키텍처 형식으로 파싱합니다.
    다양한 모델 아키텍처와 특정 명명 규칙을 처리합니다.
    파싱을 실패하면 예측을 시도합니다.
    
    Args:
        config: Hugging Face에서 가져온 모델 구성을 포함하는 딕셔너리
        
    Returns:
        표준화된 아키텍처 매개변수를 포함하는 딕셔너리
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
    Hugging Face에서 찾을 수 없는 경우 모델 이름에서 매개변수 수를 추정합니다.
    수십억(B, b)과 수백만(M, m) 매개변수 사양을 모두 처리합니다.
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
    스케일링 법칙을 사용하여 매개변수 수에 따라 모델 아키텍처 세부 정보를 추정합니다.
    이는 실제로 모델이 설계되는 방식과 일치하여 특히 작은 모델에 대해 더 현실적인
    아키텍처 매개변수를 생성합니다.
    
    Args:
        params_billions: 수십억 단위의 모델 크기
    Returns:
        추정된 아키텍처 세부 정보가 포함된 딕셔너리
    """
    # 잘 알려진 모델 크기(예: 7B, 8B, 13B, 70B 모델)에 대한 특별 사례 처리
    # 이는 업계의 일반적인 아키텍처를 기반으로 하드코딩되어 있습니다
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
        
    # 다른 모델 크기의 경우, 크기 범위에 따라 매개변수 추정
    # 매개변수 수에 따라 레이어 수, 헤드 및 어휘 크기 할당
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
        
    # 매개변수 수에서 유도된 이차 방정식을 사용하여 hidden_size 계산
    # A*hidden_size^2 + B*hidden_size + C = 0 여기서:
    # A = num_layers * 16 (어텐션 및 MLP 블록의 가중치를 계산)
    # B = 2 * vocab (임베딩 및 언임베딩 행렬을 계산)
    # C = -params_billions * 1e9 (원시 개수로 변환된 원하는 매개변수 수)
    A = num_layers * 4 + num_layers * 12  # num_layers * 16에서 단순화
    B = 2 * vocab
    C = -1 * params_billions * 1e9
    discriminant = B**2 - 4*A*C
    
    # hidden_size에 대한 이차 방정식 풀기
    if discriminant < 0:
        # 판별식이 음수인 경우 대체 계산(유효한 입력에서는 발생하지 않아야 함)
        hidden_size = int(math.sqrt(params_billions * 1e9 / (num_layers * 16)))
    else:
        hidden_size = int((-B + math.sqrt(discriminant)) / (2*A))
    
    # 하드웨어 효율성을 위해 hidden_size를 128의 가장 가까운 배수로 올림
    hidden_size = math.ceil(hidden_size / 128) * 128
    
    # 헤드 차원 결정 및 헤드 수 조정
    if hidden_size >= 2048:
        head_dim = 128  # 더 큰 모델은 128차원 헤드 사용
    else:
        head_dim = 64   # 작은 모델은 64차원 헤드 사용
    num_heads = hidden_size // head_dim
    
    # hidden_size를 기반으로 intermediate_size(MLP 너비) 계산
    # 더 큰 모델은 더 넓은 MLP를 위해 더 높은 승수 사용
    if params_billions >= 20:
        multiplier = 3.0  # 매우 큰 모델에 대한 3배 승수
    else:
        multiplier = 2.75  # 작은 모델에 대한 2.75배 승수
        
    # intermediate_size 계산 및 256의 가장 가까운 배수로 반올림
    intermediate_size = int(hidden_size * multiplier)
    intermediate_size = round(intermediate_size / 256) * 256
    
    # 완전한 아키텍처 사양 반환
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
    모델의 매개변수 수를 수십억 단위로 가져오며, 우선순위는 다음과 같습니다:
    1. 사용자 제공 값(최우선)
    2. Hugging Face 구성(대체)
    3. 모델 이름에서의 추정(최후 수단)
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
    모델 아키텍처 세부 정보를 가져오며, 우선순위는 다음과 같습니다:
    1. 사용자 제공 아키텍처 세부 정보(최우선)
    2. Hugging Face 구성(대체)
    3. 매개변수 수에 따른 추정(최후 수단)
    
    참고: 사용자 값은 항상 HuggingFace 또는 추정에서 가져온 값을 재정의합니다
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
    """모델 매개변수에 필요한 메모리를 GB 단위로 계산합니다."""
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
    특히 vLLM 추론을 위한 활성화 메모리를 계산합니다.
    이는 병렬 처리 고려 없이 단일 GPU에 대해 계산합니다.
    
    Args:
        arch: 모델 아키텍처 매개변수를 포함하는 딕셔너리
            - hidden_size: 모델의 숨겨진 차원 크기
            - num_heads: 어텐션 헤드 수
            - num_layers: 트랜스포머 레이어 수
            - head_dim (선택적): 각 어텐션 헤드의 차원
            - intermediate_size (선택적): 피드포워드 중간 레이어 크기
        max_batch_size: 동시에 처리되는 최대 시퀀스 수
        max_seq_len: 최대 시퀀스 길이
        dtype: 계산에 사용되는 데이터 유형(예: "float16", "float32")
        
    Returns:
        기가바이트 단위의 추정 활성화 메모리
    """
    # 아키텍처 매개변수 추출
    hidden_size = arch["hidden_size"]
    num_heads = arch["num_heads"]
    num_layers = arch["num_layers"]
    head_dim = arch.get("head_dim", hidden_size // num_heads)
    intermediate_size = arch.get("intermediate_size", 4 * hidden_size)
    
    # 바이트 단위의 데이터 유형 크기 정의
    DTYPE_SIZES = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "int8": 1
    }
    bytes_per_value = DTYPE_SIZES[dtype]
    
    # 추론을 위해 KV 캐시는 주요 활성화 메모리 구성 요소입니다
    # 각 레이어는 시퀀스의 각 토큰에 대해 K 및 V 텐서를 저장합니다
    kv_cache_bytes = (
        2 *  # K 및 V
        max_batch_size *
        max_seq_len *
        num_layers *
        num_heads *
        head_dim *
        bytes_per_value
    )
    
    # 디코딩 중에도 현재 토큰에 대한 일부 활성화 메모리가 필요합니다
    # 이에는 어텐션 계산 및 FFN 활성화가 포함됩니다
    # 플래시 어텐션의 경우, 이는 긴 시퀀스에 대한 KV 캐시보다 훨씬 작습니다
    
    # 현재 토큰에 대한 어텐션 활성화(모든 헤드에 대한 Q * K)
    attn_act_bytes = (
        max_batch_size *
        1 *  # 새 토큰에 대해서만 계산
        num_heads *
        head_dim *
        bytes_per_value
    )
    
    # FFN 활성화
    ffn_act_bytes = (
        max_batch_size *
        1 *  # 새 토큰에 대해서만 계산
        intermediate_size *
        bytes_per_value
    )
    
    # 레이어 수로 곱하기
    per_layer_act_bytes = attn_act_bytes + ffn_act_bytes
    decode_act_bytes = per_layer_act_bytes * num_layers
    
    # 기타 활성화에 대한 작은 버퍼 추가(10%)
    misc_buffer = 0.1 * (kv_cache_bytes + decode_act_bytes)
    
    # 총 활성화 메모리
    total_act_bytes = kv_cache_bytes + decode_act_bytes + misc_buffer
    
    # 기가바이트로 변환
    total_act_gb = total_act_bytes / (1024**3) + 0.7 # 실측에서 나온 경험적 추가 오버헤드
    
    return total_act_gb

def calculate_kv_cache_memory(arch: Dict[str, int], 
                             max_seq_len: int, 
                             max_batch_size: int, 
                             kv_cache_dtype: str,
                             max_model_len: Optional[int] = None) -> Dict[str, float]:
    """
    PagedAttention을 사용한 KV 캐시 메모리 요구 사항을 계산합니다.
    
    참고: vLLM은 이론적 계산보다 KV 캐시에 상당히 더 많은 메모리를 할당합니다
    - 이는 성능 최적화를 위한 설계입니다.
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
    텐서 병렬 처리를 사용할 때 GPU당 메모리 요구 사항을 계산합니다.
    
    실제 vLLM 로그를 기반으로 실제 분포를 더 잘 반영하도록 개선:
    - 모델 매개변수가 완벽하게 분산되지 않음(오버헤드 추가)
    - 활성화 메모리는 기본 및 작업자 GPU 간에 다름
    - KV 캐시는 실제 모델 요구 사항을 기반으로 함
    - CUDA 오버헤드는 모델 크기에 따라 다름
    - 그래핑 작업으로 모델을 처음 시작할 때 로그상 추가 오버헤드 발생, 오차율을 추가하여 극복
    
    Args:
        tensor_parallel_size: 텐서 병렬 처리를 위한 GPU 수
        model_params_memory_gb: GB 단위의 총 모델 매개변수 메모리
        activation_memory_gb: GB 단위의 총 활성화 메모리
        min_kv_cache_memory_gb: GB 단위의 최소 KV 캐시 메모리
        max_kv_cache_memory_gb: GB 단위의 최대 KV 캐시 메모리
        cuda_overhead_gb: GB 단위의 CUDA 오버헤드
        params_billions: 수십억 단위의 모델 크기
        
    Returns:
        GPU당 메모리 요구 사항이 포함된 딕셔너리
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
    staging_overhead = 0.05
    metadata_overhead = 0.05
    total_additional_overhead = communication_overhead + sync_buffer_overhead + staging_overhead + metadata_overhead

    primary_gpu_activation = activation_memory_gb / (5 + tensor_parallel_size - 1) * 5 * (1 + total_additional_overhead) * 1.1
    worker_gpu_activation = activation_memory_gb / (5 + tensor_parallel_size - 1) * (1 + total_additional_overhead) * 1.1 # 1.1은 오차를 위한 계수입니다
   
    per_gpu_activation = primary_gpu_activation # 가장 많은 메모리를 요구하는 주 GPU로 값 계산
    
    per_gpu_min_kv_cache = min_kv_cache_memory_gb / tensor_parallel_size
    per_gpu_max_kv_cache = max_kv_cache_memory_gb / tensor_parallel_size
    
    per_gpu_cuda_overhead = cuda_overhead_gb / tensor_parallel_size * 1.5 # 실측상 오버헤드가 더 크다고 판단되어 계수 증가
    
    return {
        "per_gpu_model_params": per_gpu_model_params,
        "per_gpu_activation": per_gpu_activation,
        "per_gpu_min_kv_cache": per_gpu_min_kv_cache,
        "per_gpu_max_kv_cache": per_gpu_max_kv_cache,
        "per_gpu_cuda_overhead": per_gpu_cuda_overhead,
    }

def get_context_length_from_config(config: Dict) -> Optional[int]:
    """
    모델 구성에서 컨텍스트 창 크기를 추출합니다.
    
    Args:
        config: Hugging Face에서 가져온 모델 구성을 포함하는 딕셔너리
        
    Returns:
        최대 시퀀스 길이를 나타내는 정수 또는 찾을 수 없는 경우 None
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
    vLLM으로 특정 모델 서빙을 위한 GPU 메모리 사용량을 추정합니다.
    중요!: 추천 GPU 메모리양은 모델에 최대 시퀸스 길이 * 0.2를 사용하여 계산되었습니다.
    이보다 긴 길이의 시퀸스는 보다 많은 메모리를 요구합니다.
    
    다음 사항을 고려합니다:
    1. 모델 매개변수 메모리
    2. 추론을 위한 활성화 메모리
    3. KV 캐시 메모리(PagedAttention 사용)
    4. CUDA 컨텍스트 및 기타 오버헤드
    """
    estimated_values = []
    missing_values = []
    
    if request.tensor_parallel_size <= 0:
        raise HTTPException(status_code=400, detail="tensor_parallel_size는 양수여야 합니다")
    
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
            detail="사용 가능한 시퀀스 길이가 없습니다. 이 모델의 구성에는 컨텍스트 길이 정보가 포함되어 있지 않으므로 max_seq_len을 제공하세요."
        )
    
    if effective_max_seq_len <= 0:
        raise HTTPException(status_code=400, detail="max_seq_len은 양수여야 합니다")
    
    if request.max_model_len is not None:
        max_model_len = request.max_model_len
    else:
        max_model_len = effective_max_seq_len
        
    if max_model_len is None:
        max_model_len = effective_max_seq_len
    
    if request.max_batch_size <= 0:
        raise HTTPException(status_code=400, detail="max_batch_size는 양수여야 합니다")
    if request.params_millions is not None and request.params_billions is None:
        request.params_billions = request.params_millions / 1000.0
    
    if request.kv_cache_dtype is None:
        request.kv_cache_dtype = request.dtype
    
    if request.dtype not in DTYPE_SIZES:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 dtype: {request.dtype}. 지원되는 유형: {list(DTYPE_SIZES.keys())}")
    
    if request.kv_cache_dtype not in DTYPE_SIZES:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 KV 캐시 dtype: {request.kv_cache_dtype}. 지원되는 유형: {list(DTYPE_SIZES.keys())}")
    
    if request.quantization is not None and request.quantization not in QUANTIZATION_FACTORS:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 양자화: {request.quantization}. 지원되는 값: Q4, Q8, None")
    
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
        context_length_source = "사용자"
    
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
            warning = "중요한 모델 세부 정보(매개변수 수 및 아키텍처)가 추정되어 부정확할 수 있습니다. 더 정확한 계산을 위해 이러한 값을 수동으로 제공하는 것을 고려하세요."
        elif "parameter_count" in estimated_values:
            warning = "모델 매개변수 수가 모델 이름에서 추정되어 부정확할 수 있습니다. 더 정확한 계산을 위해 params_billions 또는 params_millions 제공을 고려하세요."
        elif "model_architecture" in estimated_values:
            warning = "모델 아키텍처 세부 정보가 추정되어 부정확할 수 있습니다. 더 정확한 계산을 위해 아키텍처 세부 정보 제공을 고려하세요."
    
    if missing_values:
        if warning:
            warning += f" 결정할 수 없는 누락된 값: {', '.join(missing_values)}."
        else:
            warning = f"결정할 수 없는 누락된 값: {', '.join(missing_values)}."
    
    if recommended_memory_gb > 80 and not warning:
        warning = "이 모델은 여러 GPU 또는 충분한 메모리가 있는 고급 GPU가 필요할 수 있습니다."
    elif recommended_memory_gb > 80:
        warning += " 이 모델은 여러 GPU 또는 충분한 메모리가 있는 고급 GPU가 필요할 수 있습니다."
    
    if request.tensor_parallel_size > 1:
        multi_gpu_warning = f"텐서 병렬 처리를 위해 {request.tensor_parallel_size}개의 GPU에 걸쳐 메모리 요구 사항이 계산되었습니다."
        if warning:
            warning += " " + multi_gpu_warning
        else:
            warning = multi_gpu_warning
    
    recommended_gpu_memory_utilization = None
    if request.current_gpu_memory_size_gb:
        if request.current_gpu_memory_size_gb <= 0:
            raise HTTPException(status_code=400, detail="current_gpu_memory_size_gb는 양수여야 합니다")
        
        if request.tensor_parallel_size > 1:
            util_ratio = per_gpu_total_min_memory_gb / request.current_gpu_memory_size_gb
        else:
            util_ratio = total_min_memory_gb / request.current_gpu_memory_size_gb
        
        if util_ratio > 0.97:
            if warning:
                warning += " 모델은 GPU에서 사용 가능한 것보다 더 많은 메모리가 필요합니다. 더 큰 GPU 사용, 배치 크기 줄이기 또는 양자화 적용을 고려하세요."
            else:
                warning = "모델은 GPU에서 사용 가능한 것보다 더 많은 메모리가 필요합니다. 더 큰 GPU 사용, 배치 크기 줄이기 또는 양자화 적용을 고려하세요."
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
    쿼리를 기반으로 Hugging Face에서 모델을 검색합니다.
    검색 기준과 일치하는 모델 이름을 반환합니다.
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
                "error": f"API 요청이 상태 코드와 함께 실패했습니다: {response.status_code}",
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
        "info": "vLLM GPU 메모리 계산기 API",
        "usage": "모델 세부 정보와 함께 /estimate-gpu-memory에 POST 요청",
        "example_request": {
            "model_name": "meta-llama/Llama-3-8b",
            "dtype": "float16",
            "tensor_parallel_size": 2
        },
        "features": [
            "Hugging Face Hub에서 모델 구성을 자동으로 가져옵니다",
            "가능한 경우 모델 구성에서 컨텍스트 길이를 자동으로 감지합니다",
            "사용자 제공 아키텍처 세부 정보 및 재정의를 지원합니다",
            "제공되지 않은 경우 모델 매개변수를 지능적으로 추정합니다",
            "vLLM 배포를 위한 메모리 요구 사항을 계산합니다",
            "추정되거나 누락된 값에 대한 자세한 피드백을 제공합니다",
            "여러 GPU에 걸친 텐서 병렬 처리를 지원합니다",
            "추천 GPU 메모리양은 모델에 최대 시퀸스 길이 * 0.2를 사용하여 계산되었습니다. 이보다 긴 길이의 시퀸스는 보다 많은 메모리를 요구합니다."
        ],
        "endpoints": {
            "/estimate-gpu-memory": "POST - GPU 메모리 요구 사항 계산",
            "/search-models": "GET - Hugging Face에서 모델 검색"
        },
        "notes": [
            "시퀀스 길이(max_seq_len) 매개변수는 선택 사항입니다 - 제공되지 않으면 API는 모델의 구성에서 가져오려고 시도합니다",
            "모델의 구성에 컨텍스트 길이 정보가 없는 경우 max_seq_len을 제공해야 합니다",
            "다중 GPU 설정의 경우 GPU당 메모리 요구 사항을 확인하기 위해 'tensor_parallel_size' 매개변수를 지정하세요",
            "요청에 유저의 값이 있으면 해당 값을 최우선적으로 적용합니다"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)