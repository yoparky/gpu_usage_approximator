from fastapi import FastAPI, HTTPException
import math
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI(title="vLLM GPU Memory Calculator API")
# 일반 방정식:
# 총_메모리 = 모델_메모리 + 활성화_메모리 + kv_캐시_메모리 + cuda_오버헤드
# https://github.com/RahulSChand/gpu_poor 참조

class ModelArchitecture(BaseModel):
    """사용자가 알고 있는 경우 선택적 모델 아키텍처 세부 정보"""
    num_layers: Optional[int] = None  # 레이어 수
    hidden_size: Optional[int] = None  # 히든 크기 
    num_heads: Optional[int] = None  # 어텐션 헤드 수
    head_dim: Optional[int] = None  # 헤드 차원
    intermediate_size: Optional[int] = None  # FFN 레이어용 중간 크기


class ModelRequest(BaseModel):
    model_name: str  # 모델 이름
    max_seq_len: int  # 최대 시퀀스 길이
    dtype: str = "float16"  # 기본값은 float16
    kv_cache_dtype: Optional[str] = None  # None인 경우 dtype과 동일하게 사용
    max_batch_size: int = 32  # 기본 배치 크기
    max_model_len: Optional[int] = None  # None인 경우 max_seq_len을 사용
    gpu_memory_utilization: float = 0.9  # 기본 GPU 메모리 활용 비율
    quantization: Optional[str] = None  # Q4, Q8 또는 None (전체 정밀도)
    params_billions: Optional[float] = None  # 선택적 매개변수 수 오버라이드
    architecture: Optional[ModelArchitecture] = None  # 선택적 모델 아키텍처 세부 정보


class MemoryEstimation(BaseModel):
    model_name: str  # 모델 이름
    model_params_memory_gb: float  # 모델 파라미터 메모리 (GB)
    activation_memory_gb: float  # 활성화 메모리 (GB)
    min_kv_cache_memory_gb: float  # 최소 KV 캐시 메모리 (GB)
    max_kv_cache_memory_gb: float  # 최대 KV 캐시 메모리 (GB)
    cuda_overhead_gb: float  # CUDA 오버헤드 (GB)
    total_min_memory_gb: float  # 최소 총 메모리 (GB)
    total_max_memory_gb: float  # 최대 총 메모리 (GB)
    recommended_memory_gb: float  # 권장 메모리 (GB)
    warning: Optional[str] = None  # 경고 메시지
    components_breakdown: Dict[str, float]  # 구성 요소 내역
    architecture_used: Dict[str, int]  # 계산에 사용된 아키텍처 값


# 일반적인 모델의 매개변수 수 (십억 단위)
MODEL_PARAMS = {
    # LLaMA 계열 (기존 항목 및 누락된 항목)
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

    # Llama 3.2 계열 추가 (신규)
    "meta-llama/Llama-3.2-1B": 1.23,       # 모델 카드에서 가져온 실제 매개변수 수
    "meta-llama/Llama-3.2-3B": 3.21,       # 모델 카드에서 가져온 실제 매개변수 수
    "meta-llama/Llama-3.2-1B-instruct": 1.23,
    "meta-llama/Llama-3.2-3B-instruct": 3.21,
    
    # 추가 LLaMA 모델
    "meta-llama/Llama-3.1-8b": 8,
    "meta-llama/Llama-3.1-70b": 70,
    "meta-llama/Llama-3.1-405b": 405,
    "meta-llama/Llama-3.1-8b-instruct": 8,
    "meta-llama/Llama-3.1-70b-instruct": 70,
    "meta-llama/Llama-3.1-405b-instruct": 405,
    
    # Mistral 계열 (기존 항목)
    "mistralai/Mistral-7B-v0.1": 7,
    "mistralai/Mixtral-8x7B-v0.1": 47,  # 각각 7B인 8개의 전문가, 하지만 모두 동시에 로드되지는 않음
    "mistralai/Mistral-7B-Instruct-v0.1": 7,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 47,
    
    # 추가 Mistral 모델
    "mistralai/Mistral-Large-2-128k": 32,
    "mistralai/Mistral-Large-Instruct-2-128k": 32,
    "mistralai/Mistral-Medium-2-128k": 7,
    "mistralai/Mistral-Medium-Instruct-2-128k": 7,
    "mistralai/Mixtral-8x22B-v0.1": 176,  # 각각 22B인 8개의 전문가
    
    # OpenAI 모델
    "openai/gpt-3.5-turbo": 20,  # 추정치
    "openai/gpt-3.5-turbo-16k": 20,  # 추정치
    "openai/gpt-4": 1500,  # 추정치
    "openai/gpt-4-turbo": 1700,  # 추정치
    "openai/gpt-4o": 1800,  # 추정치
    "openai/text-davinci-003": 175,  # 추정치
    
    # Qwen 계열
    "Qwen/Qwen-1.5-0.5B": 0.5,
    "Qwen/Qwen-1.5-1.8B": 1.8,
    "Qwen/Qwen-1.5-4B": 4,
    "Qwen/Qwen-1.5-7B": 7,
    "Qwen/Qwen-1.5-14B": 14,
    "Qwen/Qwen-1.5-32B": 32,
    "Qwen/Qwen-1.5-72B": 72,
    "Qwen/Qwen-1.5-110B": 110,
    
    # Gemma 계열
    "google/gemma-2b": 2,
    "google/gemma-7b": 7,
    "google/gemma-2b-instruct": 2,
    "google/gemma-7b-instruct": 7,
    
    # Claude 계열 (Anthropic)
    "anthropic/claude-3-opus": 180,  # 추정치
    "anthropic/claude-3-sonnet": 100,  # 추정치
    "anthropic/claude-3-haiku": 40,  # 추정치
    
    # Gemini 계열
    "google/gemini-nano": 3.25,  # 추정치
    "google/gemini-pro": 50,  # 추정치
    "google/gemini-ultra": 500,  # 추정치
    
    # Yi 계열
    "01-ai/Yi-6B": 6,
    "01-ai/Yi-9B": 9,
    "01-ai/Yi-34B": 34,
    
    # Falcon 계열
    "tiiuae/falcon-7b": 7,
    "tiiuae/falcon-40b": 40,
    "tiiuae/falcon-180b": 180,
    
    # 기타 기존 모델
    "stabilityai/stablelm-tuned-alpha-7b": 7,
    "mosaicml/mpt-7b": 7,
    "mosaicml/mpt-30b": 30,
    "EleutherAI/pythia-12b": 12,
    "Together/falcon-40b-instruct": 40,
    "openchat/openchat-3.5": 7,
    
    # 기본 폴백
    "unknown": 0  # 이름을 기반으로 추정되거나 사용자가 설정
}

# 알려진 모델의 아키텍처 세부 정보
MODEL_ARCHITECTURES = {
    # LLaMA-2 계열
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
    # Llama 3.2 계열 추가 (신규)
    "meta-llama/Llama-3.2-1B": {
        "num_layers": 24,           # 일반적인 아키텍처 스케일링 기반
        "hidden_size": 2048,        # 일반적인 아키텍처 스케일링 기반
        "num_heads": 16,            # 이 크기에 대한 일반적인 헤드 구성
        "head_dim": 128,            # Llama 계열에서 유지됨
        "intermediate_size": 5632   # hidden_size의 약 2.75배 비율 기반
    },
    "meta-llama/Llama-3.2-3B": {
        "num_layers": 26,           # 일반적인 아키텍처 스케일링 기반
        "hidden_size": 3072,        # 일반적인 아키텍처 스케일링 기반
        "num_heads": 24,            # 이 크기에 대한 일반적인 헤드 구성
        "head_dim": 128,            # Llama 계열에서 유지됨
        "intermediate_size": 8448   # hidden_size의 약 2.75배 비율 기반
    },
    
    # Mistral-7B
    "mistralai/Mistral-7B-v0.1": {
        "num_layers": 32,
        "hidden_size": 4096,
        "num_heads": 32,
        "head_dim": 128,
        "intermediate_size": 14336
    }
    
    # 필요한 경우 더 많은 알려진 모델 아키텍처 추가
}


# 다양한 dtype의 메모리 크기 (바이트 단위)
DTYPE_SIZES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,  # 4비트 = 0.5 바이트
}


# 양자화 요소 (모델이 얼마나 작아지는지)
QUANTIZATION_FACTORS = {
    "Q8": 2,  # Q8 양자화의 경우 모델 크기가 2로 나눠짐
    "Q4": 4,  # Q4 양자화의 경우 모델 크기가 4로 나눠짐
    None: 1   # 양자화 없음, 원래 크기 유지
}


def estimate_model_architecture(params_billions: float) -> Dict[str, int]:
    """
    Estimate model architecture details based on parameter count using a more accurate
    approach that combines range-based estimation with quadratic equation solving.
    
    This matches how models are actually designed in practice, producing more realistic
    architecture parameters, especially for smaller models.
    
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
    """데이터베이스에 없는 경우 모델 이름에서 매개변수 수를 추정합니다."""
    # 모델 이름에서 숫자 찾기 (예: 7b, 13b 등)
    import re
    numbers = re.findall(r'(\d+)b', model_name.lower())
    if numbers:
        # 'b'가 뒤에 오는 가장 큰 숫자를 매개변수 수(십억 단위)로 취급
        return max([float(num) for num in numbers])
    return 7  # 결정할 수 없는 경우 기본값 7B 사용


def get_model_params_count(model_name: str, params_billions: Optional[float] = None) -> float:
    """모델의 매개변수 수를 십억 단위로 가져옵니다."""
    # 사용자가 매개변수 수를 제공한 경우 사용
    if params_billions is not None:
        return params_billions
    
    # 그렇지 않으면 데이터베이스 확인
    if model_name in MODEL_PARAMS:
        return MODEL_PARAMS[model_name]
    
    # 짧은 이름으로 모델 찾기 시도
    short_name = model_name.split('/')[-1].lower()
    for known_model, params in MODEL_PARAMS.items():
        if short_name in known_model.lower():
            return params
    
    # 여전히 찾을 수 없는 경우 이름에서 추정 시도
    return estimate_params_from_name(model_name)


def get_model_architecture(model_name: str, params_billions: float, 
                          user_arch: Optional[ModelArchitecture] = None) -> Dict[str, int]:
    """
    모델 아키텍처 세부 정보를 가져옵니다. 우선순위:
    1. 사용자 제공 값
    2. 데이터베이스의 알려진 아키텍처
    3. 매개변수 수를 기반으로 한 추정 값
    """
    # 추정된 아키텍처로 시작
    arch = estimate_model_architecture(params_billions)
    
    # 가능한 경우 알려진 아키텍처로 업데이트
    if model_name in MODEL_ARCHITECTURES:
        for key, value in MODEL_ARCHITECTURES[model_name].items():
            arch[key] = value
    
    # 사용자 제공 아키텍처 세부 정보로 업데이트
    if user_arch:
        user_arch_dict = user_arch.dict(exclude_unset=True, exclude_none=True)
        for key, value in user_arch_dict.items():
            if value is not None:
                arch[key] = value
    
    return arch


def calculate_model_memory(model_name: str, dtype: str, 
                          quantization: Optional[str] = None,
                          params_billions: Optional[float] = None) -> Dict[str, float]:
    """모델 매개변수에 필요한 메모리를 GB 단위로 계산합니다."""
    # 매개변수 수 가져오기 (사용자가 제공한 값 사용)
    params_billions = get_model_params_count(model_name, params_billions)
    bytes_per_param = DTYPE_SIZES[dtype]
    
    # 양자화 요소 적용 (지정된 경우)
    quant_factor = QUANTIZATION_FACTORS.get(quantization, 1)
    
    # 모델 크기를 GB 단위로 계산 (약간의 오버헤드 포함)
    model_size_bytes = params_billions * 1e9 * bytes_per_param
    model_size_gb = model_size_bytes / (1024**3) / quant_factor * 1.05  # 모델 텐서를 위한 5% 오버헤드
    
    return {
        "params_billions": params_billions,
        "model_size_gb": model_size_gb
    }


def calculate_activation_memory(arch: Dict[str, int],
                               max_batch_size: int, 
                               max_seq_len: int, 
                               dtype: str) -> float:
    """
    추론을 위한 활성화 메모리를 계산합니다.
    hidden_size와 intermediate_size 활성화를 모두 고려합니다.
    """
    num_layers = arch["num_layers"]
    hidden_size = arch["hidden_size"]
    intermediate_size = arch["intermediate_size"]
    
    # dtype 기반 바이트 단위 값
    bytes_per_value = DTYPE_SIZES[dtype]
    
    # 레이어당 5개의 활성화 가정
    # hidden_size 활성화(쿼리, 키, 값, 어텐션 출력 등)의 경우
    # 일반적으로 레이어당 3개의 활성화가 hidden_size 사용
    hidden_activation_bytes = 3 * max_batch_size * max_seq_len * hidden_size * bytes_per_value
    
    # intermediate_size 활성화(FFN 중간 출력)의 경우
    # 일반적으로 레이어당 2개의 활성화가 intermediate_size 사용
    intermediate_activation_bytes = 2 * max_batch_size * max_seq_len * intermediate_size * bytes_per_value
    
    # 모든 레이어에 걸친 총 활성화 메모리
    total_activation_bytes = num_layers * (hidden_activation_bytes + intermediate_activation_bytes)
    
    # GB 단위로 변환
    activation_memory_gb = total_activation_bytes / (1024**3)
    
    return activation_memory_gb


def calculate_kv_cache_memory(arch: Dict[str, int], 
                             max_seq_len: int, 
                             max_batch_size: int, 
                             kv_cache_dtype: str,
                             max_model_len: Optional[int] = None) -> Dict[str, float]:
    """
    PagedAttention을 사용한 KV 캐시 메모리 요구 사항을 계산합니다.
    """
    if max_model_len is None:
        max_model_len = max_seq_len
    
    num_layers = arch["num_layers"]
    hidden_size = arch["hidden_size"]
    
    # KV 캐시 크기 계산
    bytes_per_token = DTYPE_SIZES[kv_cache_dtype]
    
    # PagedAttention에서 메모리는 블록 단위로 할당됨
    # 각 블록은 일반적으로 효율적인 메모리 관리를 위해 16개의 토큰을 보유
    block_size = 16
    num_blocks_per_seq = math.ceil(max_model_len / block_size)
    
    # 레이어당 토큰당 키 + 값 캐시
    # vLLM은 Q(쿼리) 벡터가 아닌 K와 V만 저장하면 됨
    kv_cache_per_token = 2 * hidden_size * bytes_per_token  # K와 V에 대해 2
    
    # 최소 할당을 위한 총 KV 캐시 크기 (시퀀스당 하나의 블록)
    min_kv_cache_size_bytes = num_layers * max_batch_size * kv_cache_per_token * block_size
    
    # 최대 할당을 위한 총 KV 캐시 크기 (시퀀스당 모든 블록)
    max_kv_cache_size_bytes = num_layers * max_batch_size * kv_cache_per_token * max_model_len
    
    # GB 단위로 변환
    min_kv_cache_size_gb = min_kv_cache_size_bytes / (1024**3)
    max_kv_cache_size_gb = max_kv_cache_size_bytes / (1024**3)
    
    return {
        "min_kv_cache_memory_gb": min_kv_cache_size_gb,
        "max_kv_cache_memory_gb": max_kv_cache_size_gb
    }


@app.post("/estimate-gpu-memory", response_model=MemoryEstimation)
def estimate_gpu_memory(request: ModelRequest) -> MemoryEstimation:
    """
    vLLM으로 특정 모델을 서빙하기 위한 GPU 메모리 사용량을 추정합니다.
    
    이는 다음을 고려합니다:
    1. 모델 매개변수 메모리
    2. 추론을 위한 활성화 메모리
    3. KV 캐시 메모리 (PagedAttention 사용)
    4. CUDA 컨텍스트 및 기타 오버헤드
    """
    # 함수 시작 부분에 추가
    if request.max_seq_len <= 0:
        raise HTTPException(status_code=400, detail="max_seq_len은 양수여야 합니다")
    if request.max_batch_size <= 0:
        raise HTTPException(status_code=400, detail="max_batch_size는 양수여야 합니다")
    if not (0 < request.gpu_memory_utilization <= 1):
        raise HTTPException(status_code=400, detail="gpu_memory_utilization은 0과 1 사이여야 합니다")
    
    # 기본값 검증 및 설정
    if request.kv_cache_dtype is None:
        request.kv_cache_dtype = request.dtype
    
    if request.max_model_len is None:
        request.max_model_len = request.max_seq_len
    
    # dtype 값이 유효한지 확인
    if request.dtype not in DTYPE_SIZES:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 dtype: {request.dtype}. 지원되는 유형: {list(DTYPE_SIZES.keys())}")
    
    if request.kv_cache_dtype not in DTYPE_SIZES:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 KV 캐시 dtype: {request.kv_cache_dtype}. 지원되는 유형: {list(DTYPE_SIZES.keys())}")
    
    if request.quantization is not None and request.quantization not in QUANTIZATION_FACTORS:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 양자화: {request.quantization}. 지원되는 값: Q4, Q8, None")
    
    # 모델 매개변수 메모리 계산
    model_info = calculate_model_memory(
        request.model_name, 
        request.dtype, 
        request.quantization,
        request.params_billions
    )
    
    model_params_memory_gb = model_info["model_size_gb"]
    params_billions = model_info["params_billions"]
    
    # 모델 아키텍처 가져오기 (사용자 제공 값 우선)
    architecture = get_model_architecture(
        request.model_name,
        params_billions,
        request.architecture
    )
    
    # 활성화 메모리 계산
    activation_memory_gb = calculate_activation_memory(
        architecture,
        request.max_batch_size,
        request.max_seq_len,
        request.dtype
    )
    
    # KV 캐시 메모리 계산 (최소 및 최대)
    kv_cache_memory = calculate_kv_cache_memory(
        architecture,
        request.max_seq_len,
        request.max_batch_size,
        request.kv_cache_dtype,
        request.max_model_len
    )
    
    # CUDA 컨텍스트 및 기타 오버헤드
    cuda_overhead_gb = 0.75  # 권장 750 MB
    
    # 양자화를 사용하는 경우 추가 오버헤드
    if request.quantization:
        cuda_overhead_gb += 0.25  # 양자화 라이브러리를 위해 250 MB 추가
    
    # 총 메모리 요구 사항 계산
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
    
    # 지정된 활용 비율로 권장 메모리 계산
    recommended_memory_gb = total_max_memory_gb / request.gpu_memory_utilization
    
    # 더 나은 가독성을 위해 값을 반올림하면서 2자리 소수점 정밀도 유지
    model_params_memory_gb = round(model_params_memory_gb, 2)
    activation_memory_gb = round(activation_memory_gb, 2)
    min_kv_cache_memory_gb = round(kv_cache_memory["min_kv_cache_memory_gb"], 2)
    max_kv_cache_memory_gb = round(kv_cache_memory["max_kv_cache_memory_gb"], 2)
    total_min_memory_gb = round(total_min_memory_gb, 2)
    total_max_memory_gb = round(total_max_memory_gb, 2)
    recommended_memory_gb = round(recommended_memory_gb, 2)
    
    # 메모리 구성 요소의 상세 내역
    components_breakdown = {
        "model_params": model_params_memory_gb,
        "activation": activation_memory_gb,
        "min_kv_cache": min_kv_cache_memory_gb,
        "max_kv_cache": max_kv_cache_memory_gb,
        "cuda_overhead": cuda_overhead_gb
    }
    
    # 모델이 비정상적으로 큰 경우 경고 생성
    warning = None
    if recommended_memory_gb > 80:
        warning = "이 모델은 여러 개의 GPU 또는 충분한 메모리가 있는 고급 GPU가 필요할 수 있습니다."
    
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


# 알려진 모델 아키텍처를 가져오는 추가 엔드포인트
@app.get("/known-architectures")
def get_known_architectures():
    return {
        "models": MODEL_ARCHITECTURES
    }


# 사용 예시 문서화 엔드포인트
@app.get("/")
def read_root():
    return {
        "info": "vLLM GPU 메모리 계산기 API",
        "usage": "모델 세부 정보와 함께 /estimate-gpu-memory로 POST 요청",
        "example_request": {
            "model_name": "meta-llama/Llama-2-7b",
            "max_seq_len": 2048,
            "dtype": "float16",
            "max_batch_size": 32,
            "gpu_memory_utilization": 0.9,
            "quantization": None,  # 선택 사항: 양자화 모델의 경우 "Q4" 또는 "Q8"
            "params_billions": None,  # 선택 사항: 알려진 경우 매개변수 수 오버라이드
            "architecture": {  # 선택 사항: 알려진 모델 아키텍처 세부 정보 제공
                "num_layers": 32,
                "hidden_size": 4096,
                "num_heads": 32,
                "head_dim": 128,
                "intermediate_size": 11008
            }
        },
        "memory_components": {
            "model_params": "모델 가중치를 위한 메모리",
            "activation": "추론 중 중간 활성화를 위한 메모리",
            "kv_cache": "키-값 캐시를 위한 메모리 (PagedAttention 사용 시 최소 및 최대)",
            "cuda_overhead": "CUDA 컨텍스트 및 추가 라이브러리를 위한 메모리"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)