from fastapi import FastAPI, HTTPException
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

# 요청, 답변, 답변용 아키텍처 클래스 임포트
from classes.MemoryEstimation import MemoryEstimation
from classes.ModelArchitecture import ModelArchitecture
from classes.ModelRequest import ModelRequest

# 유틸 임포트: 허깅페이스 컨피그 파일 다운, 추론 메모리 4가지 항목
from utils.GetFromHuggingFace import (
    fetch_model_config_from_hf, 
    get_params_from_hf_config, 
    get_model_architecture,
    get_context_length_from_config
)
from utils.CalculateActivation import calculate_activation_memory_inference
from utils.CalculateKVCache import calculate_kv_cache_memory
from utils.CalculateModelParameters import calculate_model_memory
from utils.GeneralUtil import load_model_config, estimate_params_from_name

app = FastAPI(title="vLLM GPU 메모리 계산기 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# API 모델 관련 설정값
PATH_TO_API_DEFINES = "./data/defines.yaml"

# 글로벌 디파인
DTYPE_SIZES, QUANTIZATION_FACTORS, PARAM_MAPPINGS = load_model_config()
GPU_CSV_FILE_PATH = "./data/raw_data/gpu_specs_v6.csv"
GPU_JSON_FILE_PATH = "./data/processed_data/test.json"

def convert_max_num_seqs_to_base(max_num_seqs: int):
    # vLLM의 Continuous Batching 기능은 기존 프레임워크들의 배치 기능과 다르게 동작합니다.
    # 이때문에 배치 하나 내에 있는 시퀸스 최대 수를 조절 하는데 이 때 활성화 메모리의 총량이
    # 시퀸스 수에 따라 증감합니다. 0.7.0 버전의 디폴트 max_num_seqs 256을 기준으로 하여
    # 활성화 메모리의 값의 근사값이 정해져 이를 기준으로 최대 시퀸스 수를 계산에 활용을 할 수
    # 있는 쉬운 값으로 변환합니다.
    # 이는 실측을 통해 얻은 메모리 소모량을 기준으로 했습니다.

    conversion_ratio = max_num_seqs
    # 128 이하의 값은 128과 거의 같은 양의 메모리를 소모하는 것으로 관측됬습니다
    if conversion_ratio <= 128:
        conversion_ratio = 0.5
    else: # 이외의 값들은 linear 관계로 증감합니다.
        conversion_ratio = conversion_ratio / 256

    return conversion_ratio

def calculate_per_gpu_memory(
    tensor_parallel_size: int,
    model_params_memory_gb: float,
    activation_memory_gb: float,
    min_kv_cache_memory_gb: float,
    max_kv_cache_memory_gb: float,
    cuda_overhead_gb: float,
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
    
    communication_overhead = 0.25 # 실측 어림 값, 로그 상 non-torch memory GiB
    total_additional_overhead = communication_overhead # 차 후 추가

    primary_gpu_activation = activation_memory_gb
    # 차후 사용하기 위한 코드
    worker_gpu_activation = activation_memory_gb / (5 + tensor_parallel_size - 1) * (1 + total_additional_overhead) * 1.1 # 1.1은 오차를 위한 계수입니다 TODO: max_num_seqs_ratio 고려해서 추정 필요
   
    per_gpu_activation = primary_gpu_activation # 가장 많은 메모리를 요구하는 주 GPU로 값 계산
    
    per_gpu_min_kv_cache = min_kv_cache_memory_gb / tensor_parallel_size
    per_gpu_max_kv_cache = max_kv_cache_memory_gb / tensor_parallel_size
    
    parent_gpu_cuda_overhead = cuda_overhead_gb / tensor_parallel_size + communication_overhead # 병렬화 시 주 gpu 커뮤니케이션 오버헤드 추가
    # 차후 사용하기 위한 코드
    worker_gpu_cuda_overhead = (parent_gpu_cuda_overhead - communication_overhead) / (5 + tensor_parallel_size - 1) * (1 + total_additional_overhead)
    return {
        "per_gpu_model_params": per_gpu_model_params,
        "per_gpu_activation": per_gpu_activation,
        "per_gpu_min_kv_cache": per_gpu_min_kv_cache,
        "per_gpu_max_kv_cache": per_gpu_max_kv_cache,
        "per_gpu_cuda_overhead": parent_gpu_cuda_overhead,
    }


@app.post("/estimate-gpu-memory", response_model=MemoryEstimation)
def estimate_gpu_memory(request: ModelRequest) -> MemoryEstimation:
    """
    vLLM으로 특정 모델 서빙을 위한 GPU 메모리 사용량을 추정합니다.
    게이티드 모델(예: Llama)의 경우 유효한 Hugging Face 토큰이 필요할 수 있습니다.
    
    다음 사항을 고려합니다:
    1. 모델 매개변수 메모리
    2. 추론을 위한 활성화 메모리
    3. KV 캐시 메모리(PagedAttention 사용)
    4. CUDA 컨텍스트 및 기타 오버헤드
    """
    estimated_values = []
    missing_values = []

    current_gpu_memory_use_gb = request.curr_gpu_use if request.curr_gpu_use else 0
    current_gpu_memory_size_gb = request.current_gpu_memory_size_gb if request.current_gpu_memory_size_gb else 0

    if request.tensor_parallel_size <= 0:
        raise HTTPException(status_code=400, detail="tensor_parallel_size는 양수여야 합니다")
    
    # Hugging Face 토큰 사용 여부 확인
    token = request.token
    if token:
        print(f"🔐 {request.model_name}에 접근하기 위한 인증 토큰이 제공되었습니다.")
    else:
        print(f"ℹ️ {request.model_name}에 접근할 때 인증 토큰이 제공되지 않았습니다. 게이티드 모델의 경우 접근이 제한될 수 있습니다.")
    
    # 모델 구성 파일 가져오기
    hf_config = fetch_model_config_from_hf(request.model_name, token=token)
    if hf_config is None:
        # 게이티드 모델에 대한 특별 처리
        if "llama" in request.model_name.lower() or "mistral" in request.model_name.lower():
            if not token:
                warning_message = (
                    f"{request.model_name}은(는) 게이티드 모델일 수 있습니다. "
                    "Hugging Face 토큰을 제공하여 접근 권한을 얻으세요. "
                    "모델 구성을 가져올 수 없어 추정치를 사용합니다."
                )
                print(f"⚠️ {warning_message}")
        
        estimated_values.append("model_configuration")
    config_context_length = None
    if hf_config:
        config_context_length = get_context_length_from_config(hf_config)
        if config_context_length is None:
            estimated_values.append("context_length_from_config")

    # 시퀀스 길이 결정
    if request.max_seq_len is not None:
        effective_max_seq_len = request.max_seq_len
        if config_context_length and effective_max_seq_len > config_context_length:
            raise HTTPException(
                status_code=400, 
                detail=f"입력한 max_seq_len({request.max_seq_len})이 모델의 컨텍스트 길이({config_context_length})보다 큽니다. max_seq_len을 모델의 컨텍스트 길이보다 작게 설정하세요."
            )
    elif config_context_length:
        effective_max_seq_len = config_context_length
        estimated_values.append("context_length_from_config")
    else:
        raise HTTPException(
                status_code=400, 
                detail="사용 가능한 시퀀스 길이가 없습니다. 이 모델의 Config 파일에는 컨텍스트 길이 정보가 포함되어 있지 않으므로 max_seq_len을 제공하세요."
        )
    
    if effective_max_seq_len <= 0:
        raise HTTPException(status_code=400, detail="max_seq_len은 양수여야 합니다")

    # max_num_seqs 확인
    if request.max_num_seqs <= 0:
        raise HTTPException(status_code=400, detail="max_num_seqs는 양수여야 합니다")
    
    # 매개변수 단위 변환
    if request.params_millions is not None and request.params_billions is None:
        request.params_billions = request.params_millions / 1000.0
    
    # KV 캐시 dtype 설정
    if request.kv_cache_dtype is None:
        request.kv_cache_dtype = request.dtype
    
    # dtype 유효성 확인
    if request.dtype not in DTYPE_SIZES:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 dtype: {request.dtype}. 지원되는 유형: {list(DTYPE_SIZES.keys())}")
    
    if request.kv_cache_dtype not in DTYPE_SIZES:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 KV 캐시 dtype: {request.kv_cache_dtype}. 지원되는 유형: {list(DTYPE_SIZES.keys())}")
    
    # 양자화 유효성 확인
    if request.quantization is not None and request.quantization not in QUANTIZATION_FACTORS:
        raise HTTPException(status_code=400, detail=f"지원되지 않는 양자화: {request.quantization}. 지원되는 유형: {list(QUANTIZATION_FACTORS.keys())}")
    
    # 모델 메모리 계산
    model_info = calculate_model_memory(
        request.model_name, 
        request.dtype, 
        hf_config,
        request.quantization,
        request.params_billions,
        request.params_millions
    )
    
    model_params_memory_gb = model_info["model_size_gb"]
    params_billions = model_info["params_billions"]
    
    # 매개변수 수 추정 여부 확인
    if request.params_billions is None and request.params_millions is None:
        if hf_config:
            params_from_config = get_params_from_hf_config(hf_config)
            if params_from_config is None:
                estimated_values.append("parameter_count")
        else:
            estimated_values.append("parameter_count")
    
    # 모델 아키텍처 가져오기, 유저 입력 아키텍처 우선 적용
    architecture = get_model_architecture(
        request.model_name,
        params_billions,
        hf_config,
        request.architecture
    )
    # 각 메모리 구성 요소 계산
    activation_memory_gb = calculate_activation_memory_inference(
        architecture,
        convert_max_num_seqs_to_base(request.max_num_seqs),
        effective_max_seq_len,
        request.dtype,
        request.model_name, # 모델 별 활성화 메모리 최적화 개수 하드 코딩용
        request.quantization # 양자화 적용
    )
    
    kv_cache_memory = calculate_kv_cache_memory(
        architecture,
        effective_max_seq_len,
        convert_max_num_seqs_to_base(request.max_num_seqs),
        request.kv_cache_dtype,
        effective_max_seq_len
    )
    
    min_kv_cache_memory_gb = kv_cache_memory["min_kv_cache_memory_gb"]
    max_kv_cache_memory_gb = kv_cache_memory["max_kv_cache_memory_gb"]
    
    # CUDA 오버헤드 추정
    cuda_overhead_gb = 0.25
    # 양자화 오버헤드 추정
    if request.quantization:
        cuda_overhead_gb += 0.25
    
    # 총 메모리 요구 사항 계산
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

    # GPU당 메모리 계산 (텐서 병렬 처리 지원)
    per_gpu_memory = calculate_per_gpu_memory(
        request.tensor_parallel_size,
        model_params_memory_gb,
        activation_memory_gb,
        min_kv_cache_memory_gb,
        max_kv_cache_memory_gb,
        cuda_overhead_gb
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
    
    # 값 반올림
    model_params_memory_gb = round(model_params_memory_gb, 2)
    activation_memory_gb = round(activation_memory_gb, 2)
    min_kv_cache_memory_gb = round(min_kv_cache_memory_gb, 2)
    max_kv_cache_memory_gb = round(max_kv_cache_memory_gb, 2)
    total_min_memory_gb = round(total_min_memory_gb, 2)
    total_max_memory_gb = round(total_max_memory_gb, 2)
    recommended_memory_gb = round(recommended_memory_gb, 2)
    
    for key in per_gpu_memory:
        per_gpu_memory[key] = round(per_gpu_memory[key], 2)
    
    # 컨텍스트 길이 소스 설정
    context_length_source = "config"
    if request.max_seq_len is not None:
        context_length_source = "사용자"
    elif "context_length_estimated" in estimated_values:
        context_length_source = "추정"
    
    # 메모리 구성 요소 요약
    components_breakdown = {
        "model_params": model_params_memory_gb,
        "activation": activation_memory_gb,
        "min_kv_cache": min_kv_cache_memory_gb,
        "max_kv_cache": max_kv_cache_memory_gb,
        "cuda_overhead": cuda_overhead_gb
    }

    # 경고 메시지 생성
    warning = None
    if estimated_values:
        if "parameter_count" in estimated_values and "model_architecture" in estimated_values:
            warning = "중요한 모델 세부 정보 및 아키텍처가 추정되어 부정확할 수 있습니다. 더 정확한 계산을 위해 이러한 값을 수동으로 제공하는 것을 고려하세요.\n"
        elif "parameter_count" in estimated_values:
            warning = "모델 파라미터 수가 모델 이름에서 추정되어 부정확할 수 있습니다. 더 정확한 계산을 위해 params_billions 또는 params_millions 제공을 고려하세요.\n"
        elif "model_architecture" in estimated_values:
            warning = "모델 아키텍처 세부 정보가 추정되어 부정확할 수 있습니다. 더 정확한 계산을 위해 아키텍처 세부 정보 제공을 고려하세요.\n"
        
        # 게이티드 모델에 대한 특별 경고
        if "model_configuration" in estimated_values and ("llama" in request.model_name.lower() or "mistral" in request.model_name.lower()):
            if not token:
                if warning:
                    warning += " 이 모델은 게이티드 모델일 수 있습니다. 더 정확한 결과를 위해 Hugging Face 토큰을 제공하세요.\n"
                else:
                    warning = "이 모델은 게이티드 모델일 수 있습니다. 더 정확한 결과를 위해 Hugging Face 토큰을 제공하세요.\n"
    if config_context_length and request.max_seq_len and request.max_seq_len > config_context_length:
        if warning:
            warning += (f"ℹ️ 입력한 입력 토큰 수({request.max_seq_len})가 모델 최대 시퀸스 길이 {config_context_length}보다 깁니다.\n")
        else:
            warning = (f"ℹ️ 입력한 입력 토큰 수({request.max_seq_len})가 모델 최대 시퀸스 길이 {config_context_length}보다 깁니다.\n")
    if missing_values:
        if warning:
            warning += f" 결정할 수 없는 누락된 값: {', '.join(missing_values)}."
        else:
            warning = f"결정할 수 없는 누락된 값: {', '.join(missing_values)}."
    
    if request.tensor_parallel_size > 1:
        multi_gpu_warning = f"텐서 병렬 처리를 위해 {request.tensor_parallel_size}개의 GPU에 걸쳐 메모리 요구 사항이 계산되었습니다.\n"
        if warning:
            warning += " " + multi_gpu_warning
        else:
            warning = multi_gpu_warning

    # GPU 메모리 활용 권장 사항
    recommended_gpu_memory_utilization = None
    if request.current_gpu_memory_size_gb:
        if request.current_gpu_memory_size_gb <= 0:
            raise HTTPException(status_code=400, detail="current_gpu_memory_size_gb는 양수여야 합니다")
        
        if request.tensor_parallel_size > 1:
            util_ratio = per_gpu_total_min_memory_gb / request.current_gpu_memory_size_gb
        else:
            util_ratio = total_min_memory_gb / request.current_gpu_memory_size_gb
        
        if util_ratio > 0.95:
            if warning:
                warning += "모델이 안전한 GPU 메모리 사용 비율(0.95)보다 더 많은 메모리를 필요로 합니다. 더 큰 GPU 사용, 배치 크기 줄이기 또는 양자화 적용을 고려하세요.\n"
            else:
                warning = "모델이 안전한 GPU 메모리 사용 비율(0.95)보다 더 많은 메모리를 필요로 합니다. 더 큰 GPU 사용, 배치 크기 줄이기 또는 양자화 적용을 고려하세요.\n"
            recommended_gpu_memory_utilization = None
        else:
            recommended_gpu_memory_utilization = round(util_ratio + 0.02, 2) # 2% 과대 예측 TODO: 추후 2% 오차 확인 필요
            recommended_gpu_memory_utilization = min(recommended_gpu_memory_utilization, 0.95) # TODO: 추후 최대값 설정 확인 필요

    # 최종 메모리 추정 반환
    print(f"✅ {request.model_name}에 대한 메모리 요구 사항 추정이 완료되었습니다!")
    current_gpu_memory_use_gb = round(current_gpu_memory_use_gb + per_gpu_total_min_memory_gb, 2)

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
        max_num_seqs=request.max_num_seqs,
        estimated_values=estimated_values,
        missing_values=missing_values,
        tensor_parallel_size=request.tensor_parallel_size,
        per_gpu_memory_breakdown=per_gpu_memory,
        per_gpu_total_min_memory_gb=per_gpu_total_min_memory_gb,
        per_gpu_total_max_memory_gb=per_gpu_total_max_memory_gb,
        current_gpu_memory_size_gb=current_gpu_memory_size_gb,
        current_gpu_memory_use_gb=current_gpu_memory_use_gb,
        model_params_count=request.params_billions if request.params_billions else estimate_params_from_name(request.model_name),
        model_memory_total_use_ratio=round(current_gpu_memory_use_gb / current_gpu_memory_size_gb + 0.02, 2) # 2% 과대 예측 추가 TODO: 추후 2% 오차 확인 필요
    )

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
            "/estimate-gpu-memory": "POST - GPU 메모리 요구 사항 계산"
        },
        "notes": [
            "시퀀스 길이(max_seq_len) 매개변수는 선택 사항입니다 - 제공되지 않으면 API는 모델의 구성에서 가져오려고 시도합니다",
            "모델의 구성에 컨텍스트 길이 정보가 없는 경우 max_seq_len을 제공해야 합니다",
            "다중 GPU 설정의 경우 GPU당 메모리 요구 사항을 확인하기 위해 'tensor_parallel_size' 매개변수를 지정하세요",
            "요청에 유저의 값이 있으면 해당 값을 최우선적으로 적용합니다"
        ]
    }
@app.get("/get_gpu_catalog")
def get_gpu_catalog():
    from csv_to_json import csv_to_json_file, json_file_to_string
    try:
        csv_to_json_file(GPU_CSV_FILE_PATH, GPU_JSON_FILE_PATH)
        gpu_catalog = json_file_to_string(GPU_JSON_FILE_PATH)
        return gpu_catalog
    except Exception as e:
        print(f"Error loading csv file: {e}")
        return {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)