from utils.GeneralUtil import DTYPE_SIZES
from typing import Dict

def calculate_lora_fine_tuning_memory(
    arch: Dict[str, int],
    params_billions: float,
    batch_size: int, # max_num_seqs_ratio
    sequence_length: int,
    dtype: str,
    lora_rank: int = 8,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.1,
    adapter_layers_percentage: float = 0.3,
    optimizer: str = "adam",
    gradient_accumulation_steps: int = 1
) -> Dict[str, float]:
    """
    LoRA 미세 조정에 필요한 추가 메모리를 계산합니다.
    
    주의: 이 함수는 기존 추론 계산에서 이미 포함된 forward pass 활성화 메모리를 제외하고
    오직 LoRA 파인튜닝에 추가로 필요한 메모리만 계산합니다. 

    현재 API에서는 사용하지 않고 차 후 기능을 추가 할 계획입니다.
    
    Args:
        arch: 모델 아키텍처를 포함하는 딕셔너리
            - hidden_size: 모델의 은닉 차원 크기
            - num_heads: 어텐션 헤드 수
            - num_layers: 트랜스포머 레이어 수
        params_billions: 모델 크기(십억 단위 파라미터)
        batch_size: 훈련 배치 크기
        sequence_length: 시퀀스 길이
        dtype: 계산에 사용되는 데이터 타입(예: "float16", "float32")
        lora_rank: LoRA 어댑터 랭크 (r)
        lora_alpha: LoRA 스케일링 파라미터 (alpha)
        lora_dropout: LoRA 드롭아웃 비율
        adapter_layers_percentage: 전체 레이어 중 LoRA 어댑터를 적용할 레이어 비율
        optimizer: 사용되는 옵티마이저 ("adam", "adamw", "sgd")
        gradient_accumulation_steps: 그래디언트 누적 단계 수
        
    Returns:
        추가 메모리 구성 요소가 있는 딕셔너리(GB 단위)
    """
    # 아키텍처 매개변수 추출
    hidden_size = arch["hidden_size"]
    num_layers = arch["num_layers"]
    
    bytes_per_value = DTYPE_SIZES[dtype]
    
    # 옵티마이저 상태 크기 정의
    OPTIMIZER_STATES = {
        "sgd": 1,  # 모멘텀
        "adam": 2,  # 1차 및 2차 모멘트
        "adamw": 2  # 1차 및 2차 모멘트
    }
    optimizer_states = OPTIMIZER_STATES.get(optimizer.lower(), 2)
    
    # 어댑터 적용할 레이어 수 계산
    adapted_layers = int(num_layers * adapter_layers_percentage)
    
    # 1. LoRA 파라미터 메모리
    # 각 적응된 가중치 행렬에 대해 2개의 작은 행렬(A와 B) 저장
    # 일반적으로 어텐션 모듈의 Q, K, V, O 행렬에 적용
    matrices_per_layer = 4  # Q, K, V, 출력 프로젝션
    lora_params_per_layer = matrices_per_layer * 2 * hidden_size * lora_rank  # A와 B 행렬
    total_lora_params = lora_params_per_layer * adapted_layers
    lora_params_memory_bytes = total_lora_params * bytes_per_value
    
    # 2. 옵티마이저 상태
    # Adam/AdamW: 파라미터당 2개의 상태(m, v)
    # SGD+모멘텀: 파라미터당 1개의 상태
    optimizer_states_memory_bytes = total_lora_params * optimizer_states * bytes_per_value
    
    # 3. 그래디언트
    gradients_memory_bytes = total_lora_params * bytes_per_value
    
    # 4. 역전파를 위한 추가 메모리 (오직 추론 메모리 계산에 포함되지 않은 부분만)
    #    추론 계산은 이미 전방 패스 활성화(forward pass activations)를 포함함
    effective_batch_size = batch_size // gradient_accumulation_steps
    
    # 역전파를 위한 추가 임시 버퍼
    backward_temp_buffer_bytes = (
        effective_batch_size * 
        sequence_length * 
        hidden_size * 
        bytes_per_value * 
        2.0  # 역전파 중 임시 버퍼에 대한 추정치
    )
    
    # 5. 기타 훈련 관련 메모리 (그래디언트 계산, LoRA 합성 등)
    training_buffer_bytes = (
        effective_batch_size * 
        sequence_length * 
        hidden_size * 
        bytes_per_value
    )
    
    # 각 구성 요소를 GB로 변환
    lora_params_memory_gb = lora_params_memory_bytes / (1024**3)
    optimizer_states_memory_gb = optimizer_states_memory_bytes / (1024**3)
    gradients_memory_gb = gradients_memory_bytes / (1024**3)
    backward_temp_buffer_gb = backward_temp_buffer_bytes / (1024**3)
    training_buffer_gb = training_buffer_bytes / (1024**3)
    
    # 기존 메모리 요구 사항에 더해지는 총 추가 메모리
    # 활성화 메모리는 이제 포함하지 않음 (이미 추론 계산에 포함됨)
    total_additional_memory_gb = (
        lora_params_memory_gb +
        optimizer_states_memory_gb +
        gradients_memory_gb +
        backward_temp_buffer_gb +
        training_buffer_gb
    )
    
    return {
        "lora_params_memory_gb": round(lora_params_memory_gb, 2),
        "optimizer_states_memory_gb": round(optimizer_states_memory_gb, 2),
        "gradients_memory_gb": round(gradients_memory_gb, 2),
        "backward_buffer_memory_gb": round(backward_temp_buffer_gb, 2),
        "training_buffer_memory_gb": round(training_buffer_gb, 2),
        "total_lora_fine_tuning_memory_gb": round(total_additional_memory_gb, 2),
        "original_model_size_billions": params_billions,
        "lora_parameters_millions": round(total_lora_params / 1e6, 2),
        "lora_to_full_model_ratio": round((total_lora_params / (params_billions * 1e9)) * 100, 2)
    }
