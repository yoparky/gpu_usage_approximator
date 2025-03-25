from utils.GeneralUtil import DTYPE_SIZES, QUANTIZATION_FACTORS
from typing import Dict
import yaml
PATH_TO_ACTIVATION_DEFINES = "./data/activation_defines.yaml"

def calculate_activation_memory_inference(
    arch: Dict[str, int],
    max_num_seqs_ratio: int,
    max_seq_len: int,
    dtype: str,
    model_name: str,
    quantization: str
) -> float:
    """
    KV 캐시를 제외한 추론을 위한 활성화 메모리를 계산합니다.
    이론적인 수치와 오차가 상당합니다. 이는 모델별로 사용된 최적화 기법,
    레이어 수 등을 포함한 많은 변수가 있어 모델 종류마다 조금씩 다릅니다.
    """
    # 아키텍처 매개변수 추출
    hidden_size = arch["hidden_size"]
    intermediate_size = arch.get("intermediate_size", 4 * hidden_size)
    bytes_per_value = DTYPE_SIZES[dtype]

    try: # YAML에서 모델 별 활성화 메모리 감소 개수 추출
        with open(PATH_TO_ACTIVATION_DEFINES, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"API 컨피규레이션 YAML 파일을 찾을 수 없습니다: {PATH_TO_ACTIVATION_DEFINES}")
    except Exception as e:
        raise Exception(f"API 컨피규레이션 YAML 파일을 로드하는 중 오류가 발생했습니다: {str(e)}")
    
    model_activation_ratio = config['MODEL_ACTIVATION_RATIO']
    
    # 하드 코딩 영역, vLLM 0.7.3기준 최적 시퀀스 길이 1024 ~ 16384 사이 값을 8192로 고정 (실측으로 측정)
    if 1024 <= max_seq_len and max_seq_len <= 16384:
        max_seq_len = 8192
    
    # NVIDIA 논문(2022) "Reducing Activation Recomputation in Large Transformer Models"에 기반한 계산
    # 이는 원래 트랜스포머 모델이 사용하는 접근으로 새로운 모델들에 적용된 각종 변화를 고려하지 않아 오차가 상당합니다.
    # 오차를 잡기 위해 마지막 값에 모델별로 특징적인 오차값을 바꿉니다.
    
    # 어텐션 블록 메모리 = 10 * s * b * h
    # - QK^T 행렬 곱셈: 4 * s * b * h (Q와 K의 저장 필요)
    # - QKV 행렬 곱셈: 2 * s * b * h (공유 입력 저장)
    # - 값(V)에 대한 어텐션: 2 * s * b * h
    # - 선형 투영: 2 * s * b * h
    attention_block_memory = 10 * max_seq_len * hidden_size * bytes_per_value
    
    # MLP 활성화 메모리 = 4 * s * b * (i + h)
    # - self.gate_proj(x): 2 * s * b * h
    # - self.act_fn: 2 * s * b * i
    # - self.up_proj(x): 2 * s * b * h
    # - self.down_proj: 2 * s * b * i
    mlp_activation_memory = 4 * max_seq_len * (hidden_size + intermediate_size) * bytes_per_value
    
    # 레이어 정규화 메모리 = 4 * s * b * h
    # - 각 레이어에는 두 개의 정규화 작업 포함, 2 * s * b * h
    # - 일반적으로 총 2개의 정규화 레이어
    layer_norm_memory = 4 * max_seq_len * hidden_size * bytes_per_value
    
    # 총 활성화 메모리 (단일 레이어에 대한 피크 메모리)
    # 위 세 구성 요소를 합하면 다음 공식과 같습니다:
    # pytorch_activation_peak_memory = sequence_length * batch_size * (18 * hidden_size + 4 * intermediate_size)
    per_256_pytorch_activation_peak_memory = attention_block_memory + mlp_activation_memory + layer_norm_memory
    
    quant_factor = QUANTIZATION_FACTORS.get(quantization, 1)
    # GB 단위로 변환, 양자화 적용
    total_activation_gb = per_256_pytorch_activation_peak_memory / (1024**3) / quant_factor
    # vLLM에서 각 배치당 넣을 수 있는 최대 스퀸스 수를 정할 수 있습니다. 디폴트 값은 256입니다. 
    # 동시 디코딩을 진행하는 시퀸스 수가 많아질수록 활성화 메모리가 증가합니다.
    # 256을 기준으로 해당 값을 1의 비율로 잡아 추가되는 시퀸스 수를 비율로 계산을 진행합니다.
    # 에를 들어 512 시퀸스 수는 256의 두배이므로 2, 128은 반이므로 0.5의 활성화 메모리를 필요합니다
    # 64 밑으로는 오버헤드값이 계산값의 대부분이라 128과 같은 값으로 계산합니다.
    total_activation_gb = (total_activation_gb) * max_num_seqs_ratio # 위에 말한 비율입니다.

    # 해당 계산은 Pytorch, Cuda 커널, 메모리 패딩, 모델의 특성등을 고려하지 않아 활성화 메모리의 값을 구함에 있어 오차가 있습니다.
    # 따라서 오차값을 250mb 정도 추가하여 보정합니다.
    total_activation_gb += 0.25
    
    # 하드 코딩 영역, YAML에서 모델 활성화 메모리 감소 비율 추출
    # 모델별로 오차 비율을 직접 YAML에서 추가할 수 있습니다.
    if model_name in model_activation_ratio:
        total_activation_gb *= model_activation_ratio[model_name]

    return total_activation_gb