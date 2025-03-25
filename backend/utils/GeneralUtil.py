import re
import math
from typing import Dict
import yaml

PATH_TO_API_DEFINES = "./data/defines.yaml"

# 모델 데이터 타입 크기, 양자화 계수, 파라미터 매핑 yaml로드
def load_model_config(file_path=PATH_TO_API_DEFINES):
    """API 컨피규레이션 YAML 파일에서 로딩."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"API 컨피규레이션 YAML 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        raise Exception(f"API 컨피규레이션 YAML 파일을 로드하는 중 오류가 발생했습니다: {str(e)}")
    
    # 설정값 추출
    dtype_sizes = config['DTYPE_SIZES']
    
    # 양자화 개수 중 YAML null 키 처리, 코드 상 None은 양자화 개수 1
    quantization_factors = config['QUANTIZATION_FACTORS']
    if 'null' in quantization_factors:
        _ = quantization_factors.pop('null')
        quantization_factors[None] = 1
    
    # 유저의 필요에 따라 YAML에 항목 추가
    param_mappings = config['PARAM_MAPPINGS']

    return dtype_sizes, quantization_factors, param_mappings

# 글로벌 디파인
DTYPE_SIZES, QUANTIZATION_FACTORS, PARAM_MAPPINGS = load_model_config()

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
    # 못 찾으면 7B로 추정
    return 7.0


def estimate_model_architecture(params_billions: float) -> Dict[str, int]:
    """
    스케일링 법칙을 사용하여 매개변수 수에 따라 모델 아키텍처 세부 정보를 추정합니다.
    굉장히 부정확하며 최후의 수단으로 사용합니다.
    
    Args:
        params_billions: 수십억 단위의 모델 크기
    Returns:
        추정된 아키텍처 세부 정보가 포함된 딕셔너리
    """
    # 잘 알려진 모델 크기(예: 7B, 8B, 13B, 70B 모델)에 대한 특별 사례 처리
    # 이는 일반적인 아키텍처를 기반으로 하드코딩되어 있습니다
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
    
    # 완전한 아키텍처 사양 반환, 입력값, Config값에 없을 때 만 사용.
    return {
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size
    }