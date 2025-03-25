from typing import Optional, Dict
from utils.GetFromHuggingFace import get_params_from_hf_config
from utils.GeneralUtil import estimate_params_from_name, DTYPE_SIZES, QUANTIZATION_FACTORS

def get_model_params_count(model_name: str, 
                          hf_config: dict = None,
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
    
    if hf_config:
        params_from_config = get_params_from_hf_config(hf_config)
        if params_from_config:
            return params_from_config
    
    return estimate_params_from_name(model_name)


def calculate_model_memory(model_name: str, dtype: str, 
                          hf_config: dict = None,
                          quantization: Optional[str] = None,
                          params_billions: Optional[float] = None,
                          params_millions: Optional[float] = None) -> Dict[str, float]:
    """
    모델 매개변수에 필요한 메모리를 GB 단위로 계산합니다.
    모델은 정확히 nB개의 파라미터를 가지고 있지 않고 보통 약간 더 많은 파라미터를 가지고 있습니다.
    이를 위해 마지막 값에 5%의 오차값을 더합니다.
    """
    params_billions = get_model_params_count(
        model_name, 
        hf_config,
        params_billions,
        params_millions
    )
    print(params_billions)
    
    bytes_per_param = DTYPE_SIZES[dtype]
    
    quant_factor = QUANTIZATION_FACTORS.get(quantization, 1)
    
    model_size_bytes = params_billions * 1e9 * bytes_per_param
    # nB 모델들은 정확히 nB개의 파라미터 수 보다 많으므로 5%의 오차를 추가합니다.
    model_size_gb = model_size_bytes / (1024**3) / quant_factor * 1.05
    
    return {
        "params_billions": params_billions,
        "model_size_gb": model_size_gb
    }
