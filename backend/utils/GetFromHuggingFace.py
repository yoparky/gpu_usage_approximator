import json
import requests
from functools import lru_cache
from typing import Optional, Dict
from huggingface_hub import hf_hub_download
from utils.GeneralUtil import PARAM_MAPPINGS
from classes.ModelArchitecture import ModelArchitecture
from utils.GeneralUtil import estimate_model_architecture

@lru_cache(maxsize=100)
def fetch_model_config_from_hf(model_name: str, revision: str = None, token: Optional[str] = None) -> Optional[Dict]:
    """
    Hugging Face Hub에서 모델 구성을 가져옵니다.
    
    Args:
        model_name: Hugging Face Hub의 모델 이름
        revision: 다운로드할 특정 버전 (브랜치, 태그, 커밋 해시)
        token: Hugging Face Hub 접근을 위한 인증 토큰 (게이티드 모델에 필요)
        
    Returns:
        모델 구성을 포함하는 딕셔너리 또는 찾을 수 없는 경우 None
    """
    # 토큰 사용 여부 로깅
    if token:
        print(f"🔐 인증 토큰을 사용하여 {model_name} 접근을 시도합니다.")
    else:
        print(f"ℹ️ 토큰 없이 {model_name} 접근을 시도합니다. 게이티드 모델의 경우 접근이 제한될 수 있습니다.")
    
    # 먼저 hf_hub_download를 사용하여 config.json 파일 다운로드 시도
    try:
        config_file = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            repo_type="model",
            revision=revision,
            cache_dir=None,  # 기본 캐시 디렉토리 사용
            local_files_only=False,  # 온라인 리소스에 접근
            token=token,  # 인증 토큰 추가, 직접 입력
            etag_timeout=10,  # ETag 타임아웃 설정
        )
        
        print(f"📥 {model_name}의 config.json 파일을 성공적으로 다운로드했습니다.")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            # 구성 파일이 유효한지 확인
            if not config or not isinstance(config, dict):
                print(f"⚠️ {model_name}의 config.json 파일이 유효하지 않거나 비어 있습니다.")
                return None
                
            print(f"✅ {model_name}의 구성 파일 파싱 성공.")
            
            # num_key_value_heads 확인 로그 추가
            if "num_key_value_heads" in config:
                print(f"🔑 num_key_value_heads 발견: {config['num_key_value_heads']}, GQA 모델일 가능성이 높습니다.")
                
            return config
            
        except json.JSONDecodeError as e:
            print(f"❌ {model_name}의 구성 파일을 JSON으로 파싱하는 데 실패했습니다: {str(e)}")
    
    except Exception as e:
        # hf_hub_download 실패 시 API 요청으로 대체
        print(f"❌ hf_hub_download를 통한 {model_name} 구성 다운로드에 실패: {str(e)}")
        
        # 게이티드 모델 관련 에러 확인
        if "401" in str(e) or "403" in str(e) or "unauthorized" in str(e).lower() or "gated" in str(e).lower():
            print("⛔ 접근 거부됨: 모델이 게이티드 상태이며 유효한 토큰이 필요합니다.")
            if not token:
                print("💡 토큰을 제공하거나, Hugging Face에 로그인되어 있는지 확인하세요.")
            else:
                print("💡 제공된 토큰이 올바른지, 만료되지 않았는지, 그리고 필요한 권한이 있는지 확인하세요.")
            return None
        
        print("🔄 API 요청을 통해 구성 다운로드를 시도합니다...")
        
        branch = "main" if revision is None else revision
        api_url = f"https://huggingface.co/{model_name}/raw/{branch}/config.json"
        try:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            response = requests.get(api_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                try:
                    config = response.json()
                    
                    # API 응답이 유효한지 확인
                    if not config or not isinstance(config, dict):
                        print(f"⚠️ {model_name}의 API 응답이 유효하지 않거나 비어 있습니다.")
                        return None
                    
                    print(f"✅ {model_name}의 구성 파일을 API를 통해 성공적으로 다운로드했습니다.")
                    
                    # num_key_value_heads 확인 로그 추가
                    if "num_key_value_heads" in config:
                        print(f"🔑 num_key_value_heads 발견: {config['num_key_value_heads']}")
                        
                    return config
                    
                except json.JSONDecodeError:
                    print(f"❌ {model_name}의 API 응답을 JSON으로 파싱할 수 없습니다.")
            elif response.status_code in [401, 403]:
                print(f"⛔ 접근 거부됨 (HTTP {response.status_code}): 모델이 게이티드 상태이며 유효한 토큰이 필요합니다.")
            else:
                print(f"❌ {model_name}의 구성 파일을 API를 통해 가져오는 데 실패했습니다: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {model_name}의 API 요청 중 오류 발생: {str(e)}")
    
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
    
    Args:
        config: Hugging Face에서 가져온 모델 구성을 포함하는 딕셔너리
        
    Returns:
        표준화된 아키텍처 매개변수를 포함하는 딕셔너리
    """
    arch = {}
    
    # 어떤 값을 추출할지 로깅
    print("🔍 모델 구성에서 아키텍처 파라미터 추출 중...")
    
    # 구성에서 중요 매개변수 추출, YAML 파일에서 Huggingface 모델 config 키 추가 / 제거
    for our_key, possible_keys in PARAM_MAPPINGS.items():
        for key in possible_keys:
            if key in config:
                # TODO: 아키텍쳐 사용 방식 단순화 고려
                # 사용한 키 아키텍쳐에 임시 저장
                arch[our_key] = config[key]
                print(f"✅ {our_key} 파라미터 발견: {config[key]} (원본 키: {key})")
                break
    
    # 모델 유형 확인
    model_type = config.get("model_type", "").lower()
    print(f"📌 모델 유형: {model_type}")
    
    # head_dim이 없는 경우 계산
    # TODO: HEAD_DIM 계산 방식 더블체크
    if "head_dim" not in arch and "hidden_size" in arch and "num_heads" in arch:
        head_dim = arch["hidden_size"] // arch["num_heads"]
        if head_dim > 96:
            head_dim = round(head_dim / 32) * 32
        else:
            head_dim = 2 ** round(math.log2(head_dim))
        arch["head_dim"] = head_dim
        print(f"🧮 head_dim 계산: {head_dim}")
    
    # intermediate_size가 없는 경우 계산
    # TODO: INTERMEDIATE_SIZE 계산 방식 더블체크
    if "intermediate_size" not in arch and "hidden_size" in arch:
        if "llama" in model_type:
            multiplier = 2.75
        elif "mistral" in model_type:
            multiplier = 3.5
        else:
            multiplier = 4.0
            
        intermediate_size = int(arch["hidden_size"] * multiplier)
        intermediate_size = round(intermediate_size / 256) * 256
        arch["intermediate_size"] = intermediate_size
        print(f"🧮 intermediate_size 계산: {intermediate_size} (승수: {multiplier})")
    
    # num_key_value_heads가 없는 경우 설정, 대부분의 경우 존재. 없으면 (GQA 아닐 경우) num_heads와 동일하게 설정
    if "num_key_value_heads" not in arch and "num_heads" in arch:
        # GQA 또는 MQA 모델인지 확인
        is_gqa_model = False
        
        if "llama" in model_type and "3" in config.get("_name_or_path", ""):
            is_gqa_model = True
            print("🔍 Llama 3 모델 감지됨, GQA 구조일 가능성이 높습니다.")
        
        if "rope_scaling" in config and "llama3" in config.get("rope_scaling", {}).get("rope_type", "").lower():
            is_gqa_model = True
            print("🔍 Llama 3 로프 스케일링 감지됨, GQA 구조일 가능성이 높습니다.")
            
        if is_gqa_model:
            # Llama 3 모델의 경우 num_key_value_heads 추정
            # 참고: 이것은 추정일 뿐이며, 실제 값이 다를 수 있음
            arch["num_key_value_heads"] = arch["num_heads"] // 3
            print(f"⚠️ num_key_value_heads 추정: {arch['num_key_value_heads']} (Llama 3 GQA 추정)")
        else:
            # 기본적으로 num_heads와 동일하게 설정
            arch["num_key_value_heads"] = arch["num_heads"]
            print(f"ℹ️ num_key_value_heads을 num_heads와 동일하게 설정: {arch['num_heads']}")
    else:
        print(f"✅ num_key_value_heads 파라미터 사용: {arch['num_key_value_heads']}")
    
    # 아키텍처 요약
    print("📋 생성된 모델 아키텍처 요약:")
    for key, value in arch.items():
        print(f"  - {key}: {value}")
    
    return arch


def get_model_architecture(model_name: str, 
                           params_billions: float, 
                           hf_config: dict = None,
                           user_arch: Optional[ModelArchitecture] = None) -> Dict[str, int]:
    """
    모델 아키텍처 세부 정보를 가져오며, 우선순위는 다음과 같습니다:
    1. 사용자 제공 아키텍처 세부 정보(최우선)
    2. Hugging Face 구성(대체)
    3. 매개변수 수에 따른 추정(최후 수단)
    
    참고: 사용자 값은 항상 HuggingFace 또는 추정에서 가져온 값을 재정의합니다
    """
    arch = estimate_model_architecture(params_billions)
    
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
                print(f"🔑 사용자 제공 아키텍처로 변경: {key} = {value}")
    
    return arch


def get_context_length_from_config(config: Dict) -> Optional[int]:
    """
    모델 구성에서 컨텍스트 창 크기를 추출합니다. 대부분의 경우 유저의 직접 입력이 계산에
    사용됩니다. 차후 사용하기 위한 코드로 현재는 프론트엔드에서 4096의 디폴트값이 설정
    되어있습니다.
    
    Args:
        config: Hugging Face에서 가져온 모델 구성을 포함하는 딕셔너리
        
    Returns:
        최대 시퀀스 길이를 나타내는 정수 또는 찾을 수 없는 경우 None
    """
    context_length_fields = PARAM_MAPPINGS['context_length']

    for field in context_length_fields:
        if field in config:
            return config[field]
    
    if "rope_scaling" in config:
        base_length = config.get("max_position_embeddings", 0)
        scaling_factor = config["rope_scaling"].get("factor", 1)
        if base_length and scaling_factor > 1:
            return int(base_length * scaling_factor)
    
    return None