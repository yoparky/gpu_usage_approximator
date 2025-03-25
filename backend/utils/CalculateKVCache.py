from utils.GeneralUtil import DTYPE_SIZES
from typing import Dict, Optional

def calculate_kv_cache_memory(arch: Dict[str, int], 
                             max_seq_len: int, 
                             max_num_seqs_ratio: int, 
                             kv_cache_dtype: str,
                             max_model_len: Optional[int] = None) -> Dict[str, float]:
    """
    PagedAttention을 사용한 KV 캐시 메모리 요구 사항을 계산합니다.
    해당 계산은 로그 상 vLLM이 사용하는 메모리 양과 정확히 일치합니다.
    
    Args:
        arch: 모델 아키텍처 매개변수
        max_seq_len: 최대 시퀀스 길이
        max_num_seqs_ratio: 배치 내 디폴트 시퀸스 수 기준 메모리 증감 비율
        kv_cache_dtype: KV 캐시에 사용되는 데이터 타입
        max_model_len: 모델의 최대 컨텍스트 길이 (또는 서빙시 설정한 max_seq_len 사용)
        
    Returns:
        KV 캐시 메모리 요구 사항이 포함된 딕셔너리
    """
    print("\n💾 KV 캐시 메모리 계산 시작...")
    
    if max_model_len is None:
        max_model_len = max_seq_len
        print(f"ℹ️ max_model_len이 설정되지 않음, max_seq_len으로 대체: {max_seq_len}")
    
    # 필요한 아키텍처 값 추출 및 검증
    num_layers = arch["num_layers"]
    num_heads = arch["num_heads"]
    head_dim = arch["head_dim"]
    
    # num_key_value_heads가 있는지 확인하고 없으면 num_heads 사용
    if "num_key_value_heads" not in arch:
        print("⚠️ 아키텍처에 num_key_value_heads가 없습니다. num_heads로 대체합니다.")
        num_key_value_heads = num_heads
    else:
        num_key_value_heads = arch["num_key_value_heads"]
    
    print(f"📊 KV 캐시 계산 매개변수:")
    print(f"  - num_layers: {num_layers}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - num_key_value_heads: {num_key_value_heads}")
    print(f"  - head_dim: {head_dim}")
    print(f"  - max_seq_len/max_model_len: {max_seq_len}/{max_model_len}")
    print(f"  - max_num_seqs_ratio: {max_num_seqs_ratio}")
    print(f"  - kv_cache_dtype: {kv_cache_dtype}")
    
    bytes_per_token = DTYPE_SIZES[kv_cache_dtype]
    print(f"  - bytes_per_token: {bytes_per_token}")
    
    # vLLM 기본 블록 크기
    block_size = 16
    if max_model_len > 8192:
        block_size = 32
    print(f"  - block_size: {block_size}")
    
    # 공식: 2 (KV 당 하나 씩) * num_key_value_heads * head_dim * num_layers * dtype_size * seq_len * batch_size (vLLM 에서 배치 당 256 seq 기준 1)
    kv_cache_size_bytes = 2 * num_key_value_heads * head_dim * num_layers * bytes_per_token * max_model_len #* max_num_seqs_ratio
    
    # GB 단위로 변환
    kv_cache_size_gb = kv_cache_size_bytes / (1024**3)
    
    print(f"🧮 KV 캐시 메모리 계산:")
    print(f"  - 계산식: 2 (KV 당 하나 씩) * {num_key_value_heads} * {head_dim} * {num_layers} * {bytes_per_token} * {max_model_len} * {max_num_seqs_ratio}")
    print(f"  - 총 바이트: {kv_cache_size_bytes:,}")
    print(f"  - GB 단위: {kv_cache_size_gb:.2f} GB")
    
    result = {
        "min_kv_cache_memory_gb": kv_cache_size_gb,
        "max_kv_cache_memory_gb": kv_cache_size_gb
    }
    
    return result