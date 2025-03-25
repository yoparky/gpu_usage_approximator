from pydantic import BaseModel
from typing import Optional 
from classes.ModelArchitecture import ModelArchitecture

class ModelRequest(BaseModel):
    model_name: str
    max_seq_len: Optional[int] = None
    dtype: str = "float16"
    kv_cache_dtype: Optional[str] = None
    max_num_seqs: int = 256
    quantization: Optional[str] = None
    params_billions: Optional[float] = None
    params_millions: Optional[float] = None
    architecture: Optional[ModelArchitecture] = None
    current_gpu_memory_size_gb: Optional[float] = None
    tensor_parallel_size: int = 1
    token: Optional[str] = None  # 게이티드 모델용 Hugging Face 토큰
    # 유저 직접 입력하여 오버라이드
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_key_value_heads: Optional[float] = None
    curr_gpu_use: Optional[float] = 0
