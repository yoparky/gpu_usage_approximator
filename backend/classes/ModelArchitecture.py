from pydantic import BaseModel
from typing import Optional

class ModelArchitecture(BaseModel):
    """계산에 필요한 모델 아키텍처 세부 정보"""
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None