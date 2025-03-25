from pydantic import BaseModel
from typing import Optional, Dict, List

class MemoryEstimation(BaseModel):
    model_name: str
    model_params_memory_gb: float
    activation_memory_gb: float
    min_kv_cache_memory_gb: float
    max_kv_cache_memory_gb: float
    cuda_overhead_gb: float
    total_min_memory_gb: float
    total_max_memory_gb: float
    recommended_memory_gb: float
    warning: Optional[str] = None
    components_breakdown: Dict[str, float]
    architecture_used: Dict[str, int]
    recommended_gpu_memory_utilization: Optional[float] = None
    context_length_source: str = "사용자"
    context_length: int
    dtype: str
    kv_cache_dtype: str
    quantization: Optional[str] = None
    max_num_seqs: int
    estimated_values: List[str] = []
    missing_values: List[str] = []
    tensor_parallel_size: int = 1
    per_gpu_memory_breakdown: Dict[str, float] = {}
    per_gpu_total_min_memory_gb: float = 0.0
    per_gpu_total_max_memory_gb: float = 0.0
    current_gpu_memory_use_gb: float = 0.0
    current_gpu_memory_size_gb: float = 0.0
    recommended_gpu_memory_utilization: Optional[float] = None
    model_params_count: float
    model_memory_total_use_ratio: Optional[float] = None
