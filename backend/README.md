# vLLM GPU 메모리 계산기 API 문서
https://www.notion.so/agilesoda/VLLM-GPU-API-CAG-1bb9f036b307805c91a5e59ee99b7248?pvs=18
## 개요

vLLM GPU 메모리 계산기 API는 vLLM 프레임워크로 대규모 언어 모델(LLM)을 서빙할 때 필요한 GPU 메모리 요구사항을 추정하기 위해 설계된 전문 도구입니다. 이 API는 모델 아키텍처, 배치 크기, 시퀀스 길이 및 기타 매개변수를 기반으로 상세한 메모리 분석을 제공하여 GPU 리소스를 계획하고 최적화하는 데 도움을 줍니다.

## 핵심 개념

API를 사용하기 전에 LLM을 서빙할 때 GPU 메모리 사용량에 기여하는 구성 요소를 이해하는 것이 도움이 됩니다:

1. **모델 파라미터 메모리**: 모델 가중치 자체를 저장하는 데 필요한 메모리입니다.
2. **활성화 메모리**: 순방향 전파 중에 중간 텐서에 필요한 메모리입니다.
3. **KV 캐시 메모리**: 효율적인 추론을 위해 키-값 캐시에 사용되는 메모리로, 컨텍스트 길이에 따라 증가합니다.
4. **CUDA 오버헤드**: CUDA 컨텍스트 및 추가 라이브러리가 소비하는 메모리입니다.

API는 다음과 같은 일반 방정식을 따릅니다:
```
총_메모리 = 모델_메모리 + 활성화_메모리 + kv_캐시_메모리 + cuda_오버헤드
```

## API 엔드포인트

### 1. GPU 메모리 추정

**엔드포인트**: `POST /estimate-gpu-memory`

이것은 특정 모델 구성에 필요한 예상 GPU 메모리를 계산하는 주요 엔드포인트입니다.

#### 요청 매개변수

| 매개변수 | 타입 | 필수 | 설명 |
|-----------|------|----------|-------------|
| `model_name` | string | 예 | 모델 이름 (예: "meta-llama/Llama-2-7b") |
| `max_seq_len` | integer | 예 | 추론을 위한 최대 시퀀스 길이 |
| `dtype` | string | 아니오 | 모델 가중치의 데이터 타입 (기본값: "float16") |
| `kv_cache_dtype` | string | 아니오 | KV 캐시의 데이터 타입 (지정되지 않으면 `dtype`과 동일) |
| `max_batch_size` | integer | 아니오 | 최대 배치 크기 (기본값: 32) |
| `max_model_len` | integer | 아니오 | 최대 모델 길이 (지정되지 않으면 `max_seq_len`과 동일) |
| `gpu_memory_utilization` | float | 아니오 | GPU 메모리 활용 비율 (기본값: 0.9) |
| `quantization` | string | 아니오 | 양자화 방법 ("Q4", "Q8" 또는 전체 정밀도는 null) |
| `params_billions` | float | 아니오 | 십억 단위의 매개변수 수 오버라이드 |
| `architecture` | object | 아니오 | 선택적 모델 아키텍처 세부 정보 |

`architecture` 객체에는 다음이 포함될 수 있습니다:

- `num_layers`: 트랜스포머 레이어 수
- `hidden_size`: 히든 차원 크기
- `num_heads`: 어텐션 헤드 수
- `head_dim`: 어텐션 헤드당 차원
- `intermediate_size`: 피드포워드 네트워크 레이어 크기

#### 요청 예시

```json
{
  "model_name": "meta-llama/Llama-2-7b",
  "max_seq_len": 2048,
  "dtype": "float16",
  "max_batch_size": 32,
  "gpu_memory_utilization": 0.9,
  "quantization": null,
  "params_billions": null,
  "architecture": {
    "num_layers": 32,
    "hidden_size": 4096,
    "num_heads": 32,
    "head_dim": 128,
    "intermediate_size": 11008
  }
}
```

#### 응답

API는 메모리 요구사항에 대한 상세한 분석을 반환합니다:

| 필드 | 설명 |
|-------|-------------|
| `model_name` | 모델 이름 |
| `model_params_memory_gb` | 모델 매개변수에 필요한 메모리 (GB) |
| `activation_memory_gb` | 활성화에 필요한 메모리 (GB) |
| `min_kv_cache_memory_gb` | PagedAttention을 사용한 최소 KV 캐시 메모리 (GB) |
| `max_kv_cache_memory_gb` | PagedAttention을 사용한 최대 KV 캐시 메모리 (GB) |
| `cuda_overhead_gb` | CUDA 컨텍스트에 필요한 메모리 (GB) |
| `total_min_memory_gb` | 최소 총 메모리 요구사항 (GB) |
| `total_max_memory_gb` | 최대 총 메모리 요구사항 (GB) |
| `recommended_memory_gb` | 활용 비율에 기반한 권장 GPU 메모리 |
| `warning` | 매우 큰 모델에 대한 선택적 경고 메시지 |
| `components_breakdown` | 메모리 구성 요소의 상세 내역 |
| `architecture_used` | 계산에 사용된 아키텍처 값 |

#### 응답 예시

```json
{
  "model_name": "meta-llama/Llama-2-7b",
  "model_params_memory_gb": 14.98,
  "activation_memory_gb": 3.25,
  "min_kv_cache_memory_gb": 0.42,
  "max_kv_cache_memory_gb": 53.69,
  "cuda_overhead_gb": 0.75,
  "total_min_memory_gb": 19.4,
  "total_max_memory_gb": 72.67,
  "recommended_memory_gb": 80.74,
  "warning": null,
  "components_breakdown": {
    "model_params": 14.98,
    "activation": 3.25,
    "min_kv_cache": 0.42,
    "max_kv_cache": 53.69,
    "cuda_overhead": 0.75
  },
  "architecture_used": {
    "num_layers": 32,
    "hidden_size": 4096,
    "num_heads": 32,
    "head_dim": 128,
    "intermediate_size": 11008
  }
}
```

### 2. 알려진 아키텍처 가져오기

**엔드포인트**: `GET /known-architectures`

시스템 데이터베이스에 있는 알려진 모델 아키텍처 목록을 반환합니다.

#### 응답

알려진 모델의 아키텍처를 포함하는 JSON 객체를 반환합니다.

### 3. 루트 엔드포인트

**엔드포인트**: `GET /`

예제 요청을 포함한, API에 대한 기본 정보를 반환합니다.

## 지원되는 모델

API는 다음과 같은 많은 인기 있는 모델 계열에 대한 사전 구성 정보를 포함하고 있습니다:

- **LLaMA 계열**: Llama 2 (7B, 13B, 70B), Llama 3 (8B, 70B), Llama 3.1 (8B, 70B, 405B), Llama 3.2 (1B, 3B)
- **Mistral 계열**: Mistral 7B, Mixtral 8x7B, Mistral Large/Medium, Mixtral 8x22B
- **OpenAI 모델**: GPT-3.5-Turbo, GPT-4, GPT-4o
- **Qwen 계열**: Qwen 1.5 (0.5B~110B)
- **Gemma 계열**: Gemma 2B, 7B
- **Claude 계열**: Claude 3 (Haiku, Sonnet, Opus)
- **Gemini 계열**: Gemini Nano, Pro, Ultra
- **Yi 계열**: Yi (6B, 9B, 34B)
- **Falcon 계열**: Falcon (7B, 40B, 180B)
- 기타 (StableLM, MPT, Pythia, OpenChat)

데이터베이스에 없는 모델의 경우, API는 모델 이름에 기반하여 매개변수를 추정하거나 기본값을 사용합니다.

## 결과 이해하기

### 메모리 구성 요소

1. **모델 파라미터 메모리**
   - 모델 가중치를 저장하는 데 필요한 메모리
   - 매개변수 수, 데이터 타입 및 양자화에 영향 받음

2. **활성화 메모리**
   - 추론 중 중간 텐서에 사용되는 메모리
   - 배치 크기, 시퀀스 길이 및 모델 차원에 따라 확장됨

3. **KV 캐시 메모리**
   - 어텐션 중 캐시된 키 및 값 텐서를 위한 메모리
   - 시퀀스 길이 및 배치 크기에 따라 증가
   - API는 PagedAttention으로 최소 및 최대 추정치를 계산

4. **CUDA 오버헤드**
   - CUDA 컨텍스트를 위한 고정 메모리 소비
   - 양자화 라이브러리에 대한 추가 오버헤드

### PagedAttention

API는 더 효율적인 메모리 사용을 위해 블록 단위로 메모리를 할당하는 vLLM의 PagedAttention 메커니즘을 통합합니다:

- **최소 KV 캐시**: 초기 할당을 나타냄 (시퀀스당 하나의 블록)
- **최대 KV 캐시**: 모든 블록이 사용되는 최악의 시나리오를 나타냄

### 권장 메모리

API는 다음을 기반으로 권장 GPU 메모리 값을 제공합니다:
- 최대 총 메모리 요구사항
- 사용자가 지정한 GPU 메모리 활용 비율 (기본값: 0.9)

이를 통해 GPU 메모리 할당의 메모리 단편화 및 기타 비효율성을 고려할 수 있습니다.

## 모범 사례

1. **모델 선택**
   - 정확한 결과를 위해 전체 모델 이름 사용 (예: "meta-llama/Llama-2-7b")
   - 알려지지 않은 모델의 경우, 가능하면 매개변수 수 제공

2. **아키텍처 세부 정보**
   - 최상의 정확도를 위해 알려진 아키텍처 세부 정보 제공
   - 제공되지 않은 경우 API는 매개변수 수에 기반한 추정치 사용

3. **메모리 최적화**
   - 대규모 모델의 경우 양자화(Q4, Q8) 고려
   - 처리량 요구사항에 따라 배치 크기 조정
   - 메모리 제약과 시퀀스 길이 간의 균형 유지

4. **해석**
   - GPU 하드웨어 선택 시 가이드라인으로 권장 메모리 사용
   - 매우 큰 모델에 대한 경고 메시지 고려

## 오류 처리

API는 유효하지 않은 입력에 대해 자세한 메시지와 함께 HTTP 400 오류를 반환합니다:

- 양수가 아닌 시퀀스 길이 또는 배치 크기
- 유효하지 않은 GPU 메모리 활용 비율
- 지원되지 않는 데이터 타입 또는 양자화 방법

## 기술적 세부사항

### 지원되는 데이터 타입

- `float32`: 매개변수당 4바이트
- `float16`: 매개변수당 2바이트
- `bfloat16`: 매개변수당 2바이트
- `int8`: 매개변수당 1바이트
- `int4`: 매개변수당 0.5바이트

### 양자화 방법

- `Q8`: 모델 크기를 2배로 줄임
- `Q4`: 모델 크기를 4배로 줄임ㄴ
- `null`: 양자화 없음 (전체 정밀도), null값 대신 미입력 상태 Request 제출

### 아키텍처 추정

알려지지 않은 아키텍처를 가진 모델의 경우, API는 정교한 추정 접근법을 사용합니다:

1. 레이어 수와 헤드 수에 대한 범위 기반 추정
2. 히든 차원을 위한 이차 방정식 해결
3. 중간 차원에 대한 스케일링 요소

이를 통해 매개변수 수에 기반한 현실적인 아키텍처 추정치를 생성합니다.

## 제한 사항

- 추정치는 근사치이며 실제 메모리 사용량은 다를 수 있습니다
- API는 모델 구현의 특정 최적화를 고려하지 않습니다
- 사용자 정의 모델 아키텍처는 수동 매개변수 지정이 필요할 수 있습니다
- API는 트랜스포머 기반 모델에 중점을 두며 다른 아키텍처에 대해서는 정확하지 않을 수 있습니다

## 활용 사례 예시

1. **하드웨어 계획**
   - 특정 모델이 사용 가능한 GPU 하드웨어에서 실행될 수 있는지 확인
   - 다중 모델 배포를 위한 GPU 리소스 할당 계획

2. **최적화 전략**
   - 메모리 요구사항에 대한 양자화 영향 평가
   - 메모리-시퀀스 길이 트레이드오프 분석
   - 최대 처리량을 위한 배치 크기 최적화

3. **비용 추정**
   - 클라우드 GPU 인스턴스 요구사항 추정
   - 모델 서빙에 대한 호스팅 비용 계산