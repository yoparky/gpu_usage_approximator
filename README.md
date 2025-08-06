# vLLM VRAM Requirement Calculator 🚀

![GitHub](https://img.shields.io/badge/vLLM-Memory_Calculator-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![React](https://img.shields.io/badge/React-18-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)

A comprehensive web-based calculator for estimating GPU VRAM requirements when serving Large Language Models (LLMs) using the vLLM inference engine. This tool helps ML engineers and researchers plan and optimize their GPU resources for efficient LLM deployment.

## 📸 Demo

![vLLM Calculator Demo](https://github.com/user-attachments/assets/397d5d8d-ec9d-45c8-a66c-009820a554a9)

## ✨ Features

### Core Functionality
- **📊 Interactive Web Interface** - User-friendly React frontend with Tailwind CSS styling
- **🔍 Comprehensive Memory Analysis** - Detailed breakdown of VRAM usage components
- **🤖 Extensive Model Support** - Pre-configured architectures for popular models (Llama, Mistral, Qwen, Gemma, etc.)
- **🧮 Smart Architecture Estimation** - Automatic architecture detection for unknown models based on parameter count
- **🖥️ Multi-GPU Support** - Calculate requirements for tensor parallelism setups
- **⚡ Quantization Support** - Model the impact of int4 and int8 quantization on memory usage

### User Experience
- **📈 Dynamic Visualization** - Interactive pie and bar charts for memory breakdown analysis
- **🔎 GPU Auto-Selection** - Search-as-you-type feature with automatic VRAM capacity configuration
- **⚠️ Real-time Validation** - Immediate warnings when estimated memory exceeds available VRAM
- **🎯 Precision Calculations** - Based on vLLM v0.7.3 memory allocation principles

## 🛠️ Technology Stack

| Component | Technologies |
|-----------|-------------|
| **Backend** | Python, FastAPI |
| **Frontend** | React, Recharts, Lucide-react, Tailwind CSS |
| **Core Engine** | vLLM memory allocation algorithms |

## 📁 Project Structure

```
vllm-vram-calculator/
├── backend/          # Python API for memory calculations
│   ├── main.py      # FastAPI application
│   └── requirements.txt
└── frontend/         # React application
    ├── src/
    │   └── VLLMMemoryCalculator.js
    ├── package.json
    └── public/
```

## 🧠 Core Concepts

The calculator estimates four key memory components:

| Component | Description |
|-----------|-------------|
| **Model Parameters Memory** | VRAM for storing model weights (affected by quantization) |
| **Activation Memory** | VRAM for intermediate tensors during forward pass |
| **KV Cache Memory** | VRAM for attention mechanism key-value pairs (uses PagedAttention) |
| **CUDA Overhead** | Fixed memory consumed by CUDA context and libraries |

### Memory Calculation Formula

```
Total Memory = Model Memory + Activation Memory + KV Cache Memory + CUDA Overhead
```

## 🔌 API Documentation

### Primary Endpoint: Estimate GPU Memory

**Endpoint:** `POST /estimate-gpu-memory`

#### Request Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model_name` | string | ✅ | - | Hugging Face model identifier (e.g., "meta-llama/Llama-3-8B") |
| `max_seq_len` | integer | ❌ | 4096 | Maximum sequence length for inference |
| `tensor_parallel_size` | integer | ❌ | 1 | Number of GPUs for tensor parallelism |
| `max_num_seqs` | integer | ❌ | 256 | Maximum concurrent sequences |
| `current_gpu_memory_size_gb` | float | ❌ | 24 | Single GPU memory capacity (GB) |
| `curr_gpu_use` | float | ❌ | 0 | Currently used VRAM (GB) |
| `dtype` | string | ❌ | auto | Data type (float16, bfloat16, float32) |
| `kv_cache_dtype` | string | ❌ | dtype | KV cache data type |
| `quantization` | string | ❌ | null | Quantization method ("int4", "int8", or null) |
| `params_billions` | float | ❌ | - | Override parameter count (billions) |
| `token` | string | ❌ | - | Hugging Face token for gated models |
| `architecture` | object | ❌ | - | Manual architecture specification |

#### Architecture Object Structure

```json
{
  "num_layers": 32,
  "hidden_size": 4096,
  "num_heads": 32,
  "head_dim": 128,
  "intermediate_size": 14336,
  "num_key_value_heads": 8
}
```

#### Sample Response

```json
{
  "model_name": "meta-llama/Llama-3-8B",
  "model_params_count": 8.0,
  "context_length": 8192,
  "dtype": "bfloat16",
  "total_min_memory_gb": 24.8,
  "components_breakdown": {
    "model_params": 15.0,
    "activation": 4.1,
    "min_kv_cache": 4.2,
    "cuda_overhead": 1.5
  },
  "per_gpu_total_min_memory_gb": 24.8,
  "per_gpu_memory_breakdown": {
    "per_gpu_model_params": 7.5,
    "per_gpu_activation": 4.1,
    "per_gpu_min_kv_cache": 2.1,
    "per_gpu_cuda_overhead": 1.5
  },
  "recommended_gpu_memory_utilization": 0.95,
  "model_memory_total_use_ratio": 1.03,
  "current_gpu_memory_use_gb": 25.8,
  "current_gpu_memory_size_gb": 24.0,
  "warning": "The estimated memory requirement exceeds the available GPU memory.",
  "estimated_values": ["context_length"],
  "architecture_used": {...}
}
```

### Additional Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/known-architectures` | GET | Returns list of pre-configured model architectures |
| `/` | GET | Basic API information |

## 🚀 Quick Start

### Prerequisites

- Node.js 14+ and npm/yarn
- Python 3.8+
- pip package manager

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --host 0.0.0.0 --port 3001
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Update API URL in src/VLLMMemoryCalculator.js
# const API_URL = "http://localhost:3001";

# Start React development server
npm start
```

The application will be available at `http://localhost:3000`

## 💻 Usage Guide

### Step 1: Model Configuration
1. Enter the **Model Name** from Hugging Face (required)
2. For unknown models, provide the **Parameter Count** in billions
3. Adjust **Max Sequence Length** and **Tensor Parallel Size** as needed

### Step 2: Hardware Configuration
1. Use the **GPU Search** bar to find and select your GPU model
2. GPU selection automatically sets memory capacity
3. Specify any **Currently Used GPU Memory** if applicable

### Step 3: Advanced Options (Optional)
Expand the Advanced Options section to configure:
- Maximum number of sequences (`max_num_seqs`)
- Data types (`dtype`, `kv_cache_dtype`)
- Quantization settings
- Manual architecture overrides

### Step 4: Calculate & Analyze
1. Click **Calculate Memory**
2. Review the detailed breakdown with charts
3. Check for any warnings about memory constraints

## ⚠️ Limitations

- Calculations are **estimates** and may vary from actual usage due to implementation-specific optimizations
- Designed specifically for **transformer-based models** - accuracy may vary for other architectures
- Based on **vLLM v0.7.3** - results may differ with other vLLM versions
- Does not account for dynamic memory allocation patterns during runtime


## 🙏 Acknowledgments

- Built for the vLLM inference engine community
- Inspired by the need for better resource planning in LLM deployment

---

**Note:** This calculator provides estimates to help with capacity planning. Always validate with actual deployment testing for production use cases.
