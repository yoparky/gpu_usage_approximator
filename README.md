

https://github.com/user-attachments/assets/397d5d8d-ec9d-45c8-a66c-009820a554a9

vLLM VRAM Requirement Calculator
This project is a web-based calculator designed to estimate the GPU VRAM required to serve Large Language Models (LLMs) using the vLLM inference engine. It consists of a Python backend that performs the calculations and a React frontend that provides an interactive user interface.

The calculator provides detailed memory analysis based on model architecture, sequence length, quantization, and tensor parallelism, helping users plan and optimize their GPU resources for LLM serving.

Features
Interactive Web UI: A user-friendly interface built with React and styled with Tailwind CSS.

Comprehensive Memory Analysis: Breaks down VRAM usage into model weights, KV cache, activation memory, and CUDA overhead.

Extensive Model Support: Pre-configured with architecture details for popular models (Llama, Mistral, Qwen, Gemma, etc.).

Architecture Estimation: Estimates model architecture for unknown models based on their parameter count.

Tensor Parallelism: Calculates memory requirements for multi-GPU setups using tensor parallelism.

Quantization Support: Models the impact of int4 and int8 quantization on memory.

Dynamic Visualization: Uses charts (Pie and Bar) to visualize the memory breakdown for easy interpretation.

GPU Auto-Selection: Includes a search-as-you-type feature to select a GPU and automatically set its memory capacity.

Real-time Feedback: Provides immediate warnings if the estimated memory exceeds the available GPU VRAM.

Technology Stack
Backend: Python, FastAPI (assumed)

Frontend: React, recharts (for charts), lucide-react (for icons), Tailwind CSS

Core Logic: The calculation is based on the memory allocation principles of the vLLM engine.

Project Structure
.
├── backend/      # Python API for memory calculation
└── frontend/     # React application for the user interface

Demo
Backend API Documentation
The backend exposes a REST API to perform the memory calculations.

Key Concepts

Model Parameters Memory: VRAM needed to store the model's weights. Dependent on the number of parameters, data type, and quantization.

Activation Memory: VRAM for intermediate tensors during the forward pass. Scales with sequence length and model dimensions.

KV Cache Memory: VRAM for caching key-value pairs in the attention mechanism. The API calculates minimum and maximum estimates based on vLLM's PagedAttention.

CUDA Overhead: Fixed memory consumed by the CUDA context and related libraries.

The total memory is calculated as:


Total Memory=Model Memory+Activation Memory+KV Cache Memory+CUDA Overhead
Endpoints

1. Estimate GPU Memory

This is the primary endpoint for calculating the estimated GPU memory for a given configuration.

Endpoint: POST /estimate-gpu-memory

Request Body:

Parameter

Type

Required

Description

model_name

string

Yes

The name of the model from Hugging Face (e.g., "meta-llama/Llama-3-8B").

max_seq_len

integer

No

Maximum sequence length for inference. Defaults to 4096.

tensor_parallel_size

integer

No

The number of GPUs for tensor parallelism. Defaults to 1.

max_num_seqs

integer

No

The maximum number of sequences processed concurrently. Defaults to 256.

current_gpu_memory_size_gb

float

No

The total memory of a single GPU in GB. Defaults to 24.

curr_gpu_use

float

No

The amount of VRAM already in use on the GPU in GB. Defaults to 0.

dtype

string

No

Data type for model weights (float16, bfloat16, float32). Defaults to the value in the model's config.

kv_cache_dtype

string

No

Data type for the KV cache. Defaults to dtype.

quantization

string

No

Quantization method ("int4", "int8"). null for none.

params_billions

float

No

Override the parameter count in billions. Required if the model is not known.

token

string

No

Hugging Face token for accessing gated models.

architecture

object

No

Manually specify model architecture details. See object structure below.

architecture Object:

Field

Type

Description

num_layers

integer

Number of transformer layers.

hidden_size

integer

Hidden dimension size.

num_heads

integer

Number of attention heads.

head_dim

integer

Dimension per attention head.

intermediate_size

integer

Size of the feed-forward network layer.

num_key_value_heads

integer

Number of key-value heads (for GQA/MQA).

Sample Response:

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
  "architecture_used": {
      "num_layers": 32,
      "hidden_size": 4096,
      "num_heads": 32,
      "head_dim": 128,
      "intermediate_size": 14336,
      "num_key_value_heads": 8
  }
}

2. Get Known Architectures

Returns a list of all model architectures pre-configured in the system.

Endpoint: GET /known-architectures

Response: A JSON object containing known model architectures.

3. Root

Returns basic information about the API.

Endpoint: GET /

Frontend Usage
The frontend provides an intuitive interface to interact with the API.

Model Configuration:

Enter the Model Name from Hugging Face. This is a required field.

If the model is not in the known database, you must provide its Parameter Count.

Adjust Max Sequence Length and Tensor Parallel Size as needed.

Hardware Configuration:

Use the GPU Search bar to find your GPU model. Selecting a GPU automatically sets the Current GPU Memory Size (GB).

If a portion of the GPU VRAM is already in use, specify it in the Currently Used GPU Memory (GB) field.

Advanced Options:

For fine-tuned control, expand the Advanced Options section to set max_num_seqs, dtype, kv_cache_dtype, and quantization.

You can also manually override the Model Architecture details if the API's defaults are incorrect for your use case.

Calculate & Review:

Click Calculate Memory.

The results panel will display a full breakdown of the memory requirements, including charts and warnings if the required memory exceeds the available capacity.

Local Setup and Installation
Prerequisites

Node.js and npm (or yarn)

Python 3.8+ and pip

Backend Setup

# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the API server (assuming FastAPI and Uvicorn)
# Replace 'main:app' with your actual file and app variable names
uvicorn main:app --host 0.0.0.0 --port 3001

Frontend Setup

# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the React development server
npm start

The application should now be running at http://localhost:3000.

Configuration

// frontend/src/VLLMMemoryCalculator.js
const API_URL = "http://localhost:3001"; // Change to your local backend URL

Limitations
The calculations are estimates and may differ from actual real-world usage due to implementation-specific optimizations or overhead.

The calculator is designed for transformer-based models and may not be accurate for other architectures.

The calculations are based on vLLM v0.7.3. Results may vary with different versions of vLLM.

