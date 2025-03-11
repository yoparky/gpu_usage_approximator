import argparse
import time
import os
import subprocess
import torch
from vllm import LLM, SamplingParams

def get_gpu_memory_usage():
    """Get the current GPU memory usage in GB."""
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    # Convert to GB
    return [float(x) / 1024 for x in result.strip().split('\n')]

def clear_gpu_cache():
    """Clear GPU cache to get a clean measurement."""
    torch.cuda.empty_cache()
    
def main():
    parser = argparse.ArgumentParser(description="Test vLLM model loading and memory usage")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B", help="Model name")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="Data type")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Maximum batch size")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    
    args = parser.parse_args()
    
    # Print initial GPU state
    print(f"Initial GPU memory usage: {get_gpu_memory_usage()[args.gpu_id]:.2f} GB")
    
    # Clear cache
    clear_gpu_cache()
    
    # Record start time
    start_time = time.time()
    
    # Configure tensor parallelism based on your setup
    # (remove this if you're using a single GPU)
    tensor_parallel_size = 1
    
    # Load the model
    print(f"Loading model {args.model} with context length {args.max_seq_len}...")
    llm = LLM(model=args.model, 
              tensor_parallel_size=tensor_parallel_size,
              max_seq_len=args.max_seq_len,
              dtype=args.dtype,
              gpu_memory_utilization=0.90)
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Get memory usage after loading
    gpu_mem = get_gpu_memory_usage()[args.gpu_id]
    print(f"Current GPU memory usage: {gpu_mem:.2f} GB")
    
    # Run a sample inference to ensure all memory is allocated
    print("Running sample inference...")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
    outputs = llm.generate(["The capital of France is", "The best programming language is"], sampling_params)
    
    # Final memory measurement
    final_mem = get_gpu_memory_usage()[args.gpu_id]
    print(f"GPU memory usage after inference: {final_mem:.2f} GB")
    
    # Keep the model loaded to allow for manual memory inspection
    input("Press Enter to exit and release GPU memory...")

if __name__ == "__main__":
    main()