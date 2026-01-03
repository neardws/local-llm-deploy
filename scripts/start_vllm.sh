#!/bin/bash
# vLLM Server Startup Script
# Usage: ./start_vllm.sh [model_name] [tensor_parallel_size]

MODEL_NAME="${1:-Qwen/Qwen2.5-7B-Instruct}"
TP_SIZE="${2:-2}"
PORT="${3:-8000}"
GPU_UTIL="${4:-0.9}"

echo "Starting vLLM server..."
echo "Model: $MODEL_NAME"
echo "Tensor Parallel Size: $TP_SIZE"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_UTIL"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code
