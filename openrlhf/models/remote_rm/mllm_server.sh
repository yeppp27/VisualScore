#!/bin/bash
while true; do
  CUDA_VISIBLE_DEVICES=6 python3  -m vllm.entrypoints.openai.api_server \
    --model /data/luyt//ckt/Qwen2.5-7B-Instruct \
    --tokenizer /data/luyt//ckt/Qwen2.5-7B-Instruct \
    --served-model-name Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.3 \
    --max-model-len 1024 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 512 \
    --tokenizer-mode slow \
    --host 0.0.0.0 \
    --port 8001
  echo "[vLLM crashed] Restarting in 5 seconds..."
  sleep 5
done