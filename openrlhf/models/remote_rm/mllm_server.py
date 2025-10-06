python3 -m vllm.entrypoints.openai.api_server \
    --model /data/luyt/ckt//ckt/Qwen2.5-7B-Instruct \
    --tokenizer /data/luyt/ckt//ckt/Qwen2.5-7B-Instruct


CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model /data/luyt/ckt//ckt/Qwen2.5-7B-Instruct \
    --tokenizer /data/luyt/ckt//ckt/Qwen2.5-7B-Instruct \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.7 \
    --host 0.0.0.0 \
    --port 8080



CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model /data/luyt/ckt//ckt/Qwen2.5-7B-Instruct \
  --tokenizer /data/luyt/ckt//ckt/Qwen2.5-7B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.6 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 1024 \
  --host 0.0.0.0 \
  --port 8000


CUDA_VISIBLE_DEVICES=0 nohup  python3 -m vllm.entrypoints.openai.api_server \
  --model /data/luyt/ckt//ckt/Qwen2.5-7B-Instruct \
  --tokenizer /data/luyt/ckt//ckt/Qwen2.5-7B-Instruct \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.5 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --host 0.0.0.0 \
  --port 8000 \
  > vllm_server.log 2>&1 &



curl http://10.140.0.155:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "1+1等于几？"}]
      }'
