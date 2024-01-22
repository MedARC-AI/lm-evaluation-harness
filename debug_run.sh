#!/bin/bash
set -e

DATASET=pubmedqa
OUT_PATH="/weka/home-griffin/qwen_${DATASET}.jsonl"

echo "Running ${DATASET} and saving to ${OUT_PATH}"
lm_eval \
    --output_path $OUT_PATH \
    --log_samples \
    --model hf \
    --model_args pretrained=Qwen/Qwen-72B,trust_remote_code=True,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8,quantized=model.safetensors,gptq_use_triton=True \
    --device cuda \
    --batch_size auto \
    --tasks ${DATASET}_medprompt
