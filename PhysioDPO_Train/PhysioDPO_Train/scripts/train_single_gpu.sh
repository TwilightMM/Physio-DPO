#!/bin/bash

# Single GPU training script for PhysioDPO
# Usage: bash scripts/train_single_gpu.sh

# Configure the HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300

export CUDA_VISIBLE_DEVICES=0

python src/train.py \
    --model_id hugohrban/progen2-xlarge \
    --data_path data/demo.json \
    --output_dir checkpoints/physio-dpo-progen2-xl-1m-single \
    --micro_batch_size 1 \
    --grad_accum_steps 8 \
    --max_steps 5000 \
    --learning_rate 5e-6 \
    --max_length 512 \
    --mu 50.0 \
    --tau 10.0 \
    --lambda_param 1.0 \
    --single_gpu
