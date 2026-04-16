#!/bin/bash

# Multi-GPU training script for PhysioDPO with DeepSpeed
# Usage: bash scripts/train_multi_gpu.sh

export CUDA_VISIBLE_DEVICES=0,1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

accelerate launch \
    --config_file /dev/null \
    --num_processes 2 \
    --num_machines 1 \
    --machine_rank 0 \
    --main_process_port $MASTER_PORT \
    --deepspeed_config_file configs/ds_config.json \
    --deepspeed_plugin_zero_init=true \
    src/train.py \
    --model_id hugohrban/progen2-xlarge \
    --data_path data/demo.json \
    --output_dir checkpoints/physio-dpo-progen2-xl-1m \
    --micro_batch_size 4 \
    --grad_accum_steps 16 \
    --max_steps 5000 \
    --learning_rate 5e-6 \
    --max_length 512 \
    --mu 50.0 \
    --tau 10.0 \
    --lambda_param 1.0
