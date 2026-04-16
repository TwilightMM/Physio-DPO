#!/bin/bash

# Inference script for PhysioDPO
# Usage: bash scripts/inference.sh

export CUDA_VISIBLE_DEVICES=0

python src/inference.py \
    --model_path checkpoints/physio-dpo-progen2-xl-1m-single/checkpoint-1000 \
    --base_model hugohrban/progen2-xlarge \
    --test_data data/demo.json \
    --output_dir results \
    --num_samples 100 \
    --max_length 256 \
    --temperature 0.8 \
    --device cuda

# To compute pLDDT scores (requires ESMFold), add --compute_plddt flag:
# python src/inference.py \
#     --model_path checkpoints/physio-dpo-progen2-xl-1m \
#     --base_model hugohrban/progen2-xlarge \
#     --test_data data/physiopref_1m.json \
#     --output_dir results \
#     --num_samples 100 \
#     --max_length 256 \
#     --temperature 0.8 \
#     --device cuda \
#     --compute_plddt
