#!/bin/bash

set -e

COMMON="config/train_shakespeare_char.py \
    --device=cpu \
    --compile=False \
    --eval_iters=20 \
    --log_interval=1 \
    --block_size=64 \
    --batch_size=12 \
    --n_layer=8 \
    --n_head=4 \
    --n_embd=128 \
    --max_iters=2000 \
    --lr_decay_iters=2000 \
    --dropout=0.0 \
    --wandb_log=True"

# Baseline (no AttnRes)
python train.py $COMMON \
    --n_attn_res_blocks=0 \
    --out_dir=out-shakespeare-char-baseline \
    --wandb_run_name=nano-gpt

# Block AttnRes (4 blocks)
python train.py $COMMON \
    --n_attn_res_blocks=4 \
    --out_dir=out-shakespeare-char-BlockAttnRes-L8B4 \
    --wandb_run_name=nano-gpt-BlockAttnRes-L8B4

# Full AttnRes (1 sublayer per block = 2 * n_layer)
python train.py $COMMON \
    --n_attn_res_blocks=16 \
    --out_dir=out-shakespeare-char-FullAttnRes-L8B16 \
    --wandb_run_name=nano-gpt-FullAttnRes-L8B16
