#!/bin/bash

set -e

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    HAS_GPU=true
else
    HAS_GPU=false
fi

if [ "$HAS_GPU" = false ]; then
# ---------------------------------------------------------------------------
# Shakespeare char: quick CPU/MPS comparison of baseline vs AttnRes variants
# ---------------------------------------------------------------------------

SHAKESPEARE="config/train_shakespeare_char.py \
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

# Baseline (Standard Attention)
python train.py $SHAKESPEARE \
    --n_attn_res_blocks=0 \
    --out_dir=out-shakespeare-char-baseline \
    --wandb_run_name=shakespeare-baseline

# Block AttnRes (4 blocks of 4 sublayers each)
python train.py $SHAKESPEARE \
    --n_attn_res_blocks=4 \
    --out_dir=out-shakespeare-char-BlockAttnRes-B4 \
    --wandb_run_name=shakespeare-BlockAttnRes-B4

# Full AttnRes (2 * n_layer = 16)
python train.py $SHAKESPEARE \
    --n_attn_res_blocks=16 \
    --out_dir=out-shakespeare-char-FullAttnRes \
    --wandb_run_name=shakespeare-FullAttnRes

else
# ---------------------------------------------------------------------------
# OpenWebText: GPT-2 124M on 1x A100 40GB, ~1 hour
# Overrides the 8-GPU config: 1 GPU, ~600M tokens, expect val loss ~3.2-3.4
# ---------------------------------------------------------------------------

OWT_1GPU="config/train_gpt2.py \
    --gradient_accumulation_steps=5 \
    --max_iters=10000 \
    --lr_decay_iters=10000 \
    --warmup_iters=300 \
    --eval_interval=500"

# Baseline (Standard Attention)
python train.py $OWT_1GPU \
    --n_attn_res_blocks=0 \
    --out_dir=out-gpt2-baseline \
    --wandb_run_name=gpt2-124M-baseline

# Block AttnRes (4 blocks of 6 sublayers each)
python train.py $OWT_1GPU \
    --n_attn_res_blocks=4 \
    --out_dir=out-gpt2-BlockAttnRes \
    --wandb_run_name=gpt2-124M-BlockAttnRes

# Full AttnRes (2 * n_layer = 24)
python train.py $OWT_1GPU \
    --n_attn_res_blocks=24 \
    --out_dir=out-gpt2-FullAttnRes-B24 \
    --wandb_run_name=gpt2-124M-FullAttnRes-B24

fi
