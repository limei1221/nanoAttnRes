# nanoAttnRes

**nanoAttnRes** is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT) that adds **Block / Full Attention Residuals** from Moonshot AI's [Attention Residuals paper (arXiv 2603.15031)](https://arxiv.org/abs/2603.15031).

## install

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

## quick start

```sh
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

## training

```sh
# GPT-2 (124M) on OpenWebText — 8x A100, ~4 days, val loss ~2.85
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# GPT-2 (124M) on OpenWebText — 1x A100, ~1 hour, val loss ~3.2-3.4
python train.py config/train_gpt2.py \
    --gradient_accumulation_steps=5 --max_iters=10000 \
    --lr_decay_iters=10000 --warmup_iters=300 --eval_interval=500

# finetune GPT-2 on Shakespeare
python train.py config/finetune_shakespeare.py

# sample
python sample.py --out_dir=out
python sample.py --init_from=gpt2-xl --start="Once upon a time"
```

## Block Attention Residuals

Set `n_attn_res_blocks` to enable AttnRes:

| value | behaviour |
|-------|-----------|
| `0` (default) | standard nanoGPT |
| `1`…`n_layer` | Block AttnRes — N depth blocks, each sublayer attends over past block summaries + current partial residual |
| `2 * n_layer` | Full AttnRes — every sublayer output snapshotted |

The token embedding is always available as `b0`. Constraint: `(2 * n_layer) % n_attn_res_blocks == 0`.

```sh
python train.py config/train_shakespeare_char.py --n_attn_res_blocks=4
```

## metrics

Each eval logs `val/loss` (nats) and `val/bpb` (bits per byte, tokenizer-agnostic) to wandb. Average step time is logged as `dt` (ms).

## GPT-2 baselines (OWT val loss)

| model | params | val loss |
|-------|--------|----------|
| gpt2 | 124M | 3.12 |
| gpt2-medium | 350M | 2.84 |
| gpt2-large | 774M | 2.67 |
| gpt2-xl | 1558M | 2.54 |
