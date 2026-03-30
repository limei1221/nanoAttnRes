# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanoGPT: a minimalist repository for training and finetuning medium-sized GPT language models. The goal is reproducible GPT-2 training with clean, readable code. Note: the README mentions this repo is being deprecated in favor of a newer project called "nanochat."

## Key Commands

### Install Dependencies
No requirements.txt — install manually:
```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### Prepare Data
```bash
python data/shakespeare_char/prepare.py   # character-level Shakespeare (~seconds)
python data/shakespeare/prepare.py        # BPE Shakespeare (for GPT-2 finetuning)
python data/openwebtext/prepare.py        # OpenWebText (54GB, ~hours)
```

### Train
```bash
# Quick demo: character-level Shakespeare (~3 min on A100, slower on CPU/MPS)
python train.py config/train_shakespeare_char.py

# CPU/MPS training with reduced config
python train.py config/train_shakespeare_char.py --device=cpu --compile=False \
  --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 \
  --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0

# GPT-2 reproduction (8xA100 required)
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# Finetune GPT-2-XL on Shakespeare
python train.py config/finetune_shakespeare.py
```

### Sample / Inference
```bash
python sample.py --out_dir=out-shakespeare-char
python sample.py --init_from=gpt2-xl --num_samples=5 --max_new_tokens=100
python sample.py --init_from=gpt2-xl --start="What is the meaning of life?"
python sample.py --init_from=gpt2-xl --start=FILE:prompt.txt
```

### Evaluate Pretrained GPT-2
```bash
python train.py config/eval_gpt2.py        # 124M
python train.py config/eval_gpt2_medium.py # 350M
python train.py config/eval_gpt2_large.py  # 774M
python train.py config/eval_gpt2_xl.py     # 1558M
```

### Benchmark
```bash
python bench.py                   # timing benchmark
python bench.py --profile=True    # PyTorch profiler output
```

### Comparison Experiments (run.sh)

`run.sh` auto-detects GPU and runs baseline vs Block AttnRes vs Full AttnRes:
```bash
bash run.sh
```
- **No GPU**: Shakespeare char, CPU, 3 runs (~minutes each)
- **GPU**: OpenWebText GPT-2 124M, 1×A100, 3 runs (~1 hour each)

All runs log to wandb by default (`wandb_log=True` is set in the script).

## Configuration System

`configurator.py` implements a "poor man's configurator": scripts define defaults as module-level variables, then `configurator.py` is exec'd to override them via config files or `--key=value` CLI flags. This means:

- Config files (e.g., `config/train_shakespeare_char.py`) are just Python files that set variables.
- CLI overrides come after the config file: `python train.py config/foo.py --batch_size=16`
- Type enforcement: overriding `batch_size=12` with `--batch_size=16.0` will fail (int vs float).

## Block AttnRes

Set `n_attn_res_blocks` in `GPTConfig` (or as a CLI/config-file override) to enable the Block Attention Residuals mechanism from Moonshot AI's "Attention Residuals" paper (arXiv 2603.15031). `0` (default) gives standard nanoGPT.

```python
# config/train_shakespeare_char.py
n_attn_res_blocks = 4  # must satisfy: (2 * n_layer) % n_attn_res_blocks == 0
```

**Variants:**
- **Block AttnRes**: `n_attn_res_blocks` divides `2 * n_layer` evenly; e.g. `n_attn_res_blocks=4` with `n_layer=12` gives 4 blocks of 6 sublayers each.
- **Full AttnRes**: `n_attn_res_blocks = 2 * n_layer` — snapshot at every sublayer boundary.

**What it does.** Replaces the standard `x = x + sublayer(LN(x))` with depth-wise attention. Before each sublayer (attention and MLP), a learned pseudo-query vector scores all *past block summaries* plus the *current partial residual* via softmax; the model uses that weighted combination as the sublayer's input instead of raw `x`. At block boundaries, the partial residual is snapshotted into the `blocks` list and reset to zero, bounding hidden-state magnitude growth.

**New components per `Block`** (only when `n_attn_res_blocks > 0`):
- `attn_res_proj` / `mlp_res_proj`: `Linear(n_embd, 1, bias=False)` — the pseudo-query; **zero-initialized**
- `attn_res_norm` / `mlp_res_norm`: `RMSNorm(n_embd)` — normalises the stacked values to form keys

**Boundary schedule** (sublayer-level, 0-indexed):
- Each transformer layer has 2 sublayers (attn=even, mlp=odd). Total sublayers = `2 * n_layer`.
- `sublayers_per_block = (2 * n_layer) // n_attn_res_blocks`
- A snapshot fires at sublayer `s` where `s % sublayers_per_block == 0`
- Example: `n_layer=6, n_attn_res_blocks=3` → `sublayers_per_block=4`, snapshots at sublayer 0 (layer 0 attn), 4 (layer 2 attn), 8 (layer 4 attn)

**Gradient behaviour at init.** Pseudo-query weights start at zero → all logits = 0 → softmax is uniform. When the stack has only 1 item (early layers where `blocks` is still empty), the Jacobian of softmax is zero, so those weights receive no gradient until a second item enters the stack. This is expected and correct.

## Architecture

### Model (`model.py`)

The GPT model hierarchy:
```
GPT
└── transformer
    ├── wte: Token Embedding (vocab_size → n_embd)
    ├── wpe: Position Embedding (block_size → n_embd)
    ├── h: ModuleList[Block]
    │   └── Block: LayerNorm → CausalSelfAttention → LayerNorm → MLP
    └── ln_f: Final LayerNorm
└── lm_head: Linear (n_embd → vocab_size), weight-tied with wte
```

Key design choices:
- **Flash Attention**: Used automatically when PyTorch ≥ 2.0; falls back to manual causal mask otherwise.
- **No bias** in attention/MLP layers (slightly faster).
- **Weight tying**: `lm_head.weight == transformer.wte.weight` (saves ~38M params for GPT-2).
- **Scaled init**: residual projection weights initialized at `1/sqrt(2 * n_layer)` (per GPT-2 paper).
- `GPT.from_pretrained(model_type)` loads OpenAI GPT-2 weights from HuggingFace.
- `GPT.configure_optimizers()` splits params into weight-decay and no-decay groups (no decay on biases, layernorm, embeddings).

### Training (`train.py`)

- **Data loading**: numpy `memmap` on binary `.bin` files — no RAM overhead.
- **LR schedule**: linear warmup (`warmup_iters`) → cosine decay to `min_lr` at `lr_decay_iters` → constant `min_lr`.
- **Gradient accumulation**: `gradient_accumulation_steps` micro-steps before an optimizer update.
- **DDP**: wraps model with `DistributedDataParallel`; syncs gradients only at the final accumulation step (`no_sync` context otherwise).
- **Mixed precision**: `bfloat16` preferred on Ampere+; `float16` with `GradScaler` on older hardware.
- **Checkpointing**: saves to `out_dir/ckpt.pt` on best val loss; resumes with `init_from='resume'`.
- **`val_bpb`**: bits-per-byte metric computed alongside val loss and logged to wandb as `val/bpb`. Requires `tiktoken` for BPE datasets or `meta.pkl` with `itos` for char-level datasets; silently skipped otherwise.
- **Compilation**: `torch.compile(model)` is on by default (`compile=True`); disable with `--compile=False` on older PyTorch or for debugging.

### Data Pipeline

Data scripts tokenize source text and write `train.bin` / `val.bin` as flat `uint16` arrays (numpy). At training time, random chunks of `block_size` tokens are read directly from the memmap files.

Character-level datasets also produce `meta.pkl` with `stoi`/`itos` dicts; `sample.py` picks this up automatically when `out_dir` contains a trained checkpoint.
