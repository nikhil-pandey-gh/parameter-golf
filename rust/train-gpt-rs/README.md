# train-gpt-rs

Experimental Rust reproduction scaffold for the repository's root `train_gpt.py`.

## What this project does

- Mirrors the Python trainer's configuration surface through environment variables.
- Implements the same high-level GPT architecture in Rust:
  - grouped-query causal attention
  - RoPE
  - stateless RMSNorm
  - ReLU^2 MLP
  - learned `attn_scale`, `mlp_scale`, `resid_mix`, and `skip_weights`
  - optional tied embeddings and tanh logit softcap
- Reproduces the binary shard loader and the SentencePiece BPB byte-accounting logic.
- Preserves the Python optimizer split conceptually with AdamW for token/scalar/head parameters and a Rust Muon implementation for matrix parameters.
- Exports a raw `safetensors` checkpoint plus an int8 + zlib-compressed roundtrip artifact.

## Why this is only a scaffold

This crate is intentionally CPU-validatable first. It is not claiming leaderboard-ready 8xH100 parity with the PyTorch baseline yet.

Key maturity gaps relative to `train_gpt.py`:

1. `train_gpt.py` depends on CUDA bf16, NCCL DDP, fused Adam, and PyTorch's production attention/runtime stack.
2. In Candle today, the fused SDPA path is not a drop-in training replacement for this workload, so this crate uses a differentiable manual attention path instead.
3. That keeps the implementation honest on CPU, but it is not a credible high-performance 8xH100 training path by itself.
4. The most viable future Rust route is still a libtorch-backed or mixed Rust/PyTorch design, not a pure Rust promise of immediate multi-GPU parity.

## Commands

Smoke test on CPU:

```bash
cd rust/train-gpt-rs
cargo run -- smoke
```

Try a tiny training-style run against real shards if they exist locally:

```bash
cd rust/train-gpt-rs
ITERATIONS=1 TRAIN_BATCH_TOKENS=8192 TRAIN_SEQ_LEN=128 cargo run -- train
```

## Optional GPU-oriented features

This crate exposes Cargo features that match the intended future direction even though the current implementation is only validated on CPU here:

- `--features cuda`
- `--features cuda,nccl`

Those features only enable backend crates; they do **not** by themselves provide a production-ready distributed 8xH100 training stack.
