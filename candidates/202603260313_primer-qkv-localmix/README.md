# Primer-style causal QKV local mixing on the 11L EMA GPTQ-lite stack

## Hypothesis

This candidate tests whether a **zero-init causal depthwise mixer after each Q/K/V projection** can add a useful short-range inductive bias to the strongest non-TTT stack in the repo, without disturbing its proven training and export path.

The practical twist is that each branch starts as an exact no-op (`x + 0`), so the model begins from the familiar baseline and only learns local projection mixing if the 10-minute budget finds it worthwhile.

## Why it is promising for this repository

- The current winning line already benefits from cheap local biases such as `SmearGate` and `BigramHash`, suggesting that small sequence-local structure is valuable here.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the strongest non-TTT stack and still leaves enough artifact headroom for a small extra module.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` shows that `LeakyReLU(0.5)^2` is a real, nearly-free gain, so this candidate carries that forward too.
- Repo evidence says recurrence / looped depth is a bad fit for the 10-minute budget, so the next architecture idea should be local and cheap rather than effectively deeper.

## Which records influenced it

- Base implementation:
  - `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- Positive neighboring ideas:
  - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- Explicitly avoided dead ends:
  - `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`

There were no prior `candidates/` directories in this repository when this candidate was created.

## External research that informed it

- David So et al., `Primer: Searching for Efficient Transformers for Language Modeling`, arXiv:2109.08668.
  - Motivation used here: squared-ReLU-family MLPs plus depthwise convolution after Q/K/V are strong, code-local transformer upgrades for autoregressive language modeling.
- Maximilian Croci et al., `QuaRot`, arXiv:2404.00456.
  - Motivation used here: quantized model quality is highly sensitive to projection geometry and outlier structure.
- Zechun Liu et al., `SpinQuant`, arXiv:2405.16406.
  - Motivation used here: relatively small changes around projection transformations can materially improve low-bit accuracy.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, this candidate:

- changes the MLP from `relu^2` to `LeakyReLU(0.5)^2`,
- adds optional **causal depthwise Q/K/V mixers** with zero-initialized weights,
- enables those mixers by default for this candidate with:
  - `QKV_CONV_ENABLED=1`
  - `QKV_CONV_KERNEL=3`
- adds a **FlashAttention fallback** to PyTorch SDPA, including manual KV-head expansion on non-CUDA paths so GQA configurations remain importable for CPU smoke tests when `flash_attn_interface` is unavailable.

Everything else is intentionally kept close to the base:

- 11 layers, width 512, GQA (8 heads / 4 KV heads)
- EMA + GPTQ-lite mixed int6 export
- XSA on the deepest 4 layers
- partial RoPE, LN scaling, SmearGate, BigramHash, shared value embedding
- sliding-window evaluation support

## How to run or evaluate it

From this candidate directory:

```bash
SEED=1337 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
QKV_CONV_ENABLED=1 \
QKV_CONV_KERNEL=3 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a quick ablation, disable the new branch with `QKV_CONV_ENABLED=0`.

## Main expected risks or tradeoffs

- The new branch may overlap with `SmearGate` and `BigramHash`, adding little new signal.
- Even small convolutions add some step-time overhead; fewer training steps could erase the gain.
- The extra parameters should still fit under the 16MB total artifact cap, but reduce headroom relative to the base run.
- The zero-init residual design is safer than a full replacement, but it may also make the new branch slow to turn on in a short run.

## Validation

Commands run during creation of this candidate:

```bash
python -m compileall candidates/202603260313_primer-qkv-localmix/train_gpt.py
python3 - <<'PY'
import sys
try:
    import torch
    print("torch", torch.__version__)
except Exception as exc:
    print("torch_import_failed", repr(exc))
PY
```

Observed outcome:

- `compileall` passed.
- A CPU forward smoke test was prepared but **not feasible on this runner** because the available Python interpreters do not have `torch` installed, so the module cannot be imported end-to-end here without extra infrastructure.
