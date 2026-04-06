# LSQ-lite Bank QAT on the LeakyReLU² + Legal TTT stack

## Hypothesis

The current best stack already trains strong full-precision banked attention/MLP weights and exports them with GPTQ-lite int6 quantization, but the training-time fake-quant path does not meaningfully pressure those banked weights. This candidate applies late-stage **bank-aware LSQ-lite QAT** to the actual serialized attention/MLP banks, with small learned clip multipliers that are also offered back to the export-time GPTQ-lite search.

## Why this is promising here

- The strongest recent record is built around banked attention/MLP weights, LeakyReLU(0.5)^2, legal score-first TTT, EMA, partial RoPE, XSA, and GPTQ-lite int6 export.
- Recent winning records kept gaining from better post-training quantization and averaging, which suggests the remaining gap is still partly in the low-bit export path rather than the core architecture alone.
- This candidate keeps the proven architecture almost unchanged and instead makes the model see **its real int6 bottleneck** during late warmdown.

## Repository evidence that influenced this candidate

- **Root baseline**: `train_gpt.py` establishes the repository's fp32-master-weight + post-training quantization pattern and the general training/eval structure.
- **Primary base**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` supplied the exact starting stack: parameter banks, LeakyReLU² MLP, legal TTT, lzma-compressed int6 export, and parallel Muon.
- **Quantization influence**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed that better clip selection and EMA still move the score on essentially the same architecture family.
- **Architecture influence**: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` reinforced partial RoPE + LN-scale as stable zero-parameter wins.
- **Prior candidates**: there was no existing `candidates/` directory in this checkout, so there were no prior candidate iterations to avoid or build on directly.

## External research that informed it

- **Learned Step Size Quantization (LSQ)** — https://arxiv.org/abs/1902.08153  
  Motivated learning the quantizer scale instead of treating it as fixed.
- **PACT: Parameterized Clipping Activation** — https://arxiv.org/abs/1805.06085  
  Motivated learning clip thresholds instead of relying only on hardcoded clipping heuristics.
- **QuaRot** — https://arxiv.org/abs/2404.00456  
  Reinforced that reducing outlier sensitivity is a strong route to better low-bit performance.
- **SpinQuant** — https://arxiv.org/abs/2405.16406  
  Reinforced the same lesson at higher quality: quantization quality improves when the model is nudged toward quantization-friendly geometry instead of only fixing export code.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps the model stack and training recipe, but changes the late-QAT path:

1. Adds **bank-level QAT scale parameters** for `qo_bank`, `kv_bank`, `mlp_up_bank`, and `mlp_down_bank`.
2. Adds an **LSQ-lite fake int6 quantizer** that runs on those banked weights during late training.
3. Enables late QAT by **recompiling after the switch**, so the bank-aware QAT path is actually traced instead of being left out of the compiled graph.
4. Feeds the learned clip multipliers back into export-time `quantize_int6_per_row(...)` as an extra GPTQ-lite candidate, instead of discarding what late QAT learned.

## How to run

From the repository root:

```bash
cd candidates/202604060138_lsq-lite-bank-qat
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

EMA remains part of the candidate, but in this script it is fixed internally rather than configured through `EMA_ENABLED` / `EMA_DECAY` environment variables.

## Validation recorded for this candidate

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604060138_lsq-lite-bank-qat/train_gpt.py` | Passed |
| `python - <<'PY' ... import torch ... PY` | Failed because `torch` is not installed in this container |

A minimal runtime smoke test was **not feasible here** because:

1. the container does not currently have `torch` installed,
2. the local checkout does not contain cached FineWeb shard files under `data/datasets/...`,
3. the script hard-requires CUDA and `flash_attn_interface`, so a CPU-only start test would not represent the real runtime path anyway.

## Main risks and tradeoffs

- Late bank-QAT adds extra work exactly when the run is closest to the wallclock cap.
- Learned clip multipliers may over-clip or collapse toward conservative values if the warmdown window is too short.
- The candidate is intentionally narrow: it tries to improve the export match on the existing best stack rather than searching for a new macro-architecture.
