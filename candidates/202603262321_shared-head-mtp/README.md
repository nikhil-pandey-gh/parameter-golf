# Shared-Head MTP on the 11L LeakyReLU² + Legal TTT Stack

## Hypothesis

Add a **train-only shared-head multi-token prediction (MTP) loss** to the strongest current stack so the backbone learns slightly richer future-aware representations **without paying extra artifact bytes at export time**.

The core idea is to predict one extra future target (`t+2`) using the **same tied output projection** that already serves the main next-token loss, rather than training separate future heads that would need their own weights.

## Why this is promising for this repository

This repository's best results come from stacking small, low-risk improvements on top of the strongest 11-layer/XSA/EMA/GPTQ-lite/TTT recipe rather than replacing the architecture wholesale.

A future-token auxiliary objective fits that pattern well:

- it is **training-only**,
- it does **not need new inference infrastructure**,
- it can be made **artifact-free** by reusing the main output head,
- and it targets the one part of the stack that still has room to improve without touching the heavily-optimized quantization/eval path.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Key inherited ideas from the recent winning line:

- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: best current stack, legal score-first TTT, parameter banking + parallel Muon, LeakyReLU(0.5)^2.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: strong pre-TTT base, EMA, GPTQ-lite clip search, 3500-step warmdown.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: partial RoPE + LN scale on the 11L/XSA/EMA backbone.

Repository review also showed that:

- MTP plumbing already existed in the recent scripts,
- but no prior record README documented it as an actual winning ingredient,
- and the existing implementation used **separate zero-initialized future heads**.

## External research that informed the idea

- **DeepSeek-V3 Technical Report** (arXiv:2412.19437): explicitly reports using a **multi-token prediction training objective** for stronger model performance.
  - https://arxiv.org/abs/2412.19437
- **Multi-Token Prediction Needs Registers** (arXiv:2505.10518): argues that MTP can be effective with **negligible additional parameters** and minimal architectural disruption.
  - https://arxiv.org/abs/2505.10518
- **On multi-token prediction for efficient LLM inference** (arXiv:2502.09419): highlights that MTP heads work best when trained jointly with the backbone and that hidden states are strongly specialized for ordinary next-token prediction.
  - https://arxiv.org/abs/2502.09419

My twist for Parameter Golf is to adapt those ideas to the artifact-constrained setting by using a **shared output projection** instead of separate future heads.

## What changed vs the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Shared-head MTP objective**
   - New defaults:
     - `MTP_NUM_HEADS=1`
     - `MTP_SHARED_HEAD=1`
     - `MTP_LOSS_WEIGHT=0.15`
     - `MTP_HORIZON_DECAY=0.5`
   - The auxiliary future-token loss reuses the main tied output projection instead of separate zero-init future heads.

2. **Kept export path artifact-safe**
   - No extra future-head weights are needed for the shared-head path, so the exported model stays aligned with the existing quantization/eval pipeline.

3. **Candidate-directory usability fix**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the candidate script location so `train_gpt.py` can be run from this folder directly.

4. **Lightweight CPU smoke hook**
   - Added a `SMOKE_TEST=1` path and a non-FlashAttention fallback attention kernel for local sanity checks when the training stack is unavailable.

## How to run

From this directory:

```bash
cd candidates/202603262321_shared-head-mtp

SEED=1337 \
BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
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
MTP_NUM_HEADS=1 MTP_SHARED_HEAD=1 MTP_LOSS_WEIGHT=0.15 MTP_HORIZON_DECAY=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

A lightweight local smoke path is available with:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## What to compare first

Compare against the same base command with only the MTP flags toggled:

- pre-TTT `final_int6_roundtrip_exact`
- sliding-window `final_int6_sliding_window_exact`
- legal TTT `legal_ttt_exact`
- training step time / total steps completed under the 600s cap

## Main expected risks and tradeoffs

- **Throughput risk**: even one extra future projection adds training compute and may reduce the number of optimizer steps completed in 600 seconds.
- **Embedding pressure**: because the candidate reuses the tied output projection, the auxiliary loss also pushes directly on the shared embedding/head weights, which could help or hurt quantization robustness.
- **TTT masking**: legal TTT is already strong enough that a small training gain may only be visible in pre-TTT metrics or in tighter seed averages.
- **Objective mismatch risk**: future-token auxiliary losses can regularize the backbone well, but if weighted too heavily they may compromise ordinary next-token specialization.

## Validation performed in this workflow

Successful syntax checks:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603262321_shared-head-mtp/train_gpt.py
```

Outcome:

- both compile passes succeeded.

Attempted smoke check:

```bash
SMOKE_TEST=1 python candidates/202603262321_shared-head-mtp/train_gpt.py
```

Outcome:

- not runnable in this workflow environment because `torch` is not installed here (`ModuleNotFoundError: No module named 'torch'`), even though the repository `requirements.txt` lists `torch`, `numpy`, and `sentencepiece`.
- because of that environment limitation, this candidate was validated with syntax-only checks in CI and includes a dedicated smoke path for use in a properly provisioned training environment.
