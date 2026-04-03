# AWQ-lite weighted GPTQ export

## Hypothesis

The current frontier in this repo is limited less by pre-quant training loss than by the **int6 export gap**. If the export pass chooses row-wise clip values using **activation-weighted reconstruction error** instead of plain weight MSE, the compressed model should preserve more of the post-EMA model's useful signal at essentially zero artifact cost.

## Why this is promising for this repository

- The strongest pure training/export base here is the 11-layer EMA + GPTQ-lite stack from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`.
- The current best overall record (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`) still wins partly through better evaluation-time adaptation, which reinforces that the remaining headroom is in better use of existing model capacity rather than a wholesale architecture change.
- The non-record long-run experiment `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` also showed that extra training alone did not erase the post-quantization penalty.

## Prior records that influenced this candidate

- **Primary base**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - keeps the strongest clean 11-layer stack: XSA, partial RoPE, LN scaling, VE, EMA, tight SWA, sliding-window eval, and GPTQ-lite export.
- **Supporting evidence**: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms the value of the underlying 11-layer architecture and documents that late QAT can be fragile under `torch.compile`.
- **Current best overall**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - useful as a frontier reference, but I intentionally did not inherit its TTT and parallel-optimizer complexity for this candidate.

## External research

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** — https://arxiv.org/abs/2306.00978
  - motivates using activation statistics, not just raw weight error, when deciding what quantization distortion matters.
- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** — https://arxiv.org/abs/2210.17323
  - motivates one-shot, reconstruction-aware export rather than naive max-abs clipping.
- **EfficientQAT: Efficient Quantization-Aware Training for Large Language Models** — https://arxiv.org/abs/2407.11062
  - reinforces that small improvements in quantization-aware objectives can be worthwhile even when the base model is already strong.

This candidate implements the smallest repo-friendly step in that direction: **AWQ-lite**, i.e. activation-weighted clip selection layered on top of the existing GPTQ-lite row-wise clip search.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Adds three export/calibration flags:
   - `AWQ_LITE_ENABLED=1`
   - `AWQ_CALIBRATION_BATCHES=8`
   - `AWQ_CALIBRATION_TOKENS=131072` (global tokens per calibration batch; keep it divisible by `WORLD_SIZE * TRAIN_SEQ_LEN`)
2. Runs a short post-training calibration pass over a few training batches using the EMA-averaged model.
3. Collects per-`CastedLinear` input RMS statistics with forward pre-hooks.
4. Changes int6 clip selection so each candidate percentile is scored by **activation-weighted squared reconstruction error** instead of unweighted weight MSE.
5. Leaves **late-QAT post-compile toggling disabled by default** (`LATE_QAT_THRESHOLD=0`) because this code family already showed that `torch.compile` can constant-fold that path away.
6. Falls back to the original GPTQ-lite behavior automatically when AWQ-lite is disabled or no matching activation stats are available.

The training architecture, optimizer split, EMA/SWA logic, and evaluation path are otherwise unchanged from the base record.
EMA stays enabled with the inherited fixed decay of `0.997`.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202604031146_awq-lite-gptq
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_LITE_ENABLED=1 AWQ_CALIBRATION_BATCHES=8 AWQ_CALIBRATION_TOKENS=131072 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks or tradeoffs

- The calibration pass adds a small amount of export-time compute after training.
- This is **not** full AWQ channel rescaling; it is a lightweight approximation that only changes clip selection.
- Late-QAT toggling is intentionally disabled here; if always-on QAT is desired, it should be enabled up front with `QAT_ENABLED=1` before `torch.compile`.
- Calibration statistics come from a tiny training-slice sample, so the best batch count and token budget may need tuning.
- `AWQ_CALIBRATION_TOKENS` must stay divisible by `WORLD_SIZE * TRAIN_SEQ_LEN`, because calibration reuses the regular sequence loader.
- If the activation weighting is too aggressive, it could over-protect a few high-energy channels and slightly hurt overall compression quality.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604031146_awq-lite-gptq/train_gpt.py
```

Outcome:

- `compileall` succeeded.
- A CPU import smoke test was **not feasible** here because this runner does not have the repo's Python dependencies installed, and full execution of this script also requires CUDA plus `flash_attn_interface`.
