# Forward-Curriculum MTP + LeakyReLU^2

## Hypothesis

A **forward-curriculum multi-token prediction (MTP)** objective can improve sample efficiency for this repository's small 11-layer model stack, as long as the extra horizons are introduced gradually instead of from step 0. To keep the candidate aligned with the strongest repo evidence, this version also ports the separately validated `LeakyReLU(0.5)^2` MLP activation from the current top record.

## Why this is promising for this repository

Recent winning records mostly improved the challenge through architecture shaping, quantization/export, EMA/SWA, and evaluation tricks. The training objective itself is much less explored, which leaves room for a new idea that does not require new infrastructure.

This candidate is intentionally lower-risk than looped/shared-depth experiments. External research suggests parameter sharing is promising in general, but this repository already has negative recurrence evidence in non-record sweeps, while recent MTP research specifically argues that **small models can benefit when MTP is introduced via a curriculum** rather than applied naively from the start.

## Which records or prior experiments influenced it

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen as the base implementation because it is the strongest clean non-TTT stack in the repo: 11 layers, XSA4, partial RoPE, LN scale, EMA, tight SWA, BigramHash, SmearGate, shared value embeddings, and GPTQ-lite int6 export.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Contributed the `LeakyReLU(0.5)^2` activation, which that record README reports as a meaningful standalone gain.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Reinforced that the repo already benefits from low-byte refinements like partial RoPE and layer scaling, which this candidate preserves.

## Which external research informed it

- Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (arXiv:2404.19737)
  - Shows that predicting multiple future tokens can improve sample efficiency and downstream capability.
- Ansar Aynetdinov and Alan Akbik, *Pre-Training Curriculum for Multi-Token Prediction in Language Models* (arXiv:2505.22757)
  - Most relevant source for this repo: it argues that **small language models struggle with raw MTP**, and that a **forward curriculum** helps them benefit from it.
- Anastasios Gerontopoulos et al., *Multi-Token Prediction Needs Registers* (arXiv:2505.10518)
  - Motivated keeping the change lightweight and training-only instead of rewriting the inference-time architecture.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Replaces the base `relu^2` MLP with `LeakyReLU(0.5)^2`.
2. Activates the base file's dormant **extra-head MTP path**, but gates it with a **forward curriculum** instead of applying it uniformly from the beginning.
3. Adds time-budget-aware curriculum weights controlled by:
   - `MTP_HORIZONS` (default `2`)
   - `MTP_MAX_LOSS_WEIGHT` (default `0.15`)
   - `MTP_START_FRAC` (default `0.30`)
   - `MTP_END_FRAC` (default `0.80`)
4. Uses the script's existing **training-time budget accounting** to ramp the auxiliary objective in over the run, instead of turning it on immediately.
5. Excludes `mtp_heads` from export so the **submission artifact stays aligned with the inference-time model**, even though the auxiliary heads exist during training.

## How to run or evaluate it

Example training command:

```bash
RUN_ID=curriculum_mtp \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_HORIZONS=2 MTP_MAX_LOSS_WEIGHT=0.15 MTP_START_FRAC=0.30 MTP_END_FRAC=0.80 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Run from inside this candidate directory.

## Validation

Commands run while preparing this candidate:

```bash
python -m compileall candidates/202603250826_curriculum-mtp/train_gpt.py
python -c "import importlib.util; print(importlib.util.find_spec('torch'))"
```

Observed outcomes:

- `compileall` succeeded.
- A CPU runtime smoke test was **not feasible in this workflow environment** because `torch` is not installed here (`find_spec('torch') -> None`), so the training module cannot be imported locally.

## Main expected risks or tradeoffs

- Even with a curriculum, small models may still dislike MTP if the onset is too early or the auxiliary weight is too high.
- Training-only MTP heads are excluded from export, so this is a pure optimization aid rather than a deploy-time architectural change; if the curriculum is not helping optimization enough, the extra training compute will not pay for itself.
- The interaction with EMA, late QAT, and sliding-window evaluation is plausible but unproven in this repo until a real multi-seed GPU run is performed.
