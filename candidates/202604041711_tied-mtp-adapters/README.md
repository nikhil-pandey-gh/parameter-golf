# Tied-Head Low-Rank MTP Adapters on the PR #549 Stack

## Hypothesis

The strongest current family in this repository already has most of the obvious architecture and quantization wins: 11 layers, XSA in the tail, EMA, GPTQ-lite, partial RoPE, VE, Parallel Muon, and legal TTT. The next useful gain is more likely to come from **better token efficiency during training** than from another tiny post-training quantization tweak.

This candidate tests that idea with **multi-token prediction (MTP)**, but in a form that is cheap enough for Parameter Golf: instead of adding one full vocab head per future horizon, it keeps the main tied LM head and adds only a small horizon-specific low-rank adapter before that shared head.

## Why this is promising here

1. The records show repeated additive wins from training-side improvements on the same 11L family, while naive depth recurrence was a consistent dead end.
2. The strongest scripts already contain dormant MTP plumbing, but the logs keep it disabled with `mtp_num_heads:0`, which strongly suggests the idea was considered but never made artifact-efficient enough to turn on.
3. External research argues that MTP improves sample efficiency when trained jointly with the backbone, and that lightweight parameterizations are preferable to bolting large extra heads onto an NTP model.

## Prior repository evidence that influenced this candidate

- **Base implementation**: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best current result and the cleanest combination of 11L/XSA/VE/EMA/GPTQ-lite/Parallel Muon/LeakyReLU^2/legal TTT.
- **Closest non-TTT training/export reference**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Shows the same 11L family is stable and competitive even before TTT.
- **Artifact-budget mindset**: `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - Good reminder that tiny-model wins often come from spending bytes only where they survive export.
- **Negative evidence**:
  - `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` and
  - `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - Both point away from simple layer recurrence as the next step.

There were no prior `candidates/` directories when this candidate was created.

## External research that informed it

- **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737)**  
  MTP can improve sample efficiency when future-token heads are trained jointly on top of a shared trunk.
- **Anastasios Gerontopoulos et al., "Multi-Token Prediction Needs Registers" (arXiv:2505.10518)**  
  The best MTP variants are the ones with negligible extra parameters and minimal architectural disruption.
- **Somesh Mehra et al., "On multi-token prediction for efficient LLM inference" (arXiv:2502.09419)**  
  Retrofitting MTP onto frozen NTP representations is hard; joint training matters, which fits this repo's from-scratch training setup.

## What changed versus the chosen base

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. Replaces the dormant **full-vocab MTP heads** with **`TiedMTPAdapter`** modules.
2. Each horizon adapter is a tiny low-rank residual path (`dim -> rank -> dim`) plus a learned offset, and then reuses the main tied embedding / LM head for logits.
3. Enables MTP by default with:
   - `MTP_NUM_HEADS=2`
   - `MTP_RANK=32`
   - `MTP_LOSS_WEIGHT=0.15`
4. Keeps the adapters **strictly pretraining-only** by dropping them before export, so legal TTT and final evaluation still run on the plain next-token model.
5. Excludes the training-only MTP adapters from export, so the final eval artifact only contains weights needed for normal next-token inference.

## Why this differs from existing records

- The current records already include **disabled** MTP knobs in several 11L scripts, but they leave `mtp_num_heads=0` and pay no training-time MTP cost.
- This candidate is the first repo-local attempt to make MTP **artifact-cheap enough to actually enable**, by tying future-token prediction back to the main LM head instead of allocating separate full-vocab heads.
- It also avoids the repo's known dead end of simple depth recurrence.

## How to run / evaluate

From this candidate directory:

This fork keeps the base script's always-on `EMA(0.997)` behavior and its built-in late-QAT trigger, so the command below only includes knobs that are actually exposed as environment variables in this copied trainer.

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_RANK=32 MTP_LOSS_WEIGHT=0.15 \
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

If you want to isolate the training-side effect first, keep the same command but set `TTT_ENABLED=0`.

## Validation

- `python -m compileall candidates/202604041711_tied-mtp-adapters/train_gpt.py` -> **passed**
- Minimal CPU smoke check was **not feasible in this workspace** because:
  - the installed Python environment here does not provide the runtime dependencies this script imports (`torch`, `sentencepiece`, `flash_attn_interface`), and
  - there are no cached FineWeb shards under `data/datasets/`.

## Main risks / tradeoffs

- **Too little capacity**: tying all horizons back to the main LM head might be too weak versus full independent MTP heads.
- **Step-time tax**: even cheap MTP adds extra logits and cross-entropy work, so the gain must exceed the lost step count.
- **Small-model uncertainty**: the research signal for MTP is strongest at larger scales; a 16MB artifact-constrained model may get a smaller benefit.
- **Export gap**: because the adapters are training-only, the win has to persist in the shared trunk after those adapters are dropped for the final artifact.
