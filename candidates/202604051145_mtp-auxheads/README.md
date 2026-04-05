# MTP auxiliary head on the 2026-03-23 stack

## Hypothesis

Add a small **training-only multi-token prediction (MTP)** auxiliary loss to the current best public stack so the shared transformer trunk learns faster in the same 600 second budget, then drop the auxiliary head at export time so the artifact budget stays essentially unchanged.

## Why this is promising here

The repo's biggest recent wins have already harvested most of the obvious evaluation and compression gains: sliding evaluation, FP16/int8 embedding handling, mixed/int6 quantization, EMA/SWA, XSA, partial RoPE, GPTQ-lite clipping, and legal TTT. That makes **sample efficiency during training** the most attractive remaining lever on top of the current SOTA recipe.

The strongest nearby evidence is:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the best current record and already carries dormant MTP scaffolding in code.
- Earlier 11-layer records steadily improved by stacking low-overhead tricks rather than making large architecture jumps.
- Repo negatives around layer recurrence/depth reuse suggest avoiding heavier architectural rewrites for the next candidate.

## Prior records that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Relevant predecessor stack:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **MTP wiring reference before the parameter-banking refactor:** `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
- **Negative guidance:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` and `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` both report depth reuse / recurrence as a poor fit for a strict wall-clock budget.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-Token Prediction"** (arXiv:2404.19737): trains a shared trunk with multiple future-token heads and reports better sample efficiency with no inference-time requirement to keep the extra heads.

That paper is a particularly good fit for this repo because the challenge is training-time constrained and export-size constrained at the same time.

## What changed vs the chosen base implementation

Starting from the 2026-03-23 record script, this candidate:

1. **Enables one auxiliary MTP head by default** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15`.
2. **Puts the MTP head weights back under Muon optimization.** The parameter-banking refactor in the 2026-03-23 script kept the MTP loss path but no longer optimized the auxiliary head weights, so the feature was effectively inert.
3. **Sets `BIGRAM_VOCAB_SIZE=1536` by default** to match the actual 2026-03-23 record recipe.
4. **Disables late-QAT by default** (`LATE_QAT_THRESHOLD=0.0`) so this candidate cleanly measures the MTP effect instead of relying on an inherited path that is not central to the hypothesis.

The export path still strips `mtp_heads.*`, so the auxiliary head remains training-only and does not materially change submission size. This is **not** a perfectly single-knob ablation against the published 2026-03-23 run command, because this candidate also turns late-QAT off by default; that was intentional because the inherited late-QAT path is not the focus of this experiment and is a poor match for the banked-weight stack.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.0 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
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

## Evaluation notes

- The intended comparison is against the 2026-03-23 record stack with the same TTT settings, plus the same banked 11-layer architecture, but with MTP left off.
- This candidate also sets `LATE_QAT_THRESHOLD=0.0` on purpose, so any comparison should treat that as part of the recipe instead of attributing the entire delta to MTP alone.
- Because the auxiliary head is excluded from export, any gain should come from better trunk training rather than extra artifact bytes.

## Main risks / tradeoffs

- **Step-time overhead:** even one extra vocab projection adds training work, so the MTP gain has to beat the loss in total steps.
- **Small-model uncertainty:** the MTP paper shows stronger gains as models scale; a 512d model may realize a smaller effect.
- **Optimizer sensitivity:** auxiliary heads optimized with Muon may want a different loss weight or learning-rate balance than the inherited defaults.
- **Attribution ambiguity:** if this candidate moves the score, the next step should still check whether `MTP_NUM_HEADS=1` is best versus `2`, or whether the loss should start later in training.

## Validation

Validation commands and outcomes:

- `python -m compileall candidates/202604051145_mtp-auxheads/train_gpt.py` — **passed**
- `python candidates/202604051145_mtp-auxheads/train_gpt.py` — **did not reach runtime smoke** in this runner because the environment is missing repo Python dependencies (`ModuleNotFoundError: No module named 'numpy'`) before training setup begins
