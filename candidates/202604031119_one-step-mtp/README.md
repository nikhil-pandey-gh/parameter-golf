# Candidate: One-Step MTP on the 11L Banked Stack

## Hypothesis

Adding a **single training-only multi-token-prediction (MTP) head** to the current strongest 11-layer banked stack should improve sample efficiency enough to lower final validation bits-per-byte, while keeping the scored artifact size unchanged because the extra head is dropped before export.

## Why this looks promising here

The repo's strongest results already come from improvements that help the trunk **during training** without materially increasing artifact bytes:

- sliding-window evaluation,
- quantization-aware capacity reallocation into a larger 11-layer / 3x-MLP trunk,
- partial late-layer architectural tweaks like XSA and partial RoPE,
- smoothing and quantization polish such as EMA and GPTQ-lite.

This candidate follows that pattern. It changes the **training objective**, not the exported model family.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`: strongest current stack and code base for this candidate.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`: strongest clean training-only baseline for the 11-layer family.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`: evidence that small late architectural changes on this trunk can still matter.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`: the first strong 11-layer XSA/EMA/int6 recipe.

I also reviewed the full `records/` tree and found **no existing `candidates/` directory** in this repository, so this is the first candidate snapshot under `candidates/`.

## External research that motivated it

- **Better & Faster Large Language Models via Multi-token Prediction** (arXiv:2404.19737) argues that auxiliary future-token heads improve sample efficiency on a shared trunk with no inference-time requirement to keep those heads.
- **Self-Distillation for Multi-Token Prediction** (arXiv:2603.23911) reports that training many MTP heads jointly is harder than it looks, which motivates the conservative choice here: **one extra head first**, not an aggressive multi-head rollout.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. `MTP_NUM_HEADS` now defaults to `1` instead of `0`.
2. `MTP_LOSS_WEIGHT` now defaults to `0.1` instead of `0.2`, keeping the auxiliary objective intentionally modest.
3. The export path is left intact and explicitly documented: `mtp_heads` are excluded from the final artifact on purpose.

Everything else is inherited from the current best stack: 11 banked layers, LeakyReLU(0.5)^2 MLP, XSA on the last 4 layers, partial RoPE, VE on late layers, EMA/SWA, GPTQ-lite-style int6 export, and optional legal TTT.

Note: this inherited script keeps **EMA active with a fixed decay of 0.997**; this candidate does not change that part of the base recipe.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 \
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

If you want to isolate the **training-only** effect first, leave `TTT_ENABLED=0`.

## Evaluation notes

- The scored artifact should not grow from the MTP head, because `mtp_heads` are excluded before serialization.
- The main comparison to watch is whether the extra auxiliary loss lowers **pre-TTT** sliding-window `val_bpb` enough to justify the extra train-time compute.
- The strongest fair comparison is against the same stack with `MTP_NUM_HEADS=0`.

## Main risks / tradeoffs

- The extra head increases train-time compute, so any sample-efficiency gain must beat the loss in step count.
- MTP gains reported in the literature may transfer less strongly to this tiny-model, low-vocab, compression-scored setting than they do to standard perplexity benchmarks.
- Even one auxiliary head may slightly bias the trunk away from the exact next-token objective we ultimately score.
- The interaction between MTP and legal TTT is unknown; the cleanest first ablation is with TTT disabled.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604031119_one-step-mtp/train_gpt.py
```

Outcome:

- `compileall`: passed
- CPU startup smoke test: not feasible on this runner

Expected limitation for smoke testing in this environment:

- this runner currently lacks the Python dependencies used by the training script (`torch`, `numpy`, `sentencepiece`), and it also does not contain a prepared local FineWeb/tokenizer checkout, so a real startup run may not be feasible here even with tiny settings.
