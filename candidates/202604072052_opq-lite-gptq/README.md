# OPQ-lite GPTQ on the LeakyReLU2 + TTT stack

## Hypothesis

The current best stack already squeezes a lot out of training, evaluation, and clip-search quantization. The next cheap win is likely to come from **preserving a tiny mixed-precision set of weight outliers before int6 export**, so the GPTQ-lite percentile search spends its dynamic range on the bulk of the matrix instead of a few extreme values.

## Why this is promising here

- The repo history repeatedly shows that **quantization quality is a first-order bottleneck**, not a cleanup step:
  - `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` explicitly argues that post-training quantization penalty is larger than many architecture gains.
  - `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` gets a real win from better clip search alone.
- The top record, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, already has the strongest known training/eval stack in this repo, so a **surgical export improvement** is a better bet than replacing the model again.
- Unlike learned rotations or new quantizers, sparse outlier preservation fits the existing script with minimal infrastructure change.

## Prior experiments that influenced this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
  - Best overall stack in the repo: LeakyReLU(0.5)^2, Parameter Banking + Parallel Muon, partial RoPE, XSA, EMA/SWA, legal score-first TTT, GPTQ-lite int6 export.
- **Direct quantization precursor:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - Establishes that better per-row clipping is already worth about `-0.0006 BPB`.
- **Compression bottleneck evidence:** `records/track_10min_16mb/2026-03-19_WarmdownQuantization`
  - Useful reminder that many wins in this challenge come from better export behavior, not just lower pre-quant loss.
- **Negative control:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
  - Useful mainly because it shows recurrence/weight reuse was bad under fixed wall-clock, so this candidate avoids large structural changes.

## External research that informed it

- **LLM.int8()** — Dettmers et al., arXiv:2208.07339  
  Mixed-precision decomposition isolates emergent outlier features in higher precision instead of forcing uniform low-bit quantization to absorb them.
- **BOF4 / OPQ** — Blumenberg et al., arXiv:2505.06653  
  Explicitly reports that **outlier-preserving quantization**, which stores outlier weights in 16-bit precision while quantizing the rest, improves low-bit LLM weight reconstruction.
- **QuaRot** — Ashkboos et al., arXiv:2404.00456  
  Reinforces the broader point that outliers are a major cause of low-bit degradation.
- **SpinQuant** — Liu et al., arXiv:2405.16406  
  Further evidence that outlier mitigation is worth real accuracy even when the underlying model is unchanged.

The candidate does **not** implement rotations or calibration-heavy optimization. It takes the lower-infrastructure lesson from those papers: outliers matter enough to treat separately.

## What changed versus the chosen base

Only the export path changed.

1. Added `INT6_OUTLIER_TOPK` (default `96`).
2. Before GPTQ-lite percentile search on each large int6 matrix, the script now:
   - finds the top-`k` largest-magnitude weights in that matrix,
   - stores those values exactly in fp16 with sparse row/column indices,
   - zeros them out in the dense tensor,
   - runs the existing GPTQ-lite per-row clip search on the remainder.
3. During round-trip load, those sparse fp16 values are added back after dense dequantization.

This is intentionally an **OPQ-lite** variant:

- no new training logic,
- no activation quantization,
- no learned rotations,
- bounded metadata cost per matrix.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
INT6_OUTLIER_TOPK=96 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`INT6_OUTLIER_TOPK=0` disables the new path and recovers the base export behavior.

## Main expected risks and tradeoffs

- **Artifact-size risk:** sparse fp16 corrections improve quantization error but spend bytes; the best `topk` may be smaller than 96.
- **Late-stage interaction risk:** the best record already uses TTT, so some export improvements may be partially hidden by downstream adaptation.
- **Layer sensitivity:** the same top-`k` budget may not be optimal for attention, MLP-up, and MLP-down matrices.
- **Diminishing returns:** the base already has GPTQ-lite clip search, so this may deliver a small win rather than a dramatic one.

## Validation

Commands run in this repository:

```bash
python -m compileall candidates/202604072052_opq-lite-gptq/train_gpt.py
```

Outcome:

- `compileall` succeeded.
- A CPU-only runtime smoke test was **not feasible** here because this script is written for the CUDA/FlashAttention training path and expects real tokenizer and FineWeb shard inputs before it can execute end-to-end.
