# AWQ-lite salient-row export on the 11L EMA/GPTQ-lite stack

## Hypothesis

The strongest remaining gain is likely in the **export path**, not another broad architecture rewrite. This repository's best runs already stack 11 layers, partial RoPE, XSA, EMA/SWA, and GPTQ-lite; the recurring remaining pain point is still quantization sensitivity. An **AWQ-inspired mixed-precision escape hatch** that preserves only the most activation-salient rows in int8 while keeping the rest of the attention/MLP weights in GPTQ-lite int6 should reduce roundtrip loss at modest byte cost.

I also carry forward **LeakyReLU(0.5)^2** because the newest record showed it was the cleanest recent training-side win.

## Why this is promising here

- Prior records repeatedly improved by attacking the export gap: fp16 tied embeddings, mixed int6/int8 export, GPTQ-lite clipping, longer warmdown, and EMA.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the strongest stable **pre-TTT** stack to build from.
- AWQ's core idea matches the repo's constraints: protect a tiny subset of salient weights instead of paying higher precision everywhere.

## Prior records that influenced this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Activation carry-forward:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Partial RoPE + LN scale stack:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Quantization sensitivity evidence:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`

## External research that informed it

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** — <https://arxiv.org/abs/2306.00978>
- **QLoRA** (NF4 + double-quantized scales) — <https://arxiv.org/abs/2305.14314>
- **Primer: Searching for Efficient Transformers for Language Modeling** — <https://arxiv.org/abs/2109.08668>

AWQ is the main research driver here. QLoRA is relevant because it reinforces the value of **tiny mixed-precision escape hatches** over uniform low-bit export. Primer is relevant background for squared-activation families; the repo's newest record then validates the leaky squared variant in this exact setting.

## What changed vs the chosen base implementation

1. **LeakyReLU(0.5)^2 MLP**
   - `MLP.forward()` now uses `F.leaky_relu(..., negative_slope=0.5).square()` instead of plain ReLU^2.
2. **AWQ-lite calibration pass**
   - After EMA is applied, the script runs a short calibration pass on training batches and records mean absolute output activation per `CastedLinear` row.
3. **Salient-row mixed-precision export**
   - Attention and MLP weights are still exported with GPTQ-lite int6 by default.
   - For each large eligible matrix, the most activation-salient rows are stored as **int8 overrides** and spliced back in during dequantization.
4. **Focused defaults**
   - The inherited late-QAT path is left off by default (`LATE_QAT_THRESHOLD=0`) so this candidate stays focused on export-time activation-aware quantization.

## How to run / evaluate

```bash
cd candidates/202604081538_awq-lite-salient-rows

RUN_ID=awq_lite_srow \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MLP_LEAK=0.5 LATE_QAT_THRESHOLD=0 \
AWQ_CALIBRATION_STEPS=8 AWQ_CALIBRATION_TOKENS=131072 AWQ_CALIBRATION_SEQ_LEN=1024 \
AWQ_KEEP_RATIO=0.01 AWQ_KEEP_MIN=4 AWQ_KEEP_MAX=8 AWQ_MIN_ROWS=256 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks / tradeoffs

- This is **AWQ-lite**, not full AWQ: it protects salient **rows**, not arbitrary channels, so upside may be smaller than the paper.
- The calibration pass is intentionally tiny to stay practical; that may make the keep-list noisy.
- Int8 overrides consume artifact bytes, so aggressive keep settings can erase the win.
- LeakyReLU^2 and the new export path interact, so attribution may be messy without ablations.

## Validation

- `python -m compileall candidates/202604081538_awq-lite-salient-rows/train_gpt.py` — passed
- Minimal CPU smoke run — **not feasible in this workspace** because:
  - the current Python environment does not have `torch`, `numpy`, or `sentencepiece` installed,
  - the cached FineWeb shards are not present locally,
  - and this script requires CUDA for actual execution.
