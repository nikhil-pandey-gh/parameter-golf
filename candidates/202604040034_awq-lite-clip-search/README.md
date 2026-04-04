# AWQ-lite Activation-Weighted Clip Search

## Hypothesis

The current best stack is already close to saturated on architecture, schedule, and legal eval tricks, but its export path still picks low-bit clipping mostly from **weight-only reconstruction error**. Replacing that with a tiny **training-token calibration pass** and **activation-weighted clip selection** should reduce the int6/int8 roundtrip penalty without spending extra artifact bytes.

## Why this is promising for this repository

- The repo's biggest gains after March 20 came from **better export quality on the same 11-layer stack**: GPTQ-lite clip search, EMA, tighter warmdown, and legal TTT all beat larger architectural pivots.
- The current leaderboard-best record already sits near the **16 MB artifact ceiling**, so ideas that add residual tensors or extra preserved subspaces are harder to fit cleanly.
- This candidate keeps the whole `2026-03-23` training/eval recipe intact and only changes the **post-training quantization decision rule**, which is the highest-confidence place to intervene with minimal code.

## Prior experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best published stack in this repo (`val_bpb: 1.1194` 3-seed mean), so it is the strongest place to test a zero-byte export improvement.
- **Quantization precursor:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Showed that even a small improvement in clip selection mattered on the modern 11L/XSA/Partial-RoPE stack.
- **Architecture lineage:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Confirmed that the current record family is an accumulation of modest wins rather than one dramatic rewrite.
- There were **no prior experiments under `candidates/`** when this candidate was created.

## External research that informed it

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (`arXiv:2306.00978`)
  - Key idea used here: activation statistics are a better guide than raw weight magnitude for deciding which quantization choices matter most.
- **SmoothQuant** (`arXiv:2211.10438`)
  - Reinforced the same theme: offline calibration can improve low-bit behavior by explicitly accounting for activation outliers.
- **SpinQuant** (`arXiv:2405.16406`)
  - Rotation-based outlier flattening looks promising, but it would require a broader model/export rewrite than fits this repo's "next candidate" scope.

## What changed versus the chosen base

Starting from `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate adds:

1. **AWQ-lite calibration pass**
   - Uses a small slice of **training tokens**, not validation tokens, after EMA weights are applied and before export.
2. **Activation hooks for quantized matrices**
   - Collects per-channel RMS statistics for the inputs to the main attention, MLP, and tied-output projection weights.
3. **Activation-weighted clip search**
   - Replaces pure weight-MSE clip selection with an activation-weighted reconstruction error for both the int6 path and the large-matrix int8 path.
4. **No architecture or artifact-format rewrite**
   - Same model stack, same parameter banking, same legal TTT path, same int6+lzma export container.

## How to run / evaluate

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_ENABLED=1 AWQ_CALIBRATION_TOKENS=262144 AWQ_BATCH_SEQS=16 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `AWQ_ENABLED=1` is the candidate default.
- `AWQ_CALIBRATION_TOKENS` controls the export-time calibration budget.
- Default `DATA_PATH` and `TOKENIZER_PATH` resolve relative to this script, so the command works from the candidate directory without extra path overrides.
- If you want to isolate the export change from legal TTT, run once with `TTT_ENABLED=0`.

## Main risks / tradeoffs

- **Extra export time:** calibration is cheap relative to training, but it is still additional wall-clock work after training.
- **Calibration mismatch:** a small training-token sample may not rank matrix sensitivity perfectly.
- **Modest ceiling:** this attacks quantization loss, not model capacity, so gains may be incremental rather than dramatic.
- **More ambitious PTQ ideas remain open:** activation-aware residual/codebook or rotation-based export could still be stronger future experiments, but they cost more code and/or bytes.

## Validation

Ran from this candidate directory:

- `python -m compileall -f ../../train_gpt.py ../../train_gpt_mlx.py ../../data ./train_gpt.py`
  - **Passed**

Not run:

- CPU smoke test
  - **Not feasible in this environment** because `train_gpt.py` requires CUDA and FlashAttention at runtime, while the lightweight local check here is syntax-only compilation.
