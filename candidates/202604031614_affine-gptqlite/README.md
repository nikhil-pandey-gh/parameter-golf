# Affine GPTQ-lite on the current best stack

## Hypothesis

The current strongest record family already squeezed out most obvious architecture wins, but it still exports the final model with a **rowwise symmetric int6** path. A small **affine GPTQ-lite** variant that searches both the clip percentile and a per-row offset should reduce reconstruction error enough to improve post-quant roundtrip quality under the same training recipe.

## Why this is promising for this repository

- The repo's strongest recent gains came from **compression-aware training and export**, not just new layers. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` attributes a real gain to GPTQ-lite clip search alone.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py` still quantizes MLP and attention weights with a **zero-centered symmetric int6** routine.
- A repo-wide search did **not** find prior experiments using AWQ/SmoothQuant/QuaRot-style scaling, affine zero-points, asymmetric quantization, or other shifted low-bit export variants in `records/`.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current overall stack
  - contributes LeakyReLU^2, parameter banking, parallel Muon, legal TTT, and lzma export
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - established that better clip search in the int6 exporter is already worth chasing
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - kept the partial-RoPE + LN-scale branch that the current best stack still inherits
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative evidence that deeper recurrence was not a good next bet here

There were no prior `candidates/` directories to inherit from.

## External research that informed it

- **GPTQ** (`arXiv:2210.17323`) showed that one-shot post-training quantization quality is highly sensitive to how quantization error is minimized.
- **AWQ** (`arXiv:2306.00978`) showed that protecting salient channels can substantially reduce low-bit weight quantization error.
- **SmoothQuant** (`arXiv:2211.10438`) showed that equivalent transformations and outlier migration matter as much as raw bitwidth.
- **OmniQuant** (`arXiv:2308.13137`) showed that hand-crafted clipping is often the weak point and that learnable clipping / equivalent transforms improve low-bit PTQ.
- **QuaRot** (`arXiv:2404.00456`) reinforced that outlier handling via simple mathematically equivalent transforms can unlock stronger low-bit export.

This candidate implements the smallest repo-compatible version of that idea family: **per-row affine centering** inside the existing GPTQ-lite search loop, without adding calibration data, new training infrastructure, or a new runtime format.

## What changed versus the chosen base implementation

Base file: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. **Affine int6 export**
   - the old exporter searched clip percentile only
   - the new exporter searches:
     - row offset in `{0, row mean, row midrange}`
     - clip percentile in `{0.9990, 0.9995, 0.9999, 0.99999, 1.0}`
   - it stores a per-row fp16 offset and reconstructs with `q * scale + offset`
2. **CPU-safe attention fallback**
   - if `flash_attn_interface` is unavailable, the script falls back to `torch.nn.functional.scaled_dot_product_attention`
3. **Smoke-test path**
   - `SMOKE_TEST=1` runs a tiny synthetic CPU-only forward pass plus affine-int6 roundtrip so the candidate can be sanity-checked without FineWeb shards, GPUs, or the full training-only Python dependency set

Everything else is intentionally kept aligned with the current best stack.

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
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a local no-data sanity check:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation recorded for this candidate

1. `python -m compileall candidates/202604031614_affine-gptqlite/train_gpt.py`  
   - **passed**
2. `SMOKE_TEST=1 python candidates/202604031614_affine-gptqlite/train_gpt.py`  
   - **passed**
   - also passed after uninstalling `numpy` and `sentencepiece` from the local validation venv, which confirms the smoke path really bypasses the full training imports
   - printed: `smoke_test:ok loss:4.851839 quant_loss:4.851837 flash_attn:0`
3. `python -m compileall train_gpt.py train_gpt_mlx.py data`  
   - **passed**

## Main expected risks and tradeoffs

- The extra fp16 row offsets may improve MSE but still lose on total compressed artifact bytes.
- Better weight-space reconstruction does not guarantee better **sliding-window bpb** or better **post-TTT** bpb.
- The fallback attention path is for validation ergonomics; real leaderboard behavior still depends on the intended GPU path.
- This is still an unbenchmarked candidate, not a validated record submission.
