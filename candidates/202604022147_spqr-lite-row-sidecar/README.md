# SpQR-lite Row Sidecar on the 1.1194 TTT Stack

## Hypothesis

The current best in-tree stack already looks close to saturated on training-side changes, but it still relies on a lossy int6 artifact path. A tiny **row-residual sidecar** can spend a small extra artifact budget on the rows that quantize worst, shrinking the post-training quantization gap without changing the 10-minute training loop.

## Why this is promising for this repository

The repo’s best improvements repeatedly came from making the compressed artifact less damaging:

- `2026-03-19_WarmdownQuantization` showed that quantization-friendlier weights were worth more than many architectural tweaks.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` got another measurable gain from a better post-training int6 quantizer alone.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` then improved training/eval further, but still exports the same basic GPTQ-lite-style int6 payload.

That makes the artifact format itself the cleanest remaining place to search for a gain.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - chosen as the base because it is the current best stack in-tree (`val_bpb 1.1194`) and already includes the strongest known training/eval recipe.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - showed that a better post-training quantizer can still move BPB without paying extra training cost.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - reinforced that the winning architecture is the 11-layer XSA/partial-RoPE/EMA family, so this candidate keeps that core intact.
4. `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
   - highlighted quantization sensitivity as a first-order constraint, not an afterthought.

## External research that informed it

- **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression** (`arXiv:2306.03078`)
  - isolates outlier weights and stores them in higher precision alongside a low-bit backbone.
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (`arXiv:2306.00978`)
  - argues that only a small subset of salient channels needs protection to reduce quantization damage.
- **SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models** (`arXiv:2411.05007`)
  - uses a small high-precision auxiliary branch to absorb quantization-sensitive structure instead of spending high precision everywhere.

This candidate borrows the shared idea behind those papers — **protect only the most quantization-sensitive structure** — but adapts it to this repo’s tiny-model, single-file, artifact-budgeted workflow.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **SpQR-lite row sidecar for int6 matrices**
   - after GPTQ-lite per-row int6 quantization, compute the per-row reconstruction MSE;
   - globally rank rows across int6 attention/MLP matrices by that error;
   - spend a fixed raw budget (`INT6_SIDECAR_BUDGET_BYTES`, default `65536`) on the worst rows;
   - store those rows as **fp16 residuals** plus row indices;
   - add the residuals back after dequantization.
2. **No training-loop changes**
   - training, EMA/SWA, legal score-first TTT, and the rest of the 1.1194 stack are preserved.
3. **Safe attention fallback**
   - if `flash_attn_interface` is unavailable, the script falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - this is mainly for local smoke testing and portability, not for leaderboard-speed runs.

## How to run

From this candidate directory:

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
INT6_SIDECAR_BUDGET_BYTES=65536 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

The script keeps the base stack’s eval flow:

1. apply EMA/SWA weights,
2. export `final_model.int6.ptz`,
3. reload the compressed artifact,
4. run roundtrip eval,
5. run sliding-window eval,
6. optionally run legal score-first TTT if `TTT_ENABLED=1`.

## Validation

Commands used during candidate creation:

```bash
python -m compileall candidates/202604022147_spqr-lite-row-sidecar/train_gpt.py
/tmp/gh-aw/agent/venv/bin/python - <<'PY'
# import the candidate module in a temp venv with torch/numpy/sentencepiece,
# instantiate a small CPU GPT, run a forward pass,
# run mixed_quantize_int6(..., sidecar_budget_bytes=4096),
# dequantize, reload, and run another forward pass.
PY
```

Outcomes:

- `python -m compileall ...` **passed**.
- Tiny CPU import/forward/quantize smoke: **passed** after creating a temp venv with the missing runtime deps.
  - forward loss: `5.576221942901611`
  - quantized roundtrip loss: `5.576205253601074`
  - sidecar usage at `INT6_SIDECAR_BUDGET_BYTES=4096`: `10` rows, `3860` raw bytes selected

## Main risks and tradeoffs

1. **Artifact-size risk**: fp16 residual rows may compress worse than expected; if so, the sidecar budget must be reduced.
2. **Weight-only saliency**: this version ranks rows by post-quant reconstruction error, not by activation-aware saliency; that is simpler, but may miss the best rows to protect.
3. **Late-stage gain may be small**: the current stack is already very strong, so a clean result may be only a few 1e-4 BPB.
4. **Fallback path is slower**: the SDPA fallback is for validation portability, not the fast-path leaderboard environment.
