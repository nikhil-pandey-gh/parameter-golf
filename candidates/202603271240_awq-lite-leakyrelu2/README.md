# AWQ-lite + LeakyReLU^2

## Hypothesis

The strongest low-risk next step for this repository is to keep the proven 11-layer GPTQ-lite/EMA/XSA/Partial-RoPE stack, swap in the repo's current best MLP activation (`LeakyReLU(0.5)^2`), and make the final int6 clip search activation-aware using a tiny calibration pass over training tokens.

The core bet is that this challenge is now bottlenecked as much by export robustness as by raw pre-quant loss. If activation-aware weighting can reduce the quantization error on the same trained model, it may improve final `val_bpb` without paying a large training-time tax.

## Why this is promising for this repository

Repository evidence points in the same direction:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the strongest clean non-TTT base and already shows that a better post-training clip search is worth about `-0.0006 BPB`.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` shows `LeakyReLU(0.5)^2` is a real win on top of a late-March stack.
- The repo's trend is consistently toward better mixed quantization, better averaging, and low-cost architectural tweaks rather than broad new infrastructure.
- The non-record `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` notes that layer recurrence hurt under the fixed wallclock budget, so this candidate deliberately targets the quantization bottleneck instead of adding more depth/recurrence complexity.

## Influencing records and prior candidates

There were no prior `candidates/` directories in this repository when this candidate was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - base implementation
  - EMA + tight SWA
  - GPTQ-lite percentile search
  - XSA4 + Partial RoPE + LN scale + VE
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - `LeakyReLU(0.5)^2` activation change
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - reinforces that Partial RoPE + LN scale were durable wins, while the late-QAT toggle itself was not driving the result
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - EMA/XSA4 backbone lineage

## External research that informed it

- **GPTQ** (`arXiv:2210.17323`) argues that weight-only quantization quality depends on using a better error model than naive round-to-nearest.
- **SmoothQuant** (`arXiv:2211.10438`) highlights that activation outliers strongly shape quantization difficulty.
- **AWQ** (`arXiv:2306.00978`) shows that activation statistics identify the channels that matter most for weight-only quantization.
- **Scaling Law for Quantization-Aware Training** (`arXiv:2505.14302`) argues that weight quantization error remains important and can worsen as training-token count grows, which matches this repo's late-stage regime.

I also considered recent parameter-sharing / recurrent-depth directions such as Relaxed Recursive Transformers (`arXiv:2410.20672`) and Intra-Layer Recurrence (`arXiv:2505.01855`), but they require a more invasive architecture shift and are less aligned with the repository's current winning trend under the fixed 10-minute training cap.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

- `LeakyReLU(0.5)^2` replaces `ReLU^2` in the MLP.
- Added an **AWQ-lite calibration pass** after EMA is applied and before final export:
  - registers forward pre-hooks on quantized linear layers,
  - collects per-input-channel mean squared activations from a small number of training batches,
  - feeds those statistics into the existing GPTQ-lite percentile search.
- The int6 row-wise clip search is now **activation-aware**:
  - candidate clip percentiles are scored with activation-weighted reconstruction error instead of plain unweighted MSE.
- The script now uses **repo-relative default paths**, so it can be run from inside this candidate directory without manually overriding `DATA_PATH` and `TOKENIZER_PATH`.
- Added a **FlashAttention fallback** to PyTorch SDPA for importability and lightweight CPU-side smoke checks of the model components.
- Late QAT is left **off by default** in this candidate because prior repo evidence showed that the compiled late-toggle path was not a reliable source of gains.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603271240_awq-lite-leakyrelu2
RUN_ID=awq_lite_leakyrelu2 \
SEED=1337 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
AWQ_ENABLED=1 \
AWQ_CALIBRATION_BATCHES=8 \
AWQ_CALIBRATION_BATCH_TOKENS=131072 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `DATA_PATH` and `TOKENIZER_PATH` default to the repository's top-level `data/` tree.
- EMA remains enabled with the base script's fixed decay of `0.997`.
- `LATE_QAT_THRESHOLD` defaults to `0.0` here; set it manually only if you want to revisit that path explicitly.
- `AWQ_CALIBRATION_BATCHES=0` disables the activation-aware calibration path and falls back to the original GPTQ-lite behavior.

## Validation

Commands run for this candidate in the current runner:

```bash
python -m compileall candidates/202603271240_awq-lite-leakyrelu2/train_gpt.py
python - <<'PY'
# AST-level check that the AWQ helper uses the existing DistributedTokenLoader API
PY
```

Outcome:

- Passed.
- The static AST check confirmed that the AWQ calibration helper now calls `DistributedTokenLoader(...)` with 4 positional arguments and `next_batch(...)` with 3 positional arguments, matching the base script's loader API.

Attempted lightweight CPU smoke validation:

```bash
python - <<'PY'
# import candidate module, build a tiny GPT, run a forward, quantize/dequantize
PY
```

Outcome:

- Not feasible in this runner because the available Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- The candidate script was still adjusted to be friendlier to such smoke tests once repo dependencies are available: repo-relative defaults and a FlashAttention-to-SDPA fallback were added for that reason.

## Main expected risks or tradeoffs

- The AWQ-lite path adds a few post-training forward passes, so export/eval latency increases modestly.
- The activation-weighted clip search is only a lightweight approximation to full AWQ/GPTQ, so the gain may be smaller than hoped.
- A calibration subset that is too small may produce noisy salience estimates; one that is too large may cost too much time.
- `LeakyReLU(0.5)^2` is already strong in the repo, but its interaction with this exact EMA/GPTQ-lite stack still needs empirical confirmation.
