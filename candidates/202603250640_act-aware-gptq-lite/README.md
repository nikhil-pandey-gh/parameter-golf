# Act-Aware GPTQ-lite

## Hypothesis

The current best non-TTT export path in this repository still chooses int6 clip scales using only weight reconstruction error. Recent post-training quantization research argues that activation statistics, not just weight magnitudes, identify the channels that matter most. This candidate tests whether an **activation-aware GPTQ-lite** export step can reduce the int6 roundtrip gap on the strong 11-layer EMA/XSA/Partial-RoPE stack without changing the artifact budget or introducing a new training regime.

## Why this is promising for this repository

The record history shows that this challenge is now quantization-sensitive:

- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` pushed the architecture frontier, but still had room in the quantized export path.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` got an additional gain just by replacing fixed int6 clipping with a small clip-percentile search.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` kept GPTQ-lite in the quantization stack, suggesting that export-time compression quality is still an open lever even on stronger models.

That makes activation-aware scale selection attractive here: it is cheap, local to export, and directly targets a bottleneck that prior records already identified as worth optimizing.

## Prior records that influenced this candidate

Primary implementation base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Other strong influences:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for the 11-layer XSA4 + Partial RoPE + LN-scale stack
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` for the proven `LeakyReLU(0.5)^2` MLP activation change
- Earlier quantization-focused records showing embedding / export sensitivity, especially `2026-03-18_FP16Embed_WD3600` and `2026-03-19_MixedQuant_Int6Int8_SlidingWindow`

## External research that informed it

The candidate is mainly motivated by post-training quantization work that uses activation statistics to protect important channels:

- **GPTQ** — one-shot PTQ with second-order information for GPT-family models (`arXiv:2210.17323`)
- **SmoothQuant** — migrates quantization difficulty using activation statistics via equivalent transforms (`arXiv:2211.10438`)
- **AWQ (Activation-aware Weight Quantization)** — shows that activation distributions identify salient channels better than raw weight magnitudes (`arXiv:2306.00978`)
- **AQLM** was also considered (`arXiv:2401.06118`), but its additive-codebook machinery felt too invasive for a minimal candidate in this repository

This implementation is intentionally lighter than full AWQ or SmoothQuant. Instead of adding a new inference representation, it keeps the repo's per-row int6 format and only changes the scale-selection objective.

## What changed versus the base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **LeakyReLU(0.5)^2 MLP**
   - Replaces `relu^2` with the later record's stronger `LeakyReLU(0.5)^2` activation.

2. **Activation calibration pass before export**
   - After EMA is applied and before quantization, the script runs a short calibration pass on training shards.
   - It registers lightweight pre-forward hooks on `CastedLinear` modules in attention and MLP paths.
   - For each quantized weight matrix, it accumulates the mean squared input activation per input channel.

3. **Activation-aware int6 clip search**
   - The previous GPTQ-lite logic picked one of a few clip percentiles by minimizing plain weight reconstruction MSE.
   - This candidate keeps the same percentile search grid, but scores each candidate with an **activation-weighted reconstruction error** that approximates output distortion under observed input statistics.
   - Selection is done per row, which is slightly finer-grained than the previous matrix-level error choice.

4. **No new artifact format**
   - Export remains mixed int6/int8 in the repository's existing serialized format.
   - Compression remains `zstd` when available, otherwise `zlib`, matching the copied base script.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603250640_act-aware-gptq-lite

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
LATE_QAT_THRESHOLD=0.15 \
MLP_LEAK=0.5 \
AWQ_ENABLED=1 AWQ_CALIB_BATCHES=4 AWQ_CALIB_BATCH_TOKENS=262144 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Toggles that are most useful for ablation:

- `AWQ_ENABLED=0` to recover the copied weight-only GPTQ-lite behavior
- `MLP_LEAK=0.0` to move back toward the old `relu^2` behavior
- `AWQ_CALIB_BATCHES` and `AWQ_CALIB_BATCH_TOKENS` to trade calibration quality vs export overhead; `AWQ_CALIB_BATCH_TOKENS` must stay a positive multiple of `WORLD_SIZE * TRAIN_SEQ_LEN`
- `INT6_CLIP_CANDIDATES=...` to change the clip-percentile search grid; it must contain at least one percentile

## Main expected risks or tradeoffs

- The calibration pass adds some export-time overhead.
- Activation statistics are gathered from training shards, so if they are not representative of the evaluation distribution, the weighting could overfit slightly.
- This is an approximation to AWQ-style importance, not a full equivalent-transform implementation.
- The copied base still relies on `torch.compile`, CUDA, and FlashAttention kernels for the real training/eval path.
- Late-QAT behavior is inherited from the base family, so the candidate should still be checked carefully under the exact runtime stack.

## Validation run in this workflow

Commands attempted in this environment:

```bash
python -m compileall candidates/202603250640_act-aware-gptq-lite/train_gpt.py
python3 -m compileall candidates/202603250640_act-aware-gptq-lite/train_gpt.py
```

Outcome:

- `compileall` succeeded with both `python` and `python3`.

Additional smoke test attempt:

```bash
python3 - <<'PY'
# import candidate module with a stubbed flash_attn_interface,
# then exercise MLP and quantization helpers on CPU tensors
PY
```

Outcome:

- Not feasible on this runner because the available `python3` environment does not have `torch` installed, so even a CPU-only helper import smoke could not be executed here.
- I therefore limited automated validation to syntax compilation only.
