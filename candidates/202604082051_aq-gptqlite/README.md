# Candidate: Activation-Aware GPTQ-lite Export

## Hypothesis

The current repo frontier already has a strong 11-layer training stack, so the next cheap win is more likely to come from **reducing the remaining int6 export error** than from another large architectural rewrite. This candidate adds a small **activation-aware diagonal rescaling** pass before GPTQ-lite export: calibrate per-channel activation RMS on a tiny slice of training data, scale the attention/MLP inputs, and inversely scale the corresponding weight columns so the full-precision function is preserved while int6 quantization sees friendlier weight ranges.

## Why this is promising here

- The record history repeatedly shows that **artifact shaping and quantization quality are first-order** in this repo: fp16/int8 embedding handling, mixed int6 export, warmdown, EMA/SWA, and GPTQ-lite all mattered a lot.
- The strongest clean non-TTT base is already `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, so a minimal next step should compose with that stack instead of restarting from an older baseline.
- Repo evidence also argues against heavier alternatives here: naive recurrence/looped depth was explicitly a bad fit for the fixed 10-minute wallclock, and several QAT variants were either expensive or inert.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Why quantization is the right lever:** `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`, `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`, `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
- **Why not pivot to recurrence:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
- **Why preserve the 11-layer stack:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` and `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

At the time this candidate was created, there were **no prior folders under `candidates/`** to inherit from.

## External research that informed it

- **SmoothQuant** — migrate activation outliers into weights via an equivalent offline transform: <https://arxiv.org/abs/2211.10438>
- **AWQ** — use activation statistics to protect salient weight channels with equivalent scaling: <https://arxiv.org/abs/2306.00978>
- **QuaRot** — outlier removal via invariant rotations, showing that export-time reparameterization is a strong quantization lever: <https://arxiv.org/abs/2404.00456>
- **SpinQuant** — learned rotations further validate the same theme, but with more machinery than this repo likely wants for a minimal candidate: <https://arxiv.org/abs/2405.16406>

The key takeaway I used is the common one across those papers: **make the quantizer see less pathological channel scales without changing the model’s real function**.

## What changed vs. the chosen base

Relative to the 2026-03-22 record stack, this candidate keeps the same overall architecture and export flow, then adds:

1. **Per-channel `input_scale` buffers** on:
   - attention inputs (shared by `c_q`, `c_k`, `c_v`)
   - MLP inputs (`fc`)
2. **A short calibration pass** over training batches after EMA is applied, collecting activation RMS for each block’s attention and MLP input.
3. **Activation-aware equivalent rescaling**:
   - compute a diagonal scale from activation RMS and weight RMS (`AQ_ALPHA`, `AQ_MAX_SCALE`)
   - multiply the runtime input by that scale
   - divide the corresponding weight columns by the same scale
4. **Unchanged downstream export path**:
   - same GPTQ-lite-style int6 per-row clip search
   - same mixed int6/int8 export logic
   - same EMA / XSA / partial RoPE / LN scale / BigramHash / VE stack

So the candidate is intentionally narrow: **better export conditioning, not a new training architecture**.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
AQ_ENABLED=1 AQ_ALPHA=0.5 AQ_MAX_SCALE=8 AQ_CALIB_BATCHES=8 AQ_CALIB_BATCH_TOKENS=65536 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Unlike the older record folders, this candidate resolves its default `DATA_PATH` and `TOKENIZER_PATH` from the script location, so the command above works when launched directly from `candidates/202604082051_aq-gptqlite/`.

The script still performs normal training, EMA application, quantized export, roundtrip eval, and sliding-window eval. The new AQ pass runs only once during export.

## Expected risks / tradeoffs

- **Small or zero gain is possible.** The repo’s GPTQ-lite export is already strong, so activation-aware scaling may only shave a tiny fraction of the remaining quantization gap.
- **Slight eval-time overhead.** The new per-channel multiply is cheap, but it is still extra work in every attention and MLP block at evaluation/export time.
- **Calibration sensitivity.** If `AQ_ALPHA` or `AQ_MAX_SCALE` are too aggressive, the transform may overfit to the tiny calibration sample or simply move error around instead of reducing it.
- **This does not solve train-time quantization.** It is a PTQ/export improvement layered on top of the existing stack, not a replacement for better learned QAT.

## Validation

Commands run in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604082051_aq-gptqlite/train_gpt.py
```

Outcome:

- Both compile checks passed.
- A meaningful CPU-only smoke test was **not feasible** here because the environment does not have the runtime dependency stack installed (`torch` was unavailable) and this script also hard-requires CUDA plus FlashAttention 3 for normal execution.
