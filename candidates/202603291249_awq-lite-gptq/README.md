# AWQ-lite GPTQ on the LeakyReLU2 + Legal TTT stack

## Hypothesis

The current best line in this repository already gets strong gains from better post-training quantization (`GPTQ-lite` percentile search) without changing the trained model. My hypothesis is that a **small activation-aware extension to the existing int6 export path** can reduce the remaining quantization error further, especially for the banked attention and MLP matrices that dominate the artifact.

Concretely, this candidate keeps the full `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` training recipe intact and adds a short **train-data calibration pass** after EMA application. During export, each large int6 matrix now searches over a few **groupwise input-channel scaling exponents** before the existing per-row percentile clip search. The winning scale is folded back out during dequantization, so inference semantics stay unchanged while salient channels get better protected during quantization.

## Why this is promising for this repository

The repo history strongly suggests that the best next improvements come from **surgical export-path wins on top of the best 11-layer stack**, not from resetting the architecture.

Relevant trends from prior records:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that a better post-training clip search alone was worth about **-0.0006 BPB**.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` retained the same broad quantization philosophy but pushed the stack further with LeakyReLU², legal score-first TTT, and parameter-banked training.
- Earlier records repeatedly improved when they made the final exported artifact easier to compress or less sensitive to quantization noise.

This candidate follows that same pattern: **do not perturb the strongest training stack**, only try to make its final int6 export more faithful.

## Influential prior records and candidates

There were **no prior `candidates/` iterations** in this repository when this candidate was created.

The main local influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - chosen as the base implementation because it is the strongest current tracked stack.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - especially important because it established the repo's current `GPTQ-lite` clip-search direction.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - useful as evidence that small, zero-or-near-zero parameter changes to the best 11-layer recipe can still matter.

## External research that informed this candidate

This candidate is primarily motivated by three post-training quantization papers:

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (`arXiv:2306.00978`)
  - core idea used here: activation statistics reveal which channels are most important, and equivalent scaling transforms can protect those channels during low-bit weight quantization.
- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (`arXiv:2210.17323`)
  - motivates spending extra export-time compute on a better weight-only PTQ path.
- **SmoothQuant** (`arXiv:2211.10438`)
  - supports the broader idea that equivalent rescaling transforms can move quantization difficulty into easier-to-handle parameters without changing the represented function.

This candidate does **not** implement full AWQ or SmoothQuant. Instead, it takes the most repo-compatible slice of that literature: a **small activation-aware, groupwise scaling search layered on top of the existing GPTQ-lite-style per-row clip search**.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- Added repo-root defaults for `DATA_PATH` and `TOKENIZER_PATH` so the script can be run directly from this candidate directory.
- Added AWQ-lite export knobs:
  - `AWQ_ENABLED`
  - `AWQ_BATCH_TOKENS`
  - `AWQ_CALIBRATION_BATCHES`
  - `AWQ_GROUP_SIZE`
  - `AWQ_ALPHA_CANDIDATES`
- Added a short **post-training calibration pass on training shards only** after EMA application.
- Collected per-matrix activation RMS statistics for:
  - attention `q/k/v` inputs,
  - attention output projection inputs,
  - MLP up-projection inputs,
  - MLP down-projection inputs.
- Extended the int6 export path so each large int6 matrix can optionally:
  - build groupwise input-channel scales from calibration statistics,
  - search over a small exponent set,
  - run the existing per-row percentile clip search on the scaled weights,
  - store compact group scales in the export metadata,
  - undo the scaling during dequantization.
- Kept the training recipe, EMA, TTT, parameter banking, LeakyReLU² activation, XSA, VE, and sliding-window evaluation logic otherwise unchanged.

## How to run / evaluate

From this directory:

```bash
cd candidates/202603291249_awq-lite-gptq
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_ENABLED=1 AWQ_CALIBRATION_BATCHES=8 AWQ_BATCH_TOKENS=131072 \
AWQ_GROUP_SIZE=64 AWQ_ALPHA_CANDIDATES=0.0,0.25,0.5,0.75,1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For an ablation against the same script, disable the new export path with:

```bash
cd candidates/202603291249_awq-lite-gptq
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_ENABLED=0 AWQ_CALIBRATION_BATCHES=8 AWQ_BATCH_TOKENS=131072 \
AWQ_GROUP_SIZE=64 AWQ_ALPHA_CANDIDATES=0.0,0.25,0.5,0.75,1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation run for this candidate

Commands executed in this environment:

```bash
# From the repository root:
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603291249_awq-lite-gptq/train_gpt.py

# From this candidate directory:
cd candidates/202603291249_awq-lite-gptq
python -m compileall train_gpt.py
```

Outcome:

- **Passed from the repo root**.
- **Passed from the candidate directory**.

Attempted runtime environment check:

```bash
python3 - <<'PY'
try:
    import torch
    print(torch.__version__)
except Exception as e:
    print(type(e).__name__ + ': ' + str(e))
PY
```

Outcome:

- `ModuleNotFoundError: No module named 'torch'`

Because this container does not have `torch` installed, and this candidate also depends on CUDA/FlashAttention for the real run path, I could not complete a meaningful CPU runtime smoke test here. The static Python validation passed, but an actual start-up check needs the normal challenge training environment.

## Main expected risks / tradeoffs

- **Evaluation-time overhead**: the added calibration pass costs extra export/eval time, even though it is intentionally small and master-only.
- **Compression tradeoff**: better reconstruction error does not automatically mean better compressed artifact size after `lzma`; the additional scale metadata is small, but it is not free.
- **Calibration sensitivity**: the chosen train-data batches and exponent grid may help some matrices more than others.
- **Possible no-op on some layers**: for many matrices the best alpha may still be `0.0`, meaning the search falls back to the original GPTQ-lite path.

## Expected next experiments if this helps

- Sweep `AWQ_GROUP_SIZE` (`32`, `64`, `128`).
- Narrow the scale search to only the most quantization-sensitive matrices if export time matters.
- Combine this with the warmdown-fraction fix explored in the non-record 5090 experiments if a future branch wants a training-side change too.
