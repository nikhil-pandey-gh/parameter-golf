# Square-AWQ-lite on the 2026-03-23 banked LeakyReLU² stack

## Hypothesis

The current best record already uses a banked 11-layer stack with `LeakyReLU(0.5)^2`, GPTQ-lite int6 export, and legal score-first TTT. That stack still treats weight-only quantization as a purely weight-domain problem.

This candidate tests a small export-time twist: use a short post-training calibration pass on **training** batches to estimate each MLP hidden channel's RMS after the `LeakyReLU(0.5)^2` nonlinearity, then apply an **exact square-aware channel rescaling** before GPTQ-lite int6 quantization.

Because `LeakyReLU(x)^2` is positively homogeneous of degree 2, we can rescale the MLP pair without changing the floating-point function:

```python
up' = up / sqrt(s)
down' = down * s
```

for a positive per-channel scale vector `s`. The intent is to protect the most active down-projection channels, where int6 error should matter most, while preserving the original fp model exactly before quantization.

## Why this is promising for this repository

The record history in `records/` shows a clear pattern:

- the strongest models all converge on the same 11-layer, 512d, 3x-MLP, XSA, Partial-RoPE, LN-scale family,
- later gains mostly come from **better export/eval tricks** rather than wholesale architecture changes,
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved the stack with GPTQ-lite clip search,
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` improved it again with LeakyReLU², parameter banking, and legal TTT.

What appears to be missing is any **activation-aware equalization** step before the int6 export. That is appealing here because:

- it adds essentially no artifact bytes,
- it reuses the current strongest base script,
- it targets the exact same weight-only quantization bottleneck that recent records keep optimizing,
- it fits the repo's bias toward precise post-training improvements rather than broad infrastructure changes.

## Prior records and experiments that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Direct influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - established GPTQ-lite percentile clip search as a meaningful zero-training-cost export win.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - locked in Partial RoPE + layerwise LN scaling as part of the best architecture family.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - provided the earlier 11-layer EMA/XSA recipe that the later scripts refined.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful mainly as a negative result: deeper recurrence / broad architectural departures were less attractive than focused export-side improvements.

## External research that informed it

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (`arXiv:2306.00978`)
  - motivates using activation statistics to decide which weight channels deserve protection in low-bit weight-only quantization.
- **SmoothQuant** (`arXiv:2211.10438`)
  - motivates mathematically equivalent channel rescaling to move quantization difficulty away from the most sensitive part of the computation.

This candidate adapts those ideas to this repo's very specific MLP nonlinearity: because the active base uses `LeakyReLU(0.5)^2`, the equalization needs to respect the nonlinearity's degree-2 homogeneity, not the degree-1 scaling used in a standard linear-GELU-linear block.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- added `AWQ_LITE_*` hyperparameters for the equalization pass,
- added lightweight MLP buffers for hidden-channel RMS statistics,
- added a short **post-training calibration pass** over training shards to collect hidden RMS after `LeakyReLU(0.5)^2`,
- added a square-aware equalization transform on unbanked MLP weights immediately before the existing GPTQ-lite int6 export,
- left the rest of the training stack intact: parameter banking, Parallel Muon, Partial RoPE, LN scale, VE128, GPTQ-lite, legal TTT, etc.

## How to run / evaluate

From the repository root:

```bash
RUN_ID=square_awq_lite \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
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
AWQ_LITE_ENABLED=1 AWQ_LITE_ALPHA=0.5 AWQ_LITE_MAX_SCALE=2.0 \
AWQ_LITE_CALIBRATION_BATCHES=8 AWQ_LITE_CALIBRATION_TOKENS=131072 \
SEED=1337 \
 torchrun --standalone --nproc_per_node=8 candidates/202603311149_square-awq-lite/train_gpt.py
```

Note: this script inherits the 2026-03-23 base behavior where EMA is already baked into the export path, so there is no separate `EMA_ENABLED` toggle to pass here.

Lower-cost ablations to try first:

- `AWQ_LITE_ALPHA=0.25` or `0.75`
- `AWQ_LITE_MAX_SCALE=1.5` vs `2.5`
- `AWQ_LITE_CALIBRATION_BATCHES=4` vs `16`

## Main expected risks / tradeoffs

- The down-projection is quantized per row, while this equalization acts per hidden channel; that mismatch may blunt or even reverse the expected int6 gain.
- The short calibration pass is cheap relative to training, but it is still additional work and may need retuning to stay clearly worthwhile.
- The idea is deliberately narrow: it should help export quality if the quantization bottleneck is MLP-channel sensitivity, but it will not fix unrelated limitations in the trained checkpoint.
- If TTT remains the dominant final improvement, a pure export-side gain may only move the pre-TTT score modestly.

## Validation

Commands run in this workflow environment:

```bash
python -m compileall candidates/202603311149_square-awq-lite/train_gpt.py
python - <<'PY'
import importlib.util
print(importlib.util.find_spec('torch') is not None)
print(importlib.util.find_spec('flash_attn_interface') is not None)
PY
```

Outcomes:

- `python -m compileall ...` succeeded.
- A real runtime smoke test was **not feasible in this workflow environment** because the local Python environment does not provide `torch`, and the candidate script is intentionally CUDA + FlashAttention based.
- I therefore validated syntax only here and left runtime validation to a proper Parameter Golf training environment.
