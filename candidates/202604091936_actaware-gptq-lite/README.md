# Activation-Aware GPTQ-lite PTQ

## Hypothesis

The strongest unexplored lever in this repository is **post-training quantization quality**, not another heavier training-time architecture change. If the 11-layer EMA + GPTQ-lite stack is already near the best train-time recipe, then replacing its weight-only int6 clip search with a lightweight **activation-aware calibration pass** should shrink the final float-to-int6 gap without increasing artifact bytes.

In practice, this candidate keeps the proven training stack from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` and changes only the export path:

1. cache the last few training batches on rank 0,
2. estimate per-layer input second moments from those batches,
3. use those activation statistics to choose each int6 row's clip and rounding offset,
4. serialize the same int6 payload format as before.

## Why this is promising here

- The repository has repeatedly shown that this challenge is often **compression-limited** rather than purely train-loss-limited:
  - the non-record 4-hour run reached much stronger pre-quant quality than its deployed artifact preserved;
  - `WarmdownQuantization` explicitly framed the quantization penalty as the dominant bottleneck;
  - the `11L EMA + GPTQ-lite` record still improved by refining the PTQ step alone.
- GPTQ-lite already delivered a small but real gain in the current best training-only stack, so a stronger activation-aware PTQ variant is a natural follow-on.
- This avoids the most common dead ends in the record history:
  - no extra recurrent depth,
  - no slower attention variant,
  - no always-on QAT overhead,
  - no new artifact-side parameters.

## Prior repository work that influenced this candidate

- **Root baseline**: `train_gpt.py` in the repo root established the core Muon + Adam split, tokenizer-agnostic BPB evaluation, U-Net-like skip layout, and serialized roundtrip validation.
- **`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`**: chosen base implementation because it is the strongest local training-only stack and already isolates quantization as a meaningful optimization target.
- **`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`**: important caution that "late QAT" can silently turn into a no-op under `torch.compile`, reinforcing the appeal of a PTQ-only change.
- **`2026-03-19_WarmdownQuantization`** and **`2026-03-18_FP16Embed_WD3600`**: both emphasize that deployment quality and quantization sensitivity can dominate small hyperparameter gains.
- **`2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3`**: useful evidence that stronger float models can still lose a lot at quantized export time.

## External research that informed it

- **AdaRound** (Nagel et al., 2020): optimize rounding decisions rather than treating nearest-rounding as fixed. <https://arxiv.org/abs/2004.10568>
- **BRECQ** (Li et al., 2021): block reconstruction for PTQ, motivating layer/block output preservation instead of pure weight MSE. <https://arxiv.org/abs/2102.05426>
- **SmoothQuant** (Xiao et al., 2022): activation statistics are useful signals for weight quantization decisions. <https://arxiv.org/abs/2211.10438>

This candidate implements a deliberately smaller version of that idea: a **diagonal Hessian / second-moment proxy** using cached late-training activations, rather than a full per-layer optimization loop.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. **Caches recent training batches** on the master rank during the final training trajectory.
2. **Collects per-input-feature second moments** for each quantized `CastedLinear` just before export.
3. Replaces weight-only GPTQ-lite row search with an **activation-aware rowwise search**:
   - same percentile clip candidates,
   - plus a small rounding-offset grid,
   - scored with activation-weighted reconstruction error instead of raw weight MSE.
4. Keeps the same exported artifact format (`int6` weights + compressed payload), so the calibration logic adds **no deployment bytes**.
5. Adds a **FlashAttention fallback** and a **synthetic CPU smoke test** path so the candidate can be validated locally without CUDA-only kernels.

## How to run

From this candidate directory (the script searches upward for the repository root before resolving the default dataset/tokenizer paths, so you do not need to run from `/parameter-golf`):

```bash
ACT_AWARE_PTQ=1 \
PTQ_CALIB_BATCHES=8 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful toggles:

- `ACT_AWARE_PTQ=0` disables the new calibration and falls back to the base GPTQ-lite-style export.
- `PTQ_CALIB_BATCHES=<N>` controls how many late-training batches are used to estimate activation statistics.
- `PTQ_ROUND_OFFSETS=-0.125,0,0.125` adjusts the rowwise rounding-bias search grid.
- `PTQ_CLIP_CANDIDATES=0.9990,0.9995,0.9999,0.99999,1.0` adjusts the clip search grid.

## Validation

Commands run for this candidate from inside `candidates/202604091936_actaware-gptq-lite/`:

```bash
python -m compileall train_gpt.py
SMOKE_TEST=1 python train_gpt.py
```

Outcomes:

- `compileall`: **passed**
- `SMOKE_TEST=1`: **passed** with `smoke_test_ok loss:4.7620 calibrated_tensors:8`

The smoke mode is intentionally synthetic: it exercises model construction, forward/backward, activation-stat collection, int6 export, dequantized reload, and a final forward pass on CPU. In the workflow environment these commands were run inside a temporary virtualenv because the base runner did not have the repository's Python requirements preinstalled.

## Main expected risks and tradeoffs

- **Calibration time**: the activation-aware export pass adds extra post-training compute on rank 0.
- **Approximation quality**: second-moment weighting is cheaper than full AdaRound/BRECQ optimization, but it ignores cross-feature covariance and may leave some gain on the table.
- **Batch locality**: using only the last few training batches could overfit the PTQ decision to late-training activation statistics.
- **Interaction with existing QAT code**: this candidate is designed to help even when training-time fake quantization is ineffective or disabled, but the exact combination still needs real GPU benchmarking.
