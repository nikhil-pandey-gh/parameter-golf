# MLP AWQ-Style Equivalent Equalization

## Hypothesis

The strongest local recipe already pushed architecture, optimization, and legal TTT quite far, so the next cheap gain is likely in the export path rather than the training loop.

This candidate adds an **activation-aware, function-preserving MLP equalization step before GPTQ-lite int6 export**. Because the base stack uses `LeakyReLU(0.5)^2`, each MLP channel is positively homogeneous: scaling the MLP-up row by `s` and the matching MLP-down column by `1 / s^2` leaves the full-precision network unchanged. The hope is that this flattens salient hidden-channel outliers and lowers the post-quantization error that still separates the strong fp/bf16 model from the submitted int6 artifact.

## Why it is promising for this repository

The repo history is dominated by one theme: **quantization quality is the bottleneck**.

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` showed that protecting sensitive tensors can collapse the quantization gap.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that even a zero-training-cost GPTQ-lite improvement still bought another measurable gain.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` is the current strongest local stack, but it still relies on the same broad int6 export idea.

That makes this a good fit for a **minimal, compression-aware extension** of the best current code rather than a broad architectural rewrite.

## Which records influenced it

The base implementation is:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Other important influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - motivation for improving clip/quant choices without adding training cost
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - retained the strong 11-layer EMA / partial-RoPE / LN-scale lineage
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md`
  - reinforced that quantization-aware regularization and byte allocation dominate this challenge

## External research that informed it

- **SmoothQuant** — Xiao et al., arXiv:2211.10438  
  Training-free equivalent transformations can move quantization difficulty away from fragile outliers.

- **AWQ** — Lin et al., arXiv:2306.00978  
  Activation-aware channel scaling can protect salient weights without broad mixed-precision infrastructure.

- **OmniQuant** — Shao et al., arXiv:2308.13137  
  Learnable clipping plus equivalent transformations improved PTQ quality in diverse low-bit settings.

- **SpinQuant** — Liu et al., arXiv:2405.16406  
  Orthogonal / equivalent pre-quantization transforms can materially reduce outlier-driven PTQ damage.

- **FlatQuant** — Sun et al., arXiv:2410.09426  
  Flattening transformed weight/activation distributions before PTQ remains a strong theme in recent work.

This candidate uses the **lowest-infrastructure variant** of that literature: a small, exact MLP-only transformation that fits the existing script.

## What changed versus the chosen base implementation

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`:

1. Added a **post-training calibration pass** over a small, deterministic training-token slice (`PTQ_EQ_CALIB_TOKENS`) to estimate per-layer MLP hidden-channel saliency.
2. Added a **layer-wise search over scaling exponents** (`PTQ_EQ_ALPHA_CANDIDATES`) to choose the best equivalent rescaling for each MLP pair.
3. Applied the exact transformation:
   - `mlp_up[row_j] *= s_j`
   - `mlp_down[:, j] *= 1 / s_j^2`
4. Kept the existing **GPTQ-lite int6** export, but now quantized the equalized weights instead of the raw ones.
5. Added a **FlashAttention fallback** plus `SMOKE_TEST=1` CPU mode that now runs:
   - a random-token forward pass, and
   - a tiny synthetic calibration -> equalize -> quantize -> dequantize roundtrip
   so the smoke path covers the new PTQ/export logic rather than only the plain forward pass, and fails if the roundtrip produces non-finite outputs or excessive loss/logit drift.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603261824_mlp-awq-eq
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

- `PTQ_EQ_ENABLED=1` enables the new equalization path (default on).
- `PTQ_EQ_CALIB_TOKENS=262144` controls the calibration token budget.
- `PTQ_EQ_ALPHA_CANDIDATES=0.0,0.25,0.5,0.75,1.0` controls the exponent sweep.
- `PTQ_EQ_MAX_SCALE=4.0` bounds channel rescaling.

Minimal CPU smoke test:

```bash
SMOKE_TEST=1 python train_gpt.py
```

That smoke mode validates the candidate-specific export path on synthetic CPU tensors, but it does **not** validate real dataset loading, SentencePiece setup, or GPU training throughput.

## Main expected risks or tradeoffs

- The new equalization step only targets **MLP weights**, so attention-side outliers may still dominate the remaining quantization gap.
- Activation-aware scaling is calibrated on a small token subset; if those statistics are unrepresentative, the transformation may be neutral or mildly harmful.
- Export becomes a bit slower because each layer now runs a small alpha search before quantization.
- This is intentionally much cheaper than full AWQ / OmniQuant / SpinQuant, so the upside may be modest.

## Validation

Commands run for this candidate in this workflow:

```bash
python -m compileall candidates/202603261824_mlp-awq-eq/train_gpt.py
SMOKE_TEST=1 python candidates/202603261824_mlp-awq-eq/train_gpt.py
```

Outcomes:

- `compileall`: passed
- `SMOKE_TEST=1`: could not run in this workflow container because `torch` was not installed; after the smoke-path import fix, `numpy` and `sentencepiece` are no longer required just to reach the CPU smoke mode

If a lightweight GPU run is available later, start with the base recipe defaults and compare:

- `DIAGNOSTIC post_ema`
- `final_int6_roundtrip_exact`
- `final_int6_sliding_window_exact`
- `legal_ttt_exact`

against the March 23 record.
