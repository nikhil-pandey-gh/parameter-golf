# 202603241902_awq-gptq-lite

## Hypothesis

The current best stack in this repository is already very strong architecturally: 11 layers, MLP 3x, U-Net skips, XSA on deep layers, Partial RoPE, LN scaling, EMA, and GPTQ-lite export. The remaining gap appears to be mostly in **post-training compression quality**, so this candidate tests whether a small **activation-aware per-linear input scaling** pass can reduce the int6 reconstruction error further than clip search alone.

Concretely, the candidate keeps the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` training recipe intact, but adds an **AWQ-lite** export pass:

- collect a few batches of activation RMS statistics after EMA,
- search a tiny alpha grid per large attention/MLP matrix,
- insert a per-input-channel scale vector into each quantized linear,
- divide weight columns by that scale before GPTQ-lite/int6 export,
- and use the scale at inference time so the full-precision function is preserved.

The goal is to protect high-activation channels without paying for mixed-precision weight blocks.

## Why this is promising for this repository

Repository evidence points to quantization as the main remaining bottleneck:

- the non-record 4-hour run reached much better full-precision quality but still fell back to `1.2074` after quantization,
- the leaderboard improvements repeatedly came from techniques that made weights compress better rather than from large architecture changes alone,
- the current best run already uses GPTQ-lite clip search and EMA, suggesting the next win is likely a better **weight-only export transform** rather than another broad training rewrite.

Activation-aware scaling is attractive here because it is:

- local to the existing `CastedLinear` modules,
- very small in artifact overhead (just per-input-channel scale vectors),
- compatible with the current mixed int6/int8 export path,
- and much easier to integrate into this codebase than full second-order GPTQ or learned rotation methods.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Most relevant predecessors:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - established Partial RoPE + layerwise LN scaling as strong zero-parameter improvements.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - established XSA-on-deep-layers + EMA as a strong 11-layer base.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - showed that selective deep-layer XSA is a good cost/benefit trade.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - strong evidence that quantization quality, not just training quality, is a major bottleneck.

## External research that informed it

- **AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration** (`arXiv:2306.00978`)
  - motivates protecting weight channels according to activation statistics rather than weight magnitude alone.
- **SmoothQuant** (`arXiv:2211.10438`)
  - motivates equivalent channelwise rescaling transforms that shift quantization difficulty without changing the represented function.
- **GPTQ** (`arXiv:2210.17323`)
  - motivates improving post-training weight-only quantization quality rather than relying only on naive round-to-nearest clipping.
- **SpinQuant** (`arXiv:2405.16406`)
  - supports the broader idea that lightweight, accuracy-preserving transforms before quantization can materially improve low-bit reconstruction.

This candidate intentionally chooses the **smallest implementation from that family** that fits the current repo: per-linear local scaling, not learned rotations or a full GPTQ solver.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate adds:

1. **Activation-aware input scaling for quantized linears**
   - `CastedLinear` now carries a persistent `input_scale` buffer.
   - During normal training this does nothing.
   - Before export, the script estimates activation RMS values and chooses a small per-layer scale vector for large attention/MLP weights.
   - Quantized evaluation enables those scales so the transformed model preserves the original function while making the stored weights easier to quantize.

2. **AWQ-lite scale search on top of existing GPTQ-lite clip search**
   - The base script already searches several clip percentiles per row.
   - This candidate additionally searches a tiny alpha grid (`0.0, 0.5, 1.0` by default) for the activation-aware input scale.
   - The selection objective is an activation-weighted reconstruction error proxy, which is closer to output error than plain weight MSE.

3. **Late-QAT recompilation hook**
   - The current repo history suggests late QAT may be vulnerable to compile-time dead-code elimination when the flag flips after the model has already been compiled.
   - This candidate recompiles the training model once when late QAT activates, so the branch change is not silently ignored.

4. **Candidate-local ergonomics for validation**
   - default dataset/tokenizer paths are resolved relative to the repository root, so the script can be launched from this candidate directory,
   - FlashAttention import now falls back to PyTorch SDPA when unavailable,
   - and `SMOKE_TEST=1` runs a synthetic CPU-only path that exercises model construction, activation-aware export, quantize/dequantize roundtrip, and logits generation.

## How to run / evaluate it

From the repository root:

```bash
cd candidates/202603241902_awq-gptq-lite
RUN_ID=awq_gptq_lite torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional knobs for the new quantization pass:

```bash
AWQ_ENABLED=1 \
AWQ_CALIBRATION_BATCHES=4 \
AWQ_CALIBRATION_BATCH_TOKENS=131072 \
AWQ_CALIBRATION_SEQ_LEN=1024 \
AWQ_ALPHA_GRID=0.0,0.5,1.0 \
AWQ_SCALE_LIMIT=4.0
```

Synthetic smoke path (requires the repository Python dependencies to be installed):

```bash
cd candidates/202603241902_awq-gptq-lite
SMOKE_TEST=1 python train_gpt.py
```

## Validation run for this candidate

Validated in this workflow environment with lightweight checks only:

```bash
python -m compileall candidates/202603241902_awq-gptq-lite/train_gpt.py
```

Outcome:

- **Passed**.

Attempted CPU smoke check:

```bash
SMOKE_TEST=1 python candidates/202603241902_awq-gptq-lite/train_gpt.py
```

Outcome:

- **Not feasible in this workflow environment** because the repository runtime dependencies needed by the script were not installed (`torch`, `numpy`, and `sentencepiece` were all missing).
- The candidate still includes the `SMOKE_TEST=1` path so it can be exercised in a properly provisioned environment.

## Main expected risks / tradeoffs

- The activation statistics are collected from a very small calibration sample; the chosen scales may be noisy.
- The new per-linear input-scale vectors slightly increase artifact size and add a small inference-time multiply.
- Recompiling once when late QAT activates is intended to fix a real risk from the current stack, but it may add some late-run overhead.
- The gain may be modest if GPTQ-lite + EMA already removed most of the easy quantization error.
- More ambitious transform-based quantization ideas (for example learned rotations) may still outperform this simpler AWQ-lite variant, but they would require substantially more infrastructure than this repository currently uses.
