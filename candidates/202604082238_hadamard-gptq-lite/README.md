# Hadamard GPTQ-lite

## Hypothesis

The strongest clean next step is to keep the proven 11-layer EMA/XSA/Partial-RoPE/VE training stack from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, but reduce the remaining export-time quantization loss by rotating large int6 weight matrices into a friendlier basis before per-row GPTQ-lite clipping.

In this candidate, attention and MLP matrices are blockwise Walsh-Hadamard transformed on the input dimension immediately before int6 quantization, then inverse-transformed after dequantization for round-trip evaluation. The transform is parameter-free, deterministic, and only affects export quality, not the 10-minute training path.

## Why this is promising for this repository

The record history says the biggest durable wins after sliding-window eval came from improving the training stack and then shaving the last quantization gap:

- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` established the strong 11-layer/XSA/EMA/int6 recipe.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` improved it with Partial RoPE and LN scaling.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved it again with better export-time clipping.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` is currently strongest overall, but much of its gain comes from eval-time TTT and a more complex optimizer path rather than a simpler export improvement.

That makes export-time quantization the cleanest unexplored lever that still fits the repository's artifact and wall-clock constraints.

## External research that informed this candidate

- **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) showed that orthogonal rotations can remove outliers and make low-bit LLM quantization substantially easier, including essentially lossless 6-bit and 8-bit settings.
- **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)) showed that better rotations materially improve quantized model accuracy, and that random or learned rotations can outperform naive quantization by a large margin.
- **OptRot** ([arXiv:2512.24124](https://arxiv.org/abs/2512.24124)) argues that fusible rotations targeted at weight outliers can beat both plain Hadamard rotations and more expensive data-dependent methods for weight PTQ.
- **PolarQuant** ([arXiv:2603.29078](https://arxiv.org/abs/2603.29078)) is especially relevant here: it reports that Walsh-Hadamard rotation alone accounts for most of the quality improvement in its weight-only pipeline, which is exactly the lowest-complexity part that can be transplanted into this repository.

I did **not** implement learned rotations, calibration data, or extra quantizer infrastructure here. The candidate deliberately takes the smallest transplant that still matches the literature trend.

## Base implementation and what changed

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. Added a **blockwise Walsh-Hadamard preconditioner** for large 2D int6 matrices.
2. Chose the largest power-of-two block size up to `512` that cleanly divides the input dimension, which fits this stack's dominant `512`, `1024`, and `1536` widths well.
3. Stored the chosen rotation block size in quantization metadata so dequantization can invert the transform exactly.
4. Added a **FlashAttention → SDPA fallback** so the script can import and run non-FlashAttention forwards in lighter environments.
5. Kept the rest of the training recipe intentionally unchanged from the 2026-03-22 base.

## How to run

From this directory:

```bash
RUN_ID=hadamard_gptq_lite \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The new export knobs are enabled by default:

```bash
ROTATE_INT6=1
ROTATE_MIN_BLOCK=128
ROTATE_MAX_BLOCK=512
```

To ablate the new idea, set `ROTATE_INT6=0`.

## How to evaluate

This script follows the same pattern as the 2026-03-22 base:

- train under the 600s wall-clock cap,
- apply EMA weights,
- export a compressed int6 artifact,
- dequantize for round-trip evaluation,
- report standard and sliding-window validation metrics.

The log line to compare against prior records is:

```text
final_int6_sliding_window_exact val_loss:... val_bpb:...
```

## Validation run in this workflow

Commands run here:

```bash
python -m compileall candidates/202604082238_hadamard-gptq-lite/train_gpt.py
```

Outcome:

- `compileall` succeeded.
- A minimal CPU smoke test was **not feasible in this runner** because its available Python environments do not include `torch`, `numpy`, or `sentencepiece`, which this script needs even for import-time model construction.

## Main risks and tradeoffs

- Walsh-Hadamard rotation should reduce outlier-driven quantization loss, but it may also change value entropy in ways that help or hurt downstream compression.
- The best block size may not be `512`; `256` or more selective application could win.
- Learned or data-aware rotations from the literature could work better, but they would add more code and more tuning burden than this candidate is meant to introduce.
- The candidate intentionally avoids the heavier eval-time TTT path from the current top record, so its upside depends on quantization gains being real rather than merely plausible.
