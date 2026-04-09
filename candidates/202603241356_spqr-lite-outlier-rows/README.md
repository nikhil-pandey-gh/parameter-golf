# Candidate: SpQR-lite Outlier Rows

## Hypothesis

The current best 11-layer stack is still quantization-limited: the repo's strongest wins keep coming from better export formats, better averaging before export, and evaluation tricks rather than from radically different training dynamics. A tiny fp16 side-table for the *worst-quantized rows* should recover more of the float model than plain GPTQ-lite clip search alone, while staying inside the remaining artifact headroom.

Concretely, this candidate keeps the best current training architecture and changes only the post-training export path: after GPTQ-lite-style per-row clip search chooses the best int6 scale, it preserves a small fraction of the rows with the highest relative reconstruction error in fp16 and zeros those rows in the dense int6 tensor so the compressed artifact does not pay twice.

## Why this is promising for this repository

Repository evidence points in the same direction:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` shows the current SOTA was improved by **better post-training quantization**, not by a heavier training loop.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` is the clearest warning sign: much better float-model quality still collapsed back toward baseline after quantization.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` and `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` both argue that the leaderboard bottleneck is often the quantized artifact, not raw training loss.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md` shows that selective precision / byte allocation can be worth measurable BPB gains.

This candidate tries to spend bytes in the same spirit, but at a finer granularity: on the rows that quantize worst, not on whole tensors.

## Records or prior candidates that influenced it

### Direct base

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

### Key influencing records

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - Current best overall base.
  - Established GPTQ-lite clip search + EMA as a good export-time improvement.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - Confirms the 11-layer Partial-RoPE/LN-scale/XSA stack is the right architecture to keep stable while changing only quantization.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md`
  - Motivated treating bytes as a budget to spend where they matter most.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`
  - Reinforced that improving float quality alone is not enough if export quality lags.

### Prior candidates

- No `candidates/` directory existed when this candidate was created, so there were no prior candidate iterations to inherit from or avoid.

## External research that informed it

This is not a full reimplementation of AWQ or SpQR. Instead, it is a repo-sized adaptation of the same core idea: protect a tiny high-sensitivity subset while keeping the dense bulk low-bit.

- **AWQ** — Activation-aware Weight Quantization, arXiv:2306.00978
  - Key idea: a very small set of salient weights/channels dominates low-bit error.
  - Repo fit: points toward spending a small amount of high precision where it buys the most.
- **OWQ** — Outlier-Aware Weight Quantization, arXiv:2306.02272
  - Key idea: keep a structured sensitive subset in higher precision and quantize the rest.
  - Repo fit: closest conceptual match to this candidate's row side-table.
- **SpQR** — Sparse-Quantized Representation, arXiv:2306.03078
  - Key idea: isolate outliers that cause disproportionate quantization damage and store them separately.
  - Repo fit: motivates the "dense low-bit + sparse higher-precision correction" pattern used here.
- **LLM.int8()** — arXiv:2208.07339
  - Key idea: most values can be quantized cheaply if outliers are isolated into a higher-precision path.
  - Repo fit: another strong signal that outlier isolation is often enough.

## What changed versus the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four focused changes:

1. **SpQR-lite outlier-row export**
   - New hyperparameter: `OUTLIER_ROW_FRAC` (default `0.005`).
   - After GPTQ-lite chooses the best clip percentile for each matrix, the exporter computes per-row relative reconstruction error.
   - The worst rows are stored in fp16 side tensors (`*.outlier_rows` + `*.outlier_row_idx`).
   - Those same rows are zeroed in the dense int6 tensor so the compressed artifact does not redundantly store both versions.

2. **Dequantization restores side-table rows exactly**
   - During roundtrip load, the dense int6 tensor is reconstructed first.
   - Protected rows are then overwritten from the fp16 side-table.

3. **Candidate-directory defaults are actually self-contained**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root using `Path(__file__)`, so running from this candidate directory works without extra path rewriting.

4. **CPU-friendly import/attention fallback for smoke work**
   - `flash_attn_interface` import is optional.
   - A `scaled_dot_product_attention` fallback is available when FlashAttention is unavailable, which makes import-level CPU smoke work possible in principle.
   - `LATE_QAT_THRESHOLD` defaults to `0.0` here so the candidate isolates PTQ changes instead of relying on the compile-fragile late-QAT path from earlier records.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603241356_spqr-lite-outlier-rows

SEED=1337 \
OUTLIER_ROW_FRAC=0.005 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to be more conservative on artifact size, reduce `OUTLIER_ROW_FRAC` to `0.003` or `0.004`. If you want to test the edge of the byte budget, increase it carefully and inspect the final `Total submission size int6+...` log line.

## Main expected risks or tradeoffs

- **Artifact headroom risk**: protecting too many rows can erase the size benefit quickly. The default `0.005` is a conservative first guess, not a proven optimum.
- **Weight-only saliency**: this uses relative row reconstruction error from the weight tensor itself, not activation-aware calibration like AWQ.
- **Compression interaction risk**: zeroing protected rows in the dense tensor should help compression, but the net effect still depends on how the fp16 side-table compresses after `torch.save` + `zstd`.
- **No leaderboard run yet**: this candidate has only lightweight local validation in this workflow, not a full 8xH100 measurement.

## Validation run in this workflow

### Passed

```bash
python -m compileall candidates/202603241356_spqr-lite-outlier-rows/train_gpt.py
```

Outcome: passed.

### Attempted but not feasible here

I attempted a minimal CPU-only smoke path by importing the candidate module and exercising a tiny forward/quantize/dequantize roundtrip. That was blocked by the workflow environment rather than by the candidate code itself:

```bash
python3 - <<'PY'
import torch
import sentencepiece
PY
```

Outcome: failed because `/usr/bin/python3` in this environment does not have `torch` or `sentencepiece` installed, and shell-network installs are unavailable in this workflow.

That means this workflow can confirm syntax, path wiring, and code structure, but not execute the runtime path end-to-end without a GPU-ready Python environment.
