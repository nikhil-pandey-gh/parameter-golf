# Late LSQ/QDrop QAT on the 11L GPTQ-lite base

## Hypothesis

The strongest remaining bottleneck in this repo is the **exported quantization gap**, not the pre-quant model. A compile-safe late QAT phase that learns per-row int6 step sizes during the last part of warmdown should reduce the post-quant BPB penalty more reliably than the earlier dead-code late-QAT attempt, while keeping the artifact format and inference path unchanged.

## Why this is promising here

- The repo's record history repeatedly shows that compression-aware changes matter as much as, or more than, raw pre-quant quality.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly tried late QAT, but its README documents that `torch.compile` constant-folded the class flag so the fake-quant path never activated.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed there was still measurable headroom in the export path via better post-training clipping alone.

## Prior repo work that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the direct base implementation.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` motivated the late-QAT retry, specifically because it exposed the compile-time bug.
3. `records/track_10min_16mb/2026-03-19_WarmdownQuantization/` reinforced the broader idea that training for the export path is often the highest-leverage move in this challenge.

## External research

- **OmniQuant** (learnable weight clipping / differentiable quantization calibration): <https://arxiv.org/abs/2308.13137>
- **QDrop** (randomly keep some full-precision paths during quantization training for flatter low-bit solutions): <https://arxiv.org/abs/2203.05740>
- **LSQ** (learned step size quantization): <https://arxiv.org/abs/1902.08153>
- **LSQ+** (better initialization and trainable quantizer parameters): <https://arxiv.org/abs/2004.09576>

This candidate is intentionally a **lite** adaptation of those ideas: learn only per-row weight scales during a late QAT phase, keep the repo's existing GPTQ-lite export search, and avoid broader activation/architecture surgery.

## What changed vs the base implementation

- Added a training-only `LearnedPerRowQAT` module for every large `CastedLinear` matrix and for the tied token embedding.
- Swapped the old class-level late-QAT flag for a **compile-safe enable-and-recompile transition** when the LR scale falls below `LATE_QAT_THRESHOLD`.
- Learned per-row scales with an LSQ-style straight-through quantizer and a small dedicated Adam optimizer (`QAT_LR`).
- Added row-wise QDrop-style stochastic bypass (`QAT_QDROP`) during the late fake-quant phase.
- Stripped all training-only quantizer parameters from the exported artifact, just like the existing code already strips training-only MTP heads.
- Updated default data/tokenizer paths so the script can be launched directly from this candidate directory without first changing back to the repo root.

## How to run

From this directory:

```bash
RUN_ID=late_lsq_qdrop \
VOCAB_SIZE=1024 \
QAT_ENABLED=1 \
QAT_BITS=6 \
QAT_LR=0.002 \
QAT_QDROP=0.1 \
LATE_QAT_THRESHOLD=0.2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate defaults resolve `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so they should work from `candidates/202604041744_late-lsq-qdrop/` as-is if the standard dataset/tokenizer are present under `data/`.

## Evaluation notes

- Export format is unchanged: the final submission still uses the existing mixed int6 + GPTQ-lite + zstd roundtrip path.
- The late QAT phase is training-only; its learned scale parameters are excluded from the final artifact.
- Sliding-window evaluation remains the same as the base branch.

## Main risks / tradeoffs

- Recompiling the model when late QAT starts costs wall-clock time; if the gain from better quantization robustness is too small, the lost steps could erase it.
- Learned scales may reduce the roundtrip gap but slightly hurt pre-quant quality if `QAT_LR` or `QAT_QDROP` are poorly tuned.
- This candidate targets the export bottleneck, not the recent eval-time TTT gains, so it is best compared to the repo's strong **pre-TTT** bases first.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604041744_late-lsq-qdrop/train_gpt.py` | Passed |
| `python -m compileall candidates/202604041744_late-lsq-qdrop/train_gpt.py` | Passed |
| Minimal CPU smoke with a stubbed FlashAttention module | Blocked on this runner because the declared repo dependencies (`torch`, `numpy`, `sentencepiece`) are not installed here |
