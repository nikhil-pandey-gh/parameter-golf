# AWQ-lite RowMix GPTQ

## Hypothesis

The current 11-layer stack is already strong enough that the remaining gap is increasingly a **post-training quantization** problem rather than a pure training-loss problem. This candidate keeps the proven 11L EMA + GPTQ-lite recipe, then makes export-time quantization more selective: it uses activation statistics from already-seen training batches to identify the most important projection rows and upgrades only those rows from int6-style clipping to int8-style clipping.

The expected win is a smaller quantization roundtrip penalty at nearly the same artifact size, especially in the MLP down-projection (`mlp.proj`) and attention output projection (`attn.proj`) where activation outliers are most likely to hurt low-bit export.

## Why this is promising for this repository

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that a **zero-training-cost export improvement** (`GPTQ-lite` clip search) was worth about `-0.0006 BPB`, which is large at the current frontier.
- `2026-03-19_WarmdownQuantization` explicitly argues that the dominant bottleneck is quantization quality, not just raw model quality.
- `2026-03-19_MixedQuant_Int6Int8_SlidingWindow` and `2026-03-18_FP16Embed_WD3600` both showed that **selective precision allocation** is more effective than treating every tensor the same.
- Repo review found no prior experiments using **AWQ**, **SmoothQuant**, **QuaRot**, or other activation-aware export schemes, so this is meaningfully different from the existing record line.

## Prior records and experiments that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - base implementation and the existing GPTQ-lite export path
2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - LeakyReLU(0.5)^2 MLP activation, which looked like the cleanest training-side improvement to carry forward
3. `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
   - repository evidence that quantization error dominates late-stage gains
4. `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
   - motivation for keeping precision selective instead of uniform

## External research that informed it

- **GPTQ**: <https://arxiv.org/abs/2210.17323>
  - strong evidence that careful one-shot weight quantization can preserve language-model quality
- **AWQ**: <https://arxiv.org/abs/2306.00978>
  - key idea: protect only the most salient weights/channels, and use activation statistics rather than weights alone to find them
- **SmoothQuant**: <https://arxiv.org/abs/2211.10438>
  - motivation for activation-aware, equivalent-transform-style handling of outliers
- **Scaling Law for Quantization-Aware Training**: <https://arxiv.org/abs/2505.14302>
  - recent evidence that outlier-heavy FC2-style layers benefit from mixed precision and that quantization error remains a central bottleneck

This candidate does **not** implement full AWQ channel rescaling. Instead, it adapts the same intuition to this repo's existing per-row GPTQ-lite exporter with a much smaller code change.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. **LeakyReLU(0.5)^2 MLP**
   - replaces ReLU^2 with the activation that powered the 2026-03-23 record
2. **AWQ-lite calibration cache**
   - stores a small ring buffer of already-seen training batches, so export can estimate activation magnitudes without touching new data
3. **Activation-aware row-mix quantization**
   - for targeted 2D weights (default: `mlp.proj`, `attn.proj`), compute:
     - best int6-style rowwise clipping,
     - best int8-style rowwise clipping,
     - an activation-weighted upgrade score
   - upgrade only the top fraction of rows (default `6.25%`) to int8-style clipping
   - keep the overall export format minimal: int8 payload tensors plus per-row scales, still compressed with zstd/zlib
4. **Portability and smoke-path improvements**
   - default data/tokenizer paths resolve from the repo root even when the script is run from the candidate directory
   - adds a CPU attention fallback and a `SMOKE_TEST=1` path for lightweight sanity checks in a provisioned environment

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202604040649_awq-rowmix-gptq
SEED=1337 \
AWQ_LITE_ENABLED=1 \
AWQ_ROW_UPGRADE_FRAC=0.0625 \
AWQ_TARGET_PATTERNS=mlp.proj,attn.proj \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The script now resolves `DATA_PATH` and `TOKENIZER_PATH` from the repository root by default, so it can run directly from this folder.
- `AWQ_ROW_UPGRADE_FRAC` is the main knob. Lower values are safer on artifact size; higher values are more aggressive.
- `AWQ_TARGET_PATTERNS` defaults to `mlp.proj,attn.proj`, but `mlp.proj` alone is the simplest next ablation if size or compression regress.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604040649_awq-rowmix-gptq/train_gpt.py
```

Outcomes:

- `compileall` passed for the existing Python entry points and the new candidate script.
- A CPU smoke execution was **not feasible in this container** because the runtime image does not include the repo's required Python packages (`torch`, `numpy`, `sentencepiece`). The candidate script includes a `SMOKE_TEST=1` path and CPU attention fallback for use in a properly provisioned environment.

## Main expected risks and tradeoffs

- **Artifact-size risk:** upgrading too many rows to int8-style clipping may reduce zstd compression gains even if model quality improves.
- **Calibration bias:** activation stats currently come from a small cache of recently seen train batches, which may be noisy or rank-0-specific.
- **Export-time cost:** rowwise clip search is heavier than the base exporter, especially if more target matrices are added.
- **Interaction risk:** LeakyReLU^2 may shift which layers benefit most from row upgrades, so the default target patterns may not be optimal.
