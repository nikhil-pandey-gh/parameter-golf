# Activation-Aware Int6 Clip Search

## Hypothesis

The current 11-layer EMA + GPTQ-lite stack is already very strong before export, so the next cheap gain is likely to come from **better matching the int6 quantizer to real layer inputs** rather than from another training-side architectural change. If we choose per-row int6 clip thresholds using a short calibration pass over training tokens, post-quant reconstruction should track model behavior better than the current weight-only clip search and slightly improve final validation BPB without slowing training.

## Why this is promising for this repository

- Recent winning records increasingly come from **quantization/export quality**, not wholesale backbone changes.
- The strongest non-TTT record already improved by about **-0.0006 BPB** just from better post-training clip selection (`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`).
- The repository's best backbone already has a small artifact margin under 16MB, so a **calibration-only export improvement** is attractive because it adds almost no model bytes and does not consume training throughput.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
   - strongest non-TTT base
   - showed that better row-wise clip search stacks cleanly on the 11L EMA/XSA/Partial-RoPE backbone
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
   - confirmed that the best backbone shape is still 11L + XSA4 + Partial RoPE + LN scale
3. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
   - established EMA, XSA, MLP3x, and WD=0.04 as the right core stack
4. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
   - showed there is still leaderboard headroom, but at much higher implementation complexity
5. `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
   - documented that depth recurrence was a poor trade under wallclock constraints, which helped rule out a more invasive recurrent candidate

## External research that informed it

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978))  
  Motivated using activation statistics, not just weight magnitudes, to decide which channels matter most for low-bit weight quantization.
- **SmoothQuant** ([arXiv:2211.10438](https://arxiv.org/abs/2211.10438))  
  Reinforced the idea that offline activation calibration can move quantization difficulty away from sensitive directions without changing training.
- **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456))  
  Strengthened the general theme that outlier management at export time is a high-leverage path for quantized LLM quality.
- **SASQ: Static Activation Scaling for Quantization-Aware Training in Large Language Models** ([arXiv:2512.14481](https://arxiv.org/abs/2512.14481))  
  Suggested that static, calibration-driven handling of activation outliers can outperform heavier quantization schemes.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate keeps the training backbone unchanged and only adds a calibration-aware export path:

1. **Short activation calibration pass after EMA**
   - runs a few lightweight forward-only batches over training shards
   - collects per-input-channel second moments for every `CastedLinear`
2. **Activation-aware per-row int6 clip search**
   - reuses the existing percentile grid search idea
   - chooses the best clip percentile per row using **activation-weighted output error** instead of plain weight reconstruction error
3. **No new runtime dependencies**
   - still exports the same int6/int8 mixed state dict format
   - no extra files beyond this `README.md` and `train_gpt.py`

## How to run

From this candidate directory:

```bash
RUN_ID=actaware_int6 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs for the new idea:

```bash
ACT_AWARE_QUANT=1
CALIBRATION_STEPS=8
CALIBRATION_BATCH_TOKENS=131072
CLIP_SEARCH_PERCENTILES=0.9990,0.9995,0.9999,0.99995,1.0
```

## Validation

- From the repository root: `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604012315_actaware-int6/train_gpt.py` - succeeded
- A CPU-only runtime smoke test was **not feasible** in this environment because this script hard-requires CUDA and `flash_attn_interface`, and the candidate intentionally preserves that GPU execution path from the base record.

## Main expected risks / tradeoffs

1. The calibration pass adds a small amount of post-training time, so the gain has to outweigh that extra complexity.
2. Activation-weighted clip search may overfit the sampled train batches if the calibration set is too small or too narrow.
3. The benefit may be marginal if the existing GPTQ-lite percentile search is already capturing most of the available int6 improvement on this backbone.
