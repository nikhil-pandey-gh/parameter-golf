# AWQ-lite Int6 Clip Search

## Hypothesis

The March 22 `11L EMA + GPTQ-lite + warmdown3500` stack is already a strong non-TTT core model, so the next cheap win is likely in the export path rather than another architectural rewrite. This candidate replaces pure weight-MSE clip selection for int6 rows with an **activation-aware clip search** that scores reconstruction error using post-EMA calibration activations collected from a few short training batches.

## Why this is promising for this repository

- Several records show that **quantization quality is a first-order bottleneck** for Parameter Golf, not a minor cleanup step:
  - `2026-03-18_FP16Embed_WD3600` showed the tied embedding is highly quantization-sensitive.
  - `2026-03-19_WarmdownQuantization` improved scores mainly by making weights easier to quantize.
  - `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` got a real gain from smarter post-training clip search alone.
- The current SOTA `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` pushes architecture + evaluation harder, but it still uses the same basic int6 export idea. That makes export-aware improvements attractive because they should stack on strong training recipes later.
- This idea stays within the repo's constraints: no new infrastructure, no tokenizer change, no new artifact format, and no extra training-time backprop.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Quantization motivation:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`, `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
- **Current frontier reference:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **What not to repeat:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` documents that one late-QAT path never actually activated under `torch.compile`, so this candidate keeps the change squarely in post-training quantization.

## External research that informed it

- **GPTQ** (arXiv:2210.17323): motivates scoring quantization choices with a curvature/error proxy instead of naive rounding alone.
- **SmoothQuant** (arXiv:2211.10438): activation statistics are useful because they tell you where quantization difficulty actually lives.
- **AWQ** (arXiv:2306.00978): activation-aware saliency is a strong low-bit weight-only quantization signal.
- **QuaRot** (arXiv:2404.00456) and **SpinQuant** (arXiv:2405.16406): show that outlier-aware quantization keeps improving, but their rotation machinery is much broader than this repo needs for a first candidate.

This candidate takes the smallest repo-compatible slice of that literature: **use activation second moments to score int6 clip percentiles, while leaving the model and serialized format alone**.

## What changed versus the chosen base implementation

1. **Run-from-candidate defaults.** `DATA_PATH` and `TOKENIZER_PATH` now default relative to the repository root so `train_gpt.py` works when launched from this candidate directory.
2. **AWQ-style calibration knobs.** Added:
   - `AWQ_ENABLED`
   - `AWQ_CALIBRATION_BATCHES`
   - `AWQ_CALIBRATION_BATCH_TOKENS`
   - `AWQ_CLIP_PERCENTILES`
3. **Short post-EMA calibration pass.** After training finishes and EMA weights are applied, the script runs a few inference-only batches and collects per-input-channel second moments for large int6-targeted linear layers.
4. **Activation-aware int6 clip search.** The existing GPTQ-lite rowwise percentile sweep is preserved, but clip candidates are now ranked by **activation-weighted reconstruction error** instead of plain weight MSE.
5. **Everything else stays the same.** Architecture, optimizer split, warmdown, EMA/SWA behavior, zstd export, and evaluation flow are inherited from the March 22 base.

## How to run / evaluate

From this directory:

```bash
RUN_ID=awq_lite_int6 \
SEED=1337 \
AWQ_ENABLED=1 \
AWQ_CALIBRATION_BATCHES=8 \
AWQ_CALIBRATION_BATCH_TOKENS=131072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The default dataset and tokenizer paths resolve back to the repository `data/` directory.
- If your data lives elsewhere, override `DATA_PATH` and `TOKENIZER_PATH` exactly as in the root trainer.
- The script still performs its own post-training int6 roundtrip and sliding-window evaluation at the end.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604032149_awq-lite-int6/train_gpt.py` | Passed |
| Minimal runtime smoke test | Not run here: the current environment does not have `torch` or `flash_attn_interface`, so CUDA/FlashAttention startup could not be exercised safely |

## Main expected risks / tradeoffs

- The gain may be **small but real**: this is an export-path tweak, not a new architecture.
- The scoring still uses a **diagonal activation proxy**; full GPTQ/AWQ-style second-order or channel-rescaling methods could do better, but would require more code and more invasive graph changes.
- The calibration pass adds a bit of **export-time overhead** and some distributed reductions.
- If this helps on the March 22 base, the obvious next experiment is to port the same activation-aware scoring into the March 23 SOTA stack with LeakyReLU², TTT, and parameter banking.
