# AWQ-lite activation-aware channel scales

## Hypothesis

The current record line is already strong in fp/bf16 training, so the next cheap win is more likely to come from **better post-training low-bit deployment** than from another broad architecture change. This candidate adds a lightweight **AWQ-lite calibration pass**: measure per-input-channel activation magnitudes on a few train batches, derive a small scale vector for each large linear layer, fold those scales into the stored weights, and then reuse the existing GPTQ-lite mixed-int6 export path.

If the scales smooth outlier-heavy columns before quantization, the final compressed artifact should preserve more of the EMA model's quality at almost no training-time cost and only a small artifact-size overhead.

## Why this is promising for this repository

Recent repo history shows that most of the durable improvements came from **compression-aware decisions** rather than from radical new model families:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

The pattern is consistent: once the 11-layer/XSA/partial-RoPE stack was in place, gains mostly came from **quantization robustness, averaging, and eval-time deployment details**. There was **no existing `candidates/` directory** during review, so this does not duplicate a prior candidate iteration.

## Prior repo work that influenced this candidate

1. **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
   This was the strongest clean non-TTT springboard and already had the right structure for a post-training export change.
2. **Supporting signal:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`  
   Confirmed that the architecture was already near a strong local optimum, so a deployment-side idea made more sense than redoing the stack.
3. **Supporting signal:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`  
   Showed that remaining gains exist, but many of them come with extra evaluation complexity. This candidate intentionally keeps evaluation simple and attacks the artifact instead.

## External research that informed it

- **AWQ** — Lin et al., 2023, <https://arxiv.org/abs/2306.00978>  
  Activation-aware weight quantization preserves the most important channels instead of treating every weight column equally.
- **OmniQuant** — Shao et al., 2023, <https://arxiv.org/abs/2308.13137>  
  Reinforces the value of a short calibration-time transform instead of more full retraining.
- **QuaRot** — Ashkboos et al., 2024, <https://arxiv.org/abs/2404.00456>  
  Motivated the broader idea of smoothing outliers before low-bit export.
- **SpinQuant** — Liu et al., 2024, <https://arxiv.org/abs/2405.16406>  
  Strong evidence that low-bit quality is often bottlenecked by outlier structure, though its full rotation-learning recipe is more invasive than needed here.
- **A Survey on Transformer Compression** — Tang et al., 2024, <https://arxiv.org/abs/2402.05964>  
  Useful overview supporting quantization-first interventions for compact transformer deployments.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Adds an **AWQ-lite calibration pass** after EMA application and before export.
2. Stores a small per-layer `awq_scale` buffer on each `CastedLinear`.
3. **Folds** those scales into the corresponding weight matrices once, then enables input rescaling at inference/export time.
4. Reuses the existing **mixed int6/int8 GPTQ-lite quantizer** unchanged after the equalization step.
5. Adds a **FlashAttention fallback** to PyTorch SDPA so the script can still execute when `flash_attn_interface` is unavailable.
6. Adds a **CPU-only `SMOKE_TEST=1` path** that validates forward/backward plus AWQ-lite export/dequantize without dataset or GPU infrastructure.
7. Changes the default `DATA_PATH` and `TOKENIZER_PATH` to be **repo-root-relative**, so the script works when launched from this candidate directory.

## How to run

From this candidate directory:

```bash
SEED=1337 \
AWQ_ENABLED=1 AWQ_ALPHA=0.5 AWQ_CALIBRATION_BATCHES=8 AWQ_BATCH_TOKENS=131072 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Quick CPU smoke test:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run in this workflow

Executed from `candidates/202604091858_awq-lite-channel-scales/` using an isolated virtualenv because the base runner did not have the repo dependencies installed:

1. `/tmp/gh-aw/agent/pg-venv/bin/python -m compileall train_gpt.py`  
   **Passed**
2. `SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python train_gpt.py`  
   **Passed** with:
   - `smoke_train_step:1/2 loss:4.8460`
   - `smoke_train_step:2/2 loss:4.8497`
   - `awq:calibrated modules:26 alpha:0.50 batches:8 batch_tokens:131072`
   - `smoke_roundtrip_loss:4.8512`

## Main risks and tradeoffs

- This is a **heuristic AWQ-lite** implementation, not the full saliency search from the paper, so the gains may be smaller or more seed-sensitive.
- The added `awq_scale` buffers slightly increase artifact size; if the margin to 16MB gets tight, they may need lower precision or narrower targeting.
- The calibration batches come from the train stream, so poor batch choice could make some layers better and others worse.
- This candidate intentionally avoids a heavier rotation-based path such as SpinQuant/QuaRot; if the heuristic scales help only marginally, that is the next obvious follow-up.
