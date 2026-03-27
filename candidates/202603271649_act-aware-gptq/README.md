# Activation-Weighted GPTQ-lite

## Hypothesis

The strongest low-risk gap left in this repository is still the **post-training quantization gap**, not the training stack itself.

This candidate keeps the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` architecture and export flow, but changes the int6 clip search so it no longer optimizes plain weight MSE only. Instead, it runs a tiny train-stream calibration pass after EMA is applied, estimates each large linear layer's input second moments, and uses those statistics to weight the per-row clip-percentile search. The goal is to spend the int6 error budget on channels the model actually uses most often.

## Why this is promising for this repository

Recent record history strongly suggests that export quality is still a first-order bottleneck:

- `2026-03-19_WarmdownQuantization` explicitly argues that quantization quality dominates many architecture tweaks.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved the record mostly with a better clip-search export path plus EMA.
- The latest best run `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` gets further gains from activation and TTT changes, but still inherits the same general quantized-export framing.

That makes **activation-aware export** a good fit: it is directly targeted at the remaining gap, costs almost nothing during training, and reuses the existing code paths rather than adding a new training subsystem.

## Prior records and experiments that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`

Key influence from earlier work:

- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` for the thesis that compression-aware training/export matters disproportionately at this scale.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` because the chosen base already inherits the 11-layer partial-RoPE/XSA/EMA trendline.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` as evidence that the current frontier is stacking small targeted gains onto the same overall recipe.

I also reviewed the non-record `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`. It is useful as a negative reminder that more invasive architectural changes can easily lose under a hard wall-clock budget.

## External research that informed it

This candidate is inspired by the same family of activation-aware post-training quantization ideas used in larger LLMs, but adapted to this repository's simpler mixed-int6 export flow.

- **GPTQ**: https://arxiv.org/abs/2210.17323
  - Core lesson: better quantization objectives matter, and simple one-shot weight quantization can preserve much more quality than naive clipping.
- **AWQ (Activation-aware Weight Quantization)**: https://arxiv.org/abs/2306.00978
  - Core lesson: weight importance should be judged through activations, not weights alone.
- **SmoothQuant**: https://arxiv.org/abs/2211.10438
  - Core lesson: activation statistics can be used offline to reduce quantization damage without changing the training dataset or model semantics.
- **SpinQuant**: https://arxiv.org/abs/2405.16406
  - Core lesson: post-training quantization still improves substantially when outlier handling becomes more activation-aware, even before changing the base model.

I considered more radical parameter-sharing ideas from ALBERT / MobileLLM-style literature, but repository evidence here points to quantization/export as the more reliable next lever under a 10-minute training budget.

## What changed versus the chosen base implementation

Compared with the `2026-03-22` base script, this candidate makes two focused changes:

1. **Activation-weighted int6 clip search**
   - Adds a small post-training calibration pass over training shards after EMA weights are applied.
   - Collects per-layer input second moments for the large attention and MLP linear layers.
   - Uses those statistics to weight the existing GPTQ-lite rowwise clip-percentile search, so reconstruction error on high-usage channels counts more.

2. **FlashAttention fallback for smoke validation**
   - If `flash_attn_interface` is unavailable, the script falls back to PyTorch `scaled_dot_product_attention`.
   - This is mainly to make lightweight import/forward validation practical without changing the intended CUDA path when FlashAttention is present.

Everything else is intentionally inherited from the strong `2026-03-22` base: 11 layers, partial RoPE, XSA on the deepest layers, SmearGate + BigramHash, VE128, EMA, warmdown, and mixed int6/int8 export.

## How this differs from existing records and candidates

There were no pre-existing `candidates/` in this repository when this directory was created.

This is **not** another architecture sweep and it is **not** a repeat of the existing GPTQ-lite record. The novelty is specifically that the clip selection is now **activation-weighted**, using a small calibration pass, instead of minimizing plain weight-space MSE only.

In other words: same strong backbone, new export objective.

## How to run or evaluate it

8xH100-style run from this directory:

```bash
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
ACT_AWARE_QUANT=1 ACT_AWARE_CALIBRATION_TOKENS=131072 ACT_AWARE_CALIBRATION_STEPS=4 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Single-GPU debugging is still possible with smaller batch settings, for example:

```bash
TRAIN_BATCH_TOKENS=131072 VAL_BATCH_SIZE=131072 \
ACT_AWARE_QUANT=1 ACT_AWARE_CALIBRATION_TOKENS=65536 ACT_AWARE_CALIBRATION_STEPS=2 \
python train_gpt.py
```

## Main expected risks or tradeoffs

- The gain may be small if the current GPTQ-lite rowwise percentile search is already close to optimal for this model.
- The extra calibration pass adds some post-training wall-clock overhead, though it is much cheaper than another full eval or any TTT scheme.
- This is **AWQ-inspired**, not full AWQ channel rescaling. The implementation intentionally stays minimal and may leave some activation-aware quantization gains on the table.
- Because the calibration uses training-stream statistics, the choice of calibration shard subset may matter slightly.

## Validation

Commands run from the repository root:

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603271649_act-aware-gptq/train_gpt.py`
- `python - <<'PY' ... PY` CPU import/forward/quantization smoke test in a `torch`-enabled environment

### Validation outcomes

- `compileall`: passed
- CPU smoke test: not runnable in this environment because `torch` is not installed on the runner
- Code review: completed; fixed the CPU GQA fallback to expand KV heads instead of relying on CUDA-only `enable_gqa`
