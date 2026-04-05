# 202604051327_lsq-clipqat

## Hypothesis

The best non-TTT record stack in this repo is already strong architecturally, so the next gain is more likely to come from **shrinking the float-to-int6 export gap** than from changing the backbone again. This candidate keeps the proven 11-layer/XSA/partial-RoPE/VE/bigram recipe, then replaces heuristic late quantization with a **small reserved-budget learned clip calibration pass** that learns per-row int6 clipping on frozen EMA weights before export.

I also fold in the now-proven **LeakyReLU(0.5)^2** MLP activation from the current overall best record, because it is a low-cost architectural improvement that composes naturally with the compression-focused change.

## Why this looks promising here

Repository evidence points in the same direction:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed that a better export heuristic alone was worth a measurable gain over the same architecture.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` documented that the earlier late-QAT path was effectively dead code under `torch.compile`, so there is still room for a real quantization-aware finish.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` showed that LeakyReLU(0.5)^2 is a real win on top of the modern stack.

Taken together, that suggests the right next bet is **keep the good 11L stack, improve the export step, and only spend a tiny extra wallclock budget on learning clipping instead of searching it**.

## Prior repo runs that influenced this candidate

| Run | Borrowed idea |
|---|---|
| `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` | Base implementation, EMA + GPTQ-lite + warmdown3500 + partial RoPE/XSA/VE/bigram stack |
| `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` | Motivation to replace broken late-QAT with a compile-safe quantization finish |
| `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` | LeakyReLU(0.5)^2 MLP activation |

## External research that informed it

1. **LSQ — Learned Step Size Quantization** (Esser et al., 2020): https://arxiv.org/abs/1902.08153  
   Main takeaway used here: learn quantizer parameters with gradient scaling instead of fixing them heuristically.
2. **OmniQuant** (Shao et al., 2024): https://arxiv.org/abs/2308.13137  
   Main takeaway used here: learned clipping is a strong post-training quantization lever.
3. **SmoothQuant** (Xiao et al., 2023): https://arxiv.org/abs/2211.10438  
   Included mainly as supporting evidence that outlier management matters for low-bit export, even though this candidate stays weight-only and much simpler.
4. **AdaRound** (Nagel et al., 2020): https://arxiv.org/abs/2004.10568  
   Relevant as prior art for small calibration-style finishing passes instead of retraining the whole model.

## What changed versus the chosen base

Chosen base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **LeakyReLU(0.5)^2 MLP**
   - Swaps `relu^2` for `leaky_relu(..., 0.5)^2`.
2. **Reserved calibration budget**
   - Reserves `LEARNED_CLIP_SECONDS` from the wallclock cap so the learned clip pass can still fit within the same overall timing budget.
3. **Post-EMA learned clip calibration**
   - Freezes model weights.
   - Optimizes only per-row clip multipliers for `CastedLinear` weights on a few train batches.
   - Uses LSQ-style gradient scaling on those clip multipliers.
4. **Learned int6 export**
   - If learned row clips are available, export uses them directly for int6 per-row scales.
   - If calibration does not run (for example, budget exhausted), the script falls back to the original GPTQ-lite percentile search.
5. **Dead late-QAT path removed**
   - The candidate does not rely on runtime toggling inside the compiled training graph.
6. **Repo-root-relative defaults**
   - Dataset and tokenizer defaults resolve from the repository root, so the script can be launched directly from this candidate directory.

## How to run

From this candidate directory:

```bash
SEED=1337 \
MLP_NEGATIVE_SLOPE=0.5 \
LEARNED_CLIP_ENABLED=1 \
LEARNED_CLIP_STEPS=8 \
LEARNED_CLIP_BATCH_TOKENS=131072 \
LEARNED_CLIP_LR=0.003 \
LEARNED_CLIP_SECONDS=15 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

- Disable the new quantizer finish: `LEARNED_CLIP_ENABLED=0`
- Change calibration budget: `LEARNED_CLIP_SECONDS=<seconds>`
- Change activation back to baseline: `MLP_NEGATIVE_SLOPE=0.0`
- `LEARNED_CLIP_BATCH_TOKENS` must be a positive multiple of `TRAIN_SEQ_LEN`

## Main risks / tradeoffs

1. Reserving calibration time slightly reduces main-training time, so the learned export needs to win back more than the lost steps cost.
2. The learned clip pass optimizes a proxy on a few train batches; that may overfit or simply not outperform the GPTQ-lite percentile search.
3. The activation change and export change are coupled here, so the first ablation to run should be `LEARNED_CLIP_ENABLED=0` to isolate the calibration contribution.
4. This is intentionally a minimal compression-aware extension, not a full OmniQuant/AdaRound implementation, so upside may be capped.

## Validation

Commands run (from repo root):

```bash
python -m compileall candidates/202604051327_lsq-clipqat/train_gpt.py
```

Outcome:

- `compileall` passed.
- A CPU smoke launch was **not** feasible without changing repo behavior: this training path imports CUDA/FlashAttention-specific components and hard-requires CUDA for execution, so a meaningful CPU start test would need a separate non-repo code path.
