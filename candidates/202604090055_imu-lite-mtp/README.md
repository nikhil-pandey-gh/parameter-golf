# Candidate: IMU-lite Attention + Annealed MTP

## Hypothesis

The best next improvement is to keep the strongest clean 11-layer core stack from the 2026-03-22 GPTQ-lite/EMA record, then add **sample-efficiency improvements that cost little or nothing in the final artifact**:

1. **LeakyReLU(0.5)^2** in the MLP, which already improved the stronger 2026-03-23 stack.
2. **Per-head attention gating + value residuals**, inspired by IMU-1's small-model recipe.
3. **A single multi-token prediction (MTP) auxiliary head** whose loss weight decays with the LR warmdown and is **excluded from export**, so it helps training without spending artifact bytes.

## Why this is promising for this repository

Repository review showed a clear pattern:

- the strongest runs now cluster around **11L / seq2048 / MLPx3 / XSA / partial RoPE / LN scale / EMA / GPTQ-lite**;
- recent wins are coming from **small, composable gains** rather than wholesale rewrites;
- **training-only or tiny-parameter interventions** are especially attractive because the leaderboard is already very close to the 16 MB cap;
- naive recurrence/weight reuse looked risky here, while export-free auxiliaries and tiny control parameters remain underexplored.

This candidate therefore targets **sample efficiency under the same wallclock and artifact budget**, not a larger or slower model.

## Prior experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest clean pre-TTT stack in the repo review
  - already includes the mature 11L/XSA4/partial-RoPE/LN-scale/VE/EMA/GPTQ-lite recipe
- **LeakyReLU^2 port:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - showed a real gain from the activation swap on top of a stronger stack
- **Late-QAT caution:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - documents that compile-folded STE flags can silently turn late QAT into a no-op
- **Dead-end reminder:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - naive recurrence hurt badly there, so this candidate avoids a sharing-heavy depth-reuse bet

At repo-review time there were **no prior `candidates/` directories**, so this is the first candidate iteration in that namespace.

## External research that informed it

- **Better & Faster Large Language Models via Multi-token Prediction** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737))
  - argues that MTP improves sample efficiency by predicting multiple future tokens from a shared trunk
  - especially attractive here because extra heads can be kept **training-only**
- **IMU-1: Sample-Efficient Pre-training of Small Language Models** ([arXiv:2602.02522](https://arxiv.org/abs/2602.02522))
  - highlights **per-head gating, value residuals, LN scaling, and QK-norm** as part of a validated small-model recipe
  - this repo already had LN scaling and QK-norm-like attention normalization, so gating/value residuals were the clean missing pieces
- **FlexiGPT** ([arXiv:2501.14713](https://arxiv.org/abs/2501.14713))
  - useful as a reminder that weight sharing can extend small models, but given the repo's negative recurrence result I treated it as motivation for *parameter-aware design*, not as the change to ship first

## What changed vs. the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

| Change | Purpose |
|---|---|
| Script-relative default `DATA_PATH` and `TOKENIZER_PATH` | Lets the script run directly from the candidate directory |
| `LeakyReLU(0.5)^2` MLP | Ports the strongest simple activation win from the current SOTA record |
| `GATED_ATTENTION=1` by default | Adds per-head attention gating with tiny extra parameter cost |
| `VALUE_RESIDUAL=1` by default | Reuses early value information in later attention blocks |
| `MTP_NUM_HEADS=1`, `MTP_LOSS_WEIGHT=0.15` by default | Adds an export-free auxiliary next-next-token objective |
| LR-scaled MTP weight during training | Lets the auxiliary loss fade out during warmdown so final optimization focuses on the leaderboard objective |
| Export still strips `mtp_heads.*` | Keeps the auxiliary head out of the final artifact |
| `flash_attn_interface` fallback to SDPA | Improves portability for CUDA environments that do not have the external flash-attn interface package |
| `LATE_QAT_THRESHOLD=0.0` by default | Avoids depending on a historically suspect compile-folded late-QAT path |

## How to run or evaluate it

Run from the candidate directory so the script-relative defaults resolve to the repo's cached dataset and tokenizer:

```bash
cd candidates/202604090055_imu-lite-mtp
RUN_ID=imu_lite_mtp_seed1337 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The default run keeps the strong 2026-03-22 training recipe: 11 layers, seq2048, XSA on the last 4 layers, partial RoPE, LN scale, VE on layers 9/10, EMA, tight SWA, GPTQ-lite export, and sliding-window eval.
- The candidate-specific defaults are already baked into the script:
  - `MLP_NEGATIVE_SLOPE=0.5`
  - `GATED_ATTENTION=1`
  - `VALUE_RESIDUAL=1`
  - `MTP_NUM_HEADS=1`
  - `MTP_LOSS_WEIGHT=0.15`
  - `LATE_QAT_THRESHOLD=0.0`
- For ablations, the cleanest toggles are:

```bash
GATED_ATTENTION=0 VALUE_RESIDUAL=0 MTP_NUM_HEADS=0 MLP_NEGATIVE_SLOPE=0.0 LATE_QAT_THRESHOLD=0.0
```

That ablation gets close to the 2026-03-22 style trunk for the new features introduced here, but it still intentionally keeps late QAT disabled because earlier repo review found the compile-folded late-QAT path to be unreliable.

## Main expected risks and tradeoffs

- **Step-time risk:** even one MTP head adds extra logits/CE work, so any gain must outweigh a possible small throughput loss.
- **Interaction risk:** value residuals and attention gating may fight XSA or VE instead of stacking cleanly.
- **Objective mismatch risk:** MTP is helpful early in training, but can over-steer late optimization; that is why this candidate decays the MTP weight with the LR schedule.
- **QAT uncertainty:** I disabled late QAT by default rather than rely on a path that earlier records already flagged as compile-sensitive.
- **Portability vs benchmark parity:** the SDPA fallback is only meant for CUDA environments that lack `flash_attn_interface`; benchmark runs should still use the FlashAttention path when available.

## Validation

Commands run in this workflow:

1. Syntax check:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604090055_imu-lite-mtp/train_gpt.py
```

Outcome: **success**.

2. Runtime dependency check for a CPU smoke test:

```bash
python - <<'PY'
import importlib.util
for name in ('torch', 'sentencepiece', 'flash_attn_interface'):
    print(f'{name}={bool(importlib.util.find_spec(name))}')
PY
```

Outcome:

```text
torch=False
sentencepiece=False
flash_attn_interface=False
```

So a real CPU startup smoke test was **not feasible in this workflow environment**: the required runtime dependencies are absent, and this training script still requires CUDA even when the external flash-attn interface is missing.
