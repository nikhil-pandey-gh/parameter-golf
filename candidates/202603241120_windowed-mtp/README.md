# Candidate: windowed MTP on the 11L EMA + GPTQ-lite base

## Hypothesis

Add a **train-only multi-token prediction (MTP) auxiliary loss** to the current best 11-layer recipe, but **window it to the middle of training**: ramp it in after startup, weight nearer horizons more heavily, then fade it out before the late-QAT / hardest warmdown phase.

The bet is that MTP improves sample efficiency and representation quality early enough to help final `val_bpb`, while fading it late preserves the compression-aware behavior that has driven the recent leaderboard gains. Because the auxiliary heads are excluded from export, the idea should cost **training compute, not artifact bytes**.

## Why this is promising for this repository

The strongest recent records already look close to saturated on simple export tweaks alone:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` pushed the current best stack with EMA + GPTQ-lite + warmdown tuning.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` showed that zero-byte architectural refinements like Partial RoPE and LN scaling still matter.
- Earlier 11-layer records repeatedly improved by better use of the same 16MB budget rather than by longer training alone.

This codebase already contains dormant MTP support in the best script, but prior records kept it off (`MTP_NUM_HEADS=0`). That makes MTP a good next candidate: it is **new relative to the repo history**, already fits the existing script structure, and can be tried without rewriting the export stack.

## Prior records that influenced this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best current recipe: 11L, XSA4, Partial RoPE, LN scale, EMA, GPTQ-lite, VE, SmearGate, BigramHash.
- **Supporting predecessor:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Important warning that compile-time constant folding can silently disable intended training behavior; this candidate avoids that pattern for the MTP schedule.
- **Evidence that MTP was not previously explored in records:** `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md`
  - Its published config explicitly leaves `MTP_NUM_HEADS=0`.

## External research that informed the idea

- **Gloeckle et al., “Better & Faster Large Language Models via Multi-token Prediction”** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737))
  - Shows that predicting multiple future tokens as an auxiliary task can improve sample efficiency and downstream capability.
- **DeepSeek-V3 Technical Report** ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
  - Uses a multi-token prediction objective in a modern large-scale recipe, reinforcing that MTP is practical in competitive training stacks.
- **PTQ1.61** ([arXiv:2502.13179](https://arxiv.org/abs/2502.13179))
  - Motivated considering more aggressive mixed-bit export ideas, but those would require more new compression infrastructure than this candidate. I kept the change focused on a lower-risk training-only improvement instead.

## What changed versus the chosen base implementation

This candidate starts from the current best record script and makes four targeted changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.10`

2. **Weight nearer horizons more heavily**
   - Add `MTP_HORIZON_DECAY` (default `0.65`) so the first future token gets more weight than the second.

3. **Window the auxiliary loss**
   - Add `MTP_WARMUP_STEPS`, `MTP_FADE_START_SCALE`, `MTP_FADE_END_SCALE`, and `MTP_FINAL_SCALE`.
   - MTP ramps in over the first 500 steps, stays active through the high-LR mid-training region, then fades out as the LR scale enters warmdown.
   - Default fade window: start at LR scale `0.30`, reach final scale `0.0` by `0.12`, so MTP is mostly gone before the strongest quantization-sensitive tail.
   - When the schedule reaches zero, training switches back to the plain next-token forward path so the late phase really skips MTP compute instead of merely zeroing the loss weight.

4. **Make the MTP schedule runtime-controlled**
   - The per-step MTP multiplier is passed into `forward(...)` as a tensor instead of relying on a mutable class attribute.
   - This is deliberate: the previous record discovered that `torch.compile` could constant-fold control flags and accidentally disable intended behavior.

The export path still excludes `mtp_heads`, so the submission artifact remains driven by the base model, not by the auxiliary heads.

## Running from the candidate directory

This script is set up so its default dataset and tokenizer paths resolve relative to the repository root even when run from this candidate directory.

From the repo root:

```bash
cd candidates/202603241120_windowed-mtp
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
MTP_NUM_HEADS=0
MTP_LOSS_WEIGHT=0.08
MTP_FADE_START_SCALE=0.25
MTP_FADE_END_SCALE=0.10
DATA_PATH=/path/to/fineweb10B_sp1024
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model
```

## Validation

Commands run for this candidate:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603241120_windowed-mtp/train_gpt.py
```

Outcomes:

- `python -m compileall train_gpt.py train_gpt_mlx.py data` ✅
- `python -m compileall candidates/202603241120_windowed-mtp/train_gpt.py` ✅

CPU smoke test status:

- A full runtime smoke test was **not feasible in this environment**. The training script depends on CUDA / FlashAttention (`flash_attn_interface`) and the challenge dataset layout, so a reliable CPU-only start test here would not reflect the real execution path.

## Expected risks / tradeoffs

- **Wallclock risk:** MTP adds extra logits work during training, so the 600s budget may buy fewer optimizer steps.
- **Schedule sensitivity:** if MTP stays on too late, it could hurt the final quantization-aware tail instead of helping.
- **Head count sensitivity:** two horizons may be too much or too little for this tiny-vocab / fixed-time regime.
- **No artifact benefit:** unlike a new export trick, this only helps if the training efficiency gain is real.
