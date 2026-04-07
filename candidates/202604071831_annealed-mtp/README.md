# Annealed 1-step MTP on the current best stack

## Hypothesis

The current frontier in this repo is already very close to the 16 MB cap, so the next gain is more likely to come from **better sample efficiency per training step** than from adding more stored weights. A **single auxiliary multi-token prediction (MTP) head** should improve early training efficiency and induction-style behavior, but it should **fade out during warmdown** so the trunk can re-specialize for next-token prediction before export and quantization.

## Why this is promising for this repository

- The repo's winning trend is to stack cheap quality improvements on top of a strong 11-layer low-bit backbone: XSA, partial RoPE, LN scale, EMA/SWA, LeakyReLU(0.5)^2, and legal TTT.
- The best overall record (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) already contains latent MTP support **and already excludes MTP heads from the exported artifact**, which makes MTP unusually attractive here: it can improve training while adding **zero final artifact bytes**.
- Repo evidence also argues against chasing more effective depth via recurrence/depth reuse under a fixed 10-minute wallclock budget, so a training-only auxiliary loss is a better fit than another compute-heavy architectural expansion.

## Prior repo work that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — strongest overall stack and the direct code base for this candidate.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — strongest training-only base and a reminder that small last-mile improvements matter.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` — cautionary example that `torch.compile` can silently freeze feature toggles when they live in static class attributes.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` and `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` — both reinforced that fixed-wallclock throughput matters and that naive recurrence/depth tricks are easy to lose on.

There were no prior `candidates/` directories to avoid or extend.

## External research

- **Gloeckle et al., 2024 — "Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`): argues that predicting multiple future tokens from a shared trunk improves sample efficiency and helps induction-style behavior.
- **Mehra et al., 2025 — "On multi-token prediction for efficient LLM inference"** (`arXiv:2502.09419`): shows that hidden states are strongly specialized for next-token prediction, so MTP can interfere if left unconstrained.
- **Gerontopoulos et al., 2025 — "Multi-Token Prediction Needs Registers"** (`arXiv:2505.10518`): supports keeping the extra MTP machinery lightweight and close to the original next-token objective.

This candidate takes the simplest repo-compatible slice of that literature: **1-step MTP early, annealed away late**.

## What changed vs. the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. **Enabled 1-step MTP by default** with `MTP_NUM_HEADS=1`.
2. **Added an annealed MTP loss schedule**:
   - warm up over the first `250` steps,
   - keep the full auxiliary weight through most of training,
   - linearly fade it out once LR scale falls from `0.30` to `0.05`.
3. **Passed the MTP weight as a runtime tensor into `forward(...)`** instead of relying on a static model/class toggle. This is specifically meant to avoid the kind of `torch.compile` constant-folding problem that previously broke late-QAT in this repo.
4. **Made default dataset/tokenizer paths repo-root-relative**, so the script can actually be launched from inside this candidate directory without rewriting `DATA_PATH` or `TOKENIZER_PATH`.

The rest of the stack is intentionally unchanged: LeakyReLU(0.5)^2, parameter banks + parallel Muon, XSA, partial RoPE, LN scale, VE, EMA/SWA, GPTQ-lite-style export, and optional legal TTT remain intact.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202604071831_annealed-mtp
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs for this candidate:

- `MTP_NUM_HEADS=1`
- `MTP_LOSS_WEIGHT=0.15`
- `MTP_WARMUP_STEPS=250`
- `MTP_FADE_START_SCALE=0.30`
- `MTP_FADE_END_SCALE=0.05`

Those defaults are already baked into the script; they are listed here so sweeps are easy.

## Main risks / tradeoffs

- **Small-model interference risk**: MTP may still steal capacity from the main next-token head even with annealing.
- **Throughput risk**: the extra head is cheap, but not free; a small step-time regression could erase the gain.
- **TTT interaction risk**: if MTP changes the base model's calibration, legal TTT may gain less than it did on the parent record.
- **Quantization interaction risk**: improving pre-quant quality does not guarantee better post-quant or post-TTT bpb.

## Validation

- `python -m compileall candidates/202604071831_annealed-mtp/train_gpt.py` — passed.
- Import-only smoke testing was **not feasible in this container** because the declared runtime dependencies (`numpy`, `sentencepiece`, `torch`) are not installed here, and the trainer also depends on CUDA + FlashAttention at runtime.
