# Staged Multi-Token Prediction on the 11L LeakyReLU² / GPTQ-lite Stack

## Hypothesis

Enable **training-only multi-token prediction (MTP)** on top of the best current 11-layer trunk, but **decay the auxiliary loss to zero before the late quantization-focused phase**. The expected win is better sample efficiency early in training without paying artifact bytes at export and without perturbing the final warmdown / EMA / GPTQ-lite regime that has driven the strongest recent records.

## Why this is promising for this repository

Repository history points to a consistent pattern: the strongest gains now come from changes that improve effective context, quantization behavior, or late-phase optimization without bloating the artifact.

- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` is the best current stack and already combines the strongest trunk ideas: leaky-ReLU-squared MLP, 11L parameter-banked model, partial RoPE, deep-layer XSA, VE128, EMA, tight SWA, and GPTQ-lite int6 export.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` shows that tiny late-phase tweaks still matter once the architecture is strong: EMA, longer warmdown, and better post-training quantization each chipped away at BPB.
- The latest record scripts already contain **`MTP_NUM_HEADS` / `MTP_LOSS_WEIGHT` hooks**, but the record READMEs do not report actually turning them on. That makes MTP one of the clearest “wired but unexplored” ideas in the repo.
- The challenge artifact counts exported weights, not training-only heads. This candidate keeps MTP heads out of the final export, so the extra supervision spends training compute rather than submission bytes.

## Prior records and experiments that influenced this candidate

- **Base trunk:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Quantization / warmdown baseline:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Partial RoPE + LN scale:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Efficient XSA precedent:** `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
- **Repo dead ends to avoid:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` both report that SwiGLU and depth recurrence were too slow or too step-hungry in the 10-minute regime.

## External research that informed it

- **Fabian Gloeckle et al., “Better & Faster Large Language Models via Multi-Token Prediction” (arXiv:2404.19737).** The paper argues that predicting multiple future tokens with independent heads on a shared trunk improves sample efficiency and downstream capability, with little or no training-time overhead.
- **DeepSeek-AI et al., “DeepSeek-V3 Technical Report” (arXiv:2412.19437).** The report explicitly notes a **multi-token prediction training objective** in a modern high-performing stack, which supports MTP as a practical large-scale recipe rather than a toy ablation.
- **Tang et al., “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration” (arXiv:2306.00978)** and **Ma et al., “BitNet: Scaling 1-bit Transformers for Large Language Models” (arXiv:2310.11453)** were reviewed as low-bit training/export references. They reinforced the choice to leave the existing GPTQ-lite export path alone and keep the new idea focused on **sample efficiency without broad quantization infrastructure changes**.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps the record trunk and export path, but makes four targeted changes:

1. **Turn MTP on by default.**
   - `MTP_NUM_HEADS` now defaults to `2`
   - `MTP_LOSS_WEIGHT` now defaults to `0.15`
   - MTP heads are optimized with the same auxiliary-head Adam path used for output heads, so the auxiliary objective is live during training rather than just instantiated

2. **Stage MTP out during warmdown.**
   - Added `MTP_DECAY_START_SCALE` (default `0.35`)
   - Added `MTP_DECAY_END_SCALE` (default `0.15`)
   - The effective MTP loss is full-strength above `0.35`, linearly decays during warmdown, and is zero by the time the LR scale reaches `0.15`
   - Once the effective weight reaches zero, the training forward pass skips the MTP branch entirely instead of paying the compute and multiplying by zero afterward

3. **Make the MTP weight a runtime input instead of a compile-time constant.**
   - The loss weight is passed into `GPT.forward(...)` each training step
   - This keeps the schedule explicit and avoids relying on static class attributes for behavior changes during compiled training

4. **Add a CPU-safe smoke path.**
   - `SMOKE_TEST=1 python train_gpt.py` builds a tiny CPU model, runs a forward/backward pass, and checks `forward_logits`
   - FlashAttention now has a safe fallback to PyTorch SDPA when FlashAttention is unavailable

## How to run or evaluate it

### Standard training run

```bash
cd candidates/202604071533_staged-mtp
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The new MTP defaults are already baked in. The rest of the stack still inherits the strong record defaults from the 2026-03-23 base.

### Explicit run with the new knobs shown

```bash
cd candidates/202604071533_staged-mtp
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_DECAY_START_SCALE=0.35 \
MTP_DECAY_END_SCALE=0.15 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Minimal local smoke check

```bash
cd candidates/202604071533_staged-mtp
SMOKE_TEST=1 python train_gpt.py
```

## Validation

Validation was run inside a temporary virtualenv created under `/tmp/gh-aw/agent/pg-venv` with repo dependencies installed from `requirements.txt`.

- `. /tmp/gh-aw/agent/pg-venv/bin/activate && python -m compileall candidates/202604071533_staged-mtp/train_gpt.py` — **passed**
- `. /tmp/gh-aw/agent/pg-venv/bin/activate && cd candidates/202604071533_staged-mtp && SMOKE_TEST=1 python train_gpt.py` — **passed** (`smoke_test:ok loss:4.7917 logits_shape:(2, 32, 64)`)

## Main expected risks or tradeoffs

- **Step-time regression.** Even export-free auxiliary heads can still cost enough training throughput to cancel out the sample-efficiency gain.
- **Heuristic schedule.** Decaying MTP against LR scale is motivated by the repo’s warmdown/QAT behavior, but it is still a heuristic and may need tuning.
- **Small-model mismatch.** The main MTP paper reports stronger benefits at larger scales; a compact 11L/512d model may see a smaller gain.
- **Interaction with TTT remains unknown.** This candidate targets the trunk first. Legal TTT may stack with it, but that has not been validated here.
