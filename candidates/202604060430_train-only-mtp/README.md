# Train-only MTP on the PR #549 dense stack

## Hypothesis

Add a **training-only multi-token prediction (MTP) auxiliary head** to the current best dense stack so the shared trunk learns more sample-efficient next-step structure during the same 600-second training budget, then **strip the MTP head from the exported artifact** so inference bytes stay aligned with the record stack.

## Why this is promising here

Repository history says the best gains now come from **better use of the existing dense trunk** rather than disruptive architecture changes:

- sliding-window scoring, legal TTT, and better post-training quantization all beat bigger rewrites,
- layer recurrence was explicitly negative under a fixed 10-minute budget,
- SwiGLU helped per step but cost too much throughput on earlier branches,
- the latest SOTA already has a strong dense recipe, but its latent MTP code path is still always disabled (`mtp_num_heads:0` in the record logs).

That makes MTP a good fit for a candidate run: it targets **sample efficiency**, not artifact size, and can be evaluated on top of the strongest existing recipe.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - best current dense stack,
   - LeakyReLU(0.5)^2,
   - legal score-first TTT,
   - parameter banking + Parallel Muon.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - GPTQ-lite clip search and warmdown3500 refinements.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - partial RoPE + LN scale on the 11-layer stack.
4. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
   - XSA-on-deep-layers + EMA direction.

## External research

- **Gloeckle et al., 2024 — “Better & Faster Large Language Models via Multi-token Prediction”**  
  <https://arxiv.org/abs/2404.19737>  
  Reports that predicting multiple future tokens with independent heads improves **sample efficiency** and helps induction-style behavior, while keeping the shared trunk architecture unchanged.
- **DeepSeek-V3 Technical Report, 2024/2025**  
  <https://arxiv.org/abs/2412.19437>  
  Uses a **multi-token prediction training objective** in a modern high-performing model stack, reinforcing that MTP is a practical training objective rather than just a toy ablation.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. Enable **one auxiliary MTP head by default**:
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`
2. Make the script runnable directly from this candidate directory by default:
   - dataset/tokenizer paths resolve from the repository root,
   - `RUN_ID` defaults to the candidate directory name.
3. Align defaults to the actual PR #549-style eval recipe:
   - `BIGRAM_VOCAB_SIZE=1536`
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`
4. Keep the existing **artifact pruning** path that excludes `mtp_heads.*` tensors before serialization, so MTP remains training-only.

## How to run

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
# faster local iteration without legal TTT
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# compare stronger/weaker auxiliary pressure
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 torchrun --standalone --nproc_per_node=8 train_gpt.py
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main risks and tradeoffs

- **Training-time overhead:** the extra logits/head loss can reduce steps completed in 600 seconds.
- **Loss-mixing risk:** too much MTP weight may help representation learning but hurt the final next-token objective.
- **TTT interaction uncertainty:** MTP may improve the pre-TTT trunk, but the post-TTT gain could shrink or expand depending on how adaptation uses the better features.

## Validation

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604060430_train-only-mtp/train_gpt.py` — success
- minimal CPU-only smoke test — not feasible without adding a separate fallback path, because this candidate intentionally stays on the CUDA + FlashAttention-3 record stack used by the dense records
