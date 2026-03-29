# Warmdown-Annealed MTP on the LeakyReLU² + Legal TTT Stack

## Hypothesis

The strongest recent scripts in this repo already spend almost the full 16 MB artifact budget, so the next gain is more likely to come from **better training signal** than from another export-time compression trick. This candidate enables a **single multi-token prediction (MTP) head** during training and linearly fades its loss weight out with the existing warmdown schedule.

The goal is to get the early/mid-training sample-efficiency benefit reported in the MTP literature while removing most of the train/eval mismatch before EMA export, quantization, and final evaluation.

## Why this is promising for this repository

- The 2026-03-23 record already has a mature high-performing stack: LeakyReLU(0.5)^2, parameter banking + Parallel Muon, XSA on the last 4 layers, partial RoPE, VE128, EMA + tight SWA, GPTQ-lite int6, and legal score-first TTT.
- The recent 2026-03-22 and 2026-03-23 scripts already contain dormant MTP plumbing and explicitly exclude `mtp_heads` from export, which makes MTP unusually cheap to test here: it adds **training compute**, but essentially **no artifact bytes**.
- The non-record 2026-03-19 single-5090 study found that layer recurrence was a bad trade under fixed wall-clock because it reduced the number of optimizer steps too much. MTP targets the same "more supervision per token" goal without changing inference/export depth.

## Prior records and experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Chosen as the direct base because it is the strongest current full stack in this checkout.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Useful as the strongest recent non-TTT base and as confirmation that the 11-layer export path is already well tuned.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Important negative result: deeper/recurrent compute was not a good fit under the time cap, which pushed this candidate toward a training-only auxiliary loss instead.

## External research

- Fabian Gloeckle et al., [*Better & Faster Large Language Models via Multi-token Prediction*](https://arxiv.org/abs/2404.19737), arXiv:2404.19737.
  - Key takeaway for this repo: predicting multiple future tokens with independent heads on top of a shared trunk can improve sample efficiency and induction-head formation without needing those extra heads at inference.
- I also reviewed the recent arXiv search surface around MTP/lookahead objectives while selecting this idea:
  - <https://arxiv.org/search/?query=multi-token+prediction+language+model&searchtype=all&abstracts=show&order=-announced_date_first&size=50>
  - That survey turned up follow-on directions such as **Self-Distillation for Multi-Token Prediction** and **Thinking into the Future: Latent Lookahead Training for Transformers**. I intentionally picked the simpler branch that this repository can absorb with minimal infrastructure changes.

## What changed versus the chosen base implementation

This candidate copies the 2026-03-23 record script and makes five focused changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` now defaults to `1`.
   - `MTP_LOSS_WEIGHT` now defaults to `0.15`.

2. **Anneal the MTP objective away during warmdown**
   - The training loop now passes `args.mtp_loss_weight * lr_scale` into the compiled forward path, so the auxiliary objective naturally falls to zero as the learning rate warms down and the extra MTP branch shuts off once that scheduled weight reaches zero.

3. **Pass the MTP weight explicitly through `GPT.forward(...)`**
   - This avoids relying on a mutable attribute inside the compiled graph and keeps the scheduling logic in the training loop where it is easy to reason about.

4. **Make the script runnable from this candidate directory**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root, so `train_gpt.py` can be launched from `candidates/202603290837_mtp-warmdown/` directly.

5. **Actually optimize the MTP heads**
   - The base script had dormant MTP plumbing but did not attach `mtp_heads` to an optimizer path. This candidate adds them to the small non-bank AdamW group so the auxiliary objective really trains and so those gradients are properly zeroed/clipped.

The export path is otherwise unchanged: `mtp_heads` are still excluded from the final serialized model state, so this remains a training-time-only idea.

## How to run

From this candidate directory:

```bash
cd candidates/202603290837_mtp-warmdown
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The default `DATA_PATH` / `TOKENIZER_PATH` now resolve relative to the repository root, so this command works directly from the candidate directory **once the standard challenge data has been populated under `repo_root/data/`** (see the root repository README for the `data/cached_challenge_fineweb.py` setup flow). If your data lives elsewhere, override `DATA_PATH` and `TOKENIZER_PATH` explicitly.

If you want to isolate the pure training-side effect before spending time on TTT, rerun the same command with `TTT_ENABLED=0`.

## Expected risks and tradeoffs

- **Training cost:** even one extra head adds logits and cross-entropy work, so steps-per-600s may drop.
- **Objective mismatch:** if the auxiliary target stays too strong too late, it can help early optimization but hurt the final next-token model. That is the reason for annealing it to zero during warmdown.
- **Interaction risk:** MTP may couple differently with EMA, GPTQ-lite quantization, or legal TTT than it does with plain next-token-only training.
- **Small-model uncertainty:** the MTP literature is strongest at larger scales, so the payoff in this tiny-budget regime is plausible but not guaranteed.

## Validation

Validation was run locally in this repository after the candidate files were created.

- `python -m compileall candidates/202603290837_mtp-warmdown/train_gpt.py`
  - **Passed.**
- `python - <<'PY' ... PY` import smoke with a stubbed `flash_attn_interface`
  - **Blocked by environment:** this runner does not have the repository Python dependencies installed (`numpy` is missing), so a true import-level smoke test could not be completed here.
- `python - <<'PY' ... PY` repo-relative path check
  - Confirmed that the candidate resolves `DATA_PATH` / `TOKENIZER_PATH` relative to the repository root rather than the candidate directory.
  - The standard challenge data directories are **not present in this checkout**, so no end-to-end runtime smoke was possible without first downloading or mounting the dataset/tokenizer artifacts.
