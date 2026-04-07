# Self-Distilled MTP on the LeakyReLU2 + Legal TTT Stack

## Hypothesis

Turn on a **single training-only multi-token prediction (MTP) head** and stabilize it with a **self-distillation loss from the future main-head logits**. The exported model stays unchanged because the script already strips `mtp_heads.*` before serialization, so the candidate spends extra parameters and compute **only during training**, not in the 16 MB artifact.

## Why this looks promising here

The record history shows that this repo has already extracted most of the obvious gains from:

- compression-aware export (`int6`, GPTQ-lite, fp16/int8 passthrough),
- architectural capacity funded by compression (10L/11L, 3x MLP),
- small attention tweaks (XSA, partial RoPE, LN scaling, VE),
- and evaluation tricks (sliding eval, legal TTT).

What is still underexplored is a **training-only auxiliary objective** that improves representation quality without increasing export size. This codebase was already carrying dormant MTP support, but every checked record and record-derived script kept `MTP_NUM_HEADS=0`, so the idea was present but never activated.

## Prior repo runs that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best in-repo score,
  - already includes the current strongest stack,
  - already excludes MTP heads from export, which makes a training-only auxiliary especially attractive.
- **Core stack lineage:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest non-TTT static stack,
  - confirms that the repo's remaining gains are incremental and usually come from small orthogonal improvements.
- **Negative evidence:** no prior `candidates/` directory existed, and the inspected record scripts kept MTP disabled even when the code path was available.

## External research that informed it

- Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** — https://arxiv.org/abs/2404.19737
  - argues that MTP improves sample efficiency by predicting several future tokens from a shared trunk.
- Guoliang Zhao et al., **Self-Distillation for Multi-Token Prediction** — https://arxiv.org/abs/2603.23911
  - shows that self-distillation stabilizes MTP heads and improves their usefulness with modest extra cost.

## What changed versus the chosen base

Starting from the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` script, this candidate:

1. **Enables one MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Adds self-distillation controls**
   - `MTP_DISTILL_WEIGHT=0.25`
   - `MTP_DISTILL_TEMP=1.5`
3. **Applies a future-main-head KL term**
   - for MTP head `k`, distill from the detached main-head logits at offset `k+1`,
   - keep the standard future-token CE term as the ground-truth objective.
4. **Preserves export size behavior**
   - `mtp_heads.*` are still excluded from the serialized artifact.

The rest of the stack is intentionally unchanged: LeakyReLU(0.5)^2, parameter banks + Parallel Muon, XSA, partial RoPE, VE, EMA/SWA, mixed int6 export, and legal TTT all stay intact.

## How to run

From this candidate directory:

```bash
RUN_ID=self_distilled_mtp \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 MTP_DISTILL_WEIGHT=0.25 MTP_DISTILL_TEMP=1.5 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation run in this environment

Succeeded:

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604072215_self-distilled-mtp/train_gpt.py`
- direct `compile(...)` syntax check for:
  - `train_gpt.py`
  - `candidates/202604072215_self-distilled-mtp/train_gpt.py`

Not feasible here:

- a CPU-only runtime smoke test, because this runner currently lacks the project runtime dependencies (`torch`, `numpy`, `sentencepiece`, `flash_attn_interface`) and the script targets the CUDA/FlashAttention training path.

## Main risks / tradeoffs

- **Training-time cost:** even one auxiliary head adds another vocab projection, so steps may slow down enough to offset some sample-efficiency gain.
- **Interference risk:** the distillation target may regularize too aggressively if the future main-head logits are too sharp early in training.
- **TTT interaction:** the auxiliary may improve the static model but not survive the full legal-TTT evaluation stack as cleanly as it helps pre-TTT validation.
