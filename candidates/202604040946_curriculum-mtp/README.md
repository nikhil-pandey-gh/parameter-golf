# Curriculum MTP on the LeakyReLU² + Legal TTT stack

## Hypothesis

Tiny models in this repo are heavily constrained by **sample efficiency** during the 10-minute training window, not just by artifact bytes. A **curriculum multi-token prediction (MTP)** auxiliary loss should improve trunk learning early enough to matter, while keeping artifact size unchanged because the extra MTP heads are **training-only** and excluded from export.

## Why this is promising here

- The record history already converged on a strong 11-layer recipe: int6/int5-aware compression, XSA in deep layers, partial RoPE, EMA/SWA, BigramHash/SmearGate, sliding-window eval, and finally legal score-first TTT.
- Multiple strong record scripts already carried dormant `MTP_*` plumbing, but none of the published record READMEs or submissions actually enabled it. That makes MTP a real unexplored gap in this repository rather than a repeat.
- The latest external research points in the same direction:
  - Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** (arXiv:2404.19737) argues that predicting multiple future tokens improves sample efficiency.
  - Ansar Aynetdinov and Alan Akbik, **Pre-Training Curriculum for Multi-Token Prediction in Language Models** (arXiv:2505.22757) shows that **small language models need a curriculum** instead of naive always-on MTP.

## Prior work that influenced this candidate

- **Chosen base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current record in `records/`
  - keeps LeakyReLU², parameter banking / parallel Muon, partial RoPE, XSA, value embeddings, legal score-first TTT
- **Earlier record trends that matter here**
  - `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` and `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` for the mature 11L compressed stack
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for partial RoPE and the warning that `torch.compile` can silently constant-fold training-only branches
  - `2026-03-17_LoRA_TTT/` and `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` for evaluation-aware improvements and the legal score-first TTT protocol
- **Prior candidates:** none existed before this directory was created

## What changed versus the chosen base

1. **Enabled training-only MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
   - MTP heads are still excluded from export, so submission bytes remain driven by the real model, not the auxiliary objective.

2. **Added a small-model curriculum for MTP**
   - `MTP_START_FRAC=0.10`
   - `MTP_HEAD_GAP_FRAC=0.20`
   - `MTP_RAMP_FRAC=0.18`
   - `MTP_HORIZON_DECAY=0.5`
   - Head 1 ramps in first, head 2 later, and the whole auxiliary term naturally shrinks with warmdown because the schedule is multiplied by the current LR scale.

3. **Actually train the auxiliary heads**
   - The candidate explicitly adds `mtp_heads` to the AdamW-managed non-bank parameters so the auxiliary predictors and trunk both receive useful gradient.

4. **Made the script locally importable without FlashAttention 3**
   - If `flash_attn_interface` is unavailable, attention falls back to a correct causal manual path instead of failing at import time.
   - This is only for portability and smokeability; the target fast path is still FlashAttention 3 on Hopper.

5. **Tightened evaluation-mode handling**
   - Validation helpers now restore the incoming train/eval mode instead of always forcing `.train()` afterward.

## How to run

From this candidate directory:

```bash
NCCL_IB_DISABLE=1 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
TTT_ENABLED=1 \
TTT_FREEZE_BLOCKS=0 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_START_FRAC=0.10 \
MTP_HEAD_GAP_FRAC=0.20 \
MTP_RAMP_FRAC=0.18 \
MTP_HORIZON_DECAY=0.5 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected risks and tradeoffs

- **More train-time compute:** two extra MTP heads cost additional projection/loss work and may reduce steps if the auxiliary benefit is too small.
- **Small-model uncertainty:** the curriculum is research-grounded, but this repo’s models are tiny enough that MTP could still underperform if the onset is too early or too strong.
- **Gain may concentrate pre-TTT:** the clearest win may show up in pre-TTT validation loss/BPB rather than fully adapted post-TTT BPB.
- **Fallback path is not the target benchmark path:** the manual causal-attention fallback is for portability and smoke checks, not for competitive H100 timing.

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604040946_curriculum-mtp/train_gpt.py` | Passed |
| `python -m compileall candidates/202604040946_curriculum-mtp/train_gpt.py` | Passed after final fixes |
| Focused code review of `candidates/202604040946_curriculum-mtp/train_gpt.py` | Initial findings addressed; final review clean |
| Import-based CPU smoke test | Not feasible in this environment because `torch` is not installed (`importlib.util.find_spec("torch")` returned false) |
