# Sparse MTP Auxiliary on the 11L Frontier Stack

## Hypothesis

The strongest open space in this repo is no longer another small quantization tweak or another eval-only trick; it is **better supervision per training token**. A lightweight **multi-token prediction (MTP)** auxiliary should improve sample efficiency on the current 11-layer frontier stack, but the repo's 10-minute wallclock cap makes dense MTP risky. This candidate therefore uses a **sparse 1-head future-token auxiliary**: predict one extra future token, but only on every 4th position.

That preserves the main idea from MTP papers while cutting most of the extra projection cost. The MTP head is still **training-only** and is excluded from the exported model, so the candidate does not pay model-artifact bytes for the auxiliary head.

## Why this is promising for this repository

- The current frontier is already saturated with:
  - 11L / 512d / 4KV / MLP3x,
  - XSA on deep layers,
  - partial RoPE + LN scale,
  - EMA/SWA variants,
  - GPTQ-lite / int6 export refinements,
  - legal TTT and sliding-window eval.
- The best base script already contained dormant MTP code paths, but current record logs still show `mtp_num_heads:0`, so this direction appears **prepared but untested** in the submitted record line.
- Sparse MTP is a repo-specific adaptation of a research-backed idea to this challenge's real bottleneck: **wallclock-limited training on tiny models**.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - Best current stack.
   - Supplies LeakyReLU^2, parameter banking / parallel Muon, XSA, partial RoPE, LN scale, VE, GPTQ-lite-style export, and the dormant MTP hooks.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - Best pure training/export base before TTT.
   - Confirms the value of EMA + GPTQ-lite + warmdown3500 on the 11L stack.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - Established partial RoPE and LN scale as robust zero-parameter wins.

## External research that informed it

- **Better & Faster Large Language Models via Multi-token Prediction** (`arXiv:2404.19737`)  
  Motivates MTP as a sample-efficiency improvement over plain next-token training.
- **On multi-token prediction for efficient LLM inference** (`arXiv:2502.09419`)  
  Notes that hidden states specialize for NTP and that **joint training** works better than retrofitted frozen-head MTP, which matches this repo's from-scratch regime.
- **Multi-Token Prediction via Self-Distillation** (`arXiv:2602.06019`)  
  Reinforces that future-token objectives remain an active, practical direction even when deployment constraints matter.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Sparse MTP defaults**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`
   - new `MTP_STRIDE=4`
2. **Sparse auxiliary loss**
   - the extra future-token head only trains on every `MTP_STRIDE`-th position instead of every token.
3. **No artifact-cost change in exported model**
   - `mtp_heads` are still removed from the exported state dict before quantization/eval.
4. **Cheaper local validation path**
   - `flash_attn_interface` import is optional.
   - attention falls back to PyTorch SDPA when FlashAttention is unavailable.
   - `SMOKE_TEST=1` runs a tiny local forward/backward without tokenizer or dataset setup when `torch` is available.

## How to run or evaluate it

From the repo root:

```bash
cd candidates/202604020641_sparse-mtp-aux
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 MTP_STRIDE=4 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This keeps the training-side change isolated. The script still retains the legal TTT path from the base implementation if you want to stack the candidate with that evaluation method afterward.

Local smoke command:

```bash
cd candidates/202604020641_sparse-mtp-aux
SMOKE_TEST=1 python train_gpt.py
```

## Main expected risks and tradeoffs

- Even sparse MTP still adds training-time output projection work; if the stride is too small, step count may drop enough to erase the gain.
- MTP gains are strongest in larger models in the literature, so the effect size on this tiny 1024-vocab setup may be modest.
- If `MTP_LOSS_WEIGHT` is too high, the trunk may over-optimize for future-token auxiliaries and hurt next-token perplexity.
- This candidate intentionally leaves quantization/export unchanged, so if the next frontier is actually export-side again, the gain may not surface.

## Validation

Commands run in this container:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604020641_sparse-mtp-aux/train_gpt.py
SMOKE_TEST=1 python candidates/202604020641_sparse-mtp-aux/train_gpt.py
```

Outcomes:

- `compileall` passed for the root scripts, `data/`, and the candidate script.
- The candidate now supports a dependency-light smoke path, but a true runtime smoke test was **not feasible in this container** because `torch` is not installed here, and installing it into a temporary venv was blocked by network/proxy restrictions.
