# Train-only weighted MTP on the legal-TTT stack

## Hypothesis

Add a small **multi-token prediction (MTP)** auxiliary objective to the current best 10-minute stack so the shared trunk learns richer future-token structure during training, then **drop the auxiliary heads from the exported artifact** and **keep TTT adaptation on the main next-token loss only**. The bet is that this improves sample efficiency without spending artifact bytes on inference-time capacity.

## Why this is promising for this repository

- The strongest records already stack efficient attention, EMA/SWA, Partial RoPE, LN scaling, leaky-ReLU-squared MLPs, and legal score-first TTT. The remaining headroom looks more like a **training-objective** problem than another architecture rewrite.
- The 2026-03-23 record already contains dormant MTP plumbing and explicitly excludes `mtp_heads` from export, so this idea is unusually cheap to test in this repo.
- External research points the same way:
  - **Gloeckle et al., “Better & Faster Large Language Models via Multi-Token Prediction” (arXiv:2404.19737)** reports higher sample efficiency from predicting multiple future tokens with independent auxiliary heads on a shared trunk.
  - **DeepSeek-V3 Technical Report (arXiv:2412.19437)** also adopts a multi-token prediction objective in a modern high-performance stack, which is a strong signal that the idea remained relevant into late 2024 / early 2025.

## Prior records that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Current best mean score in-repo: **1.1194 val_bpb**
  - Supplies the base stack: Parameter Banking + Parallel Muon, legal score-first TTT, LeakyReLU(0.5)^2, Partial RoPE, LN scale, VE, XSA, EMA/SWA, GPTQ-lite int6 + lzma
- **Architecture lineage:** `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`, and `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - These runs established the current winning stack for export, EMA, XSA, Partial RoPE, and LN scaling.
- **Negative prior that shaped the choice:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Its layer-recurrence ablation regressed badly, which made a shared-core / recurrent-depth candidate less attractive for the next incremental step.

## What changed versus the chosen base implementation

Starting from `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. **Turns on MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Weights shorter-horizon auxiliary heads more strongly**
   - New `MTP_LOSS_DECAY=0.5`
   - Auxiliary losses are combined as a decay-weighted average instead of a flat mean, so the shorter auxiliary horizon dominates the longer one (`t+2` over `t+3` for the default 2-head setup).
3. **Keeps MTP strictly training-only**
   - The auxiliary heads are optimized during training.
   - They are stripped from the exported state dict, so the final quantized eval model and legal TTT path both use only the main next-token objective.
4. **Aligns defaults with the best published stack**
   - `BIGRAM_VOCAB_SIZE=1536`
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`
5. **Adds a PyTorch SDP fallback when FlashAttention 3 is unavailable**
   - This does not change the intended GPU path when `flash_attn_interface` is installed.
   - It mainly makes lightweight non-GPU inspection and smoke imports less brittle.

## Why this differs from the existing records and candidates

- No previous `records/` README documents an actual MTP-enabled run, even though later record scripts already contain dormant MTP support.
- This candidate adds a **repo-specific twist** rather than only flipping the old flag:
  - **decay-weighted auxiliary horizons** for a tiny model under a short training budget
  - **training-only MTP**, so the auxiliary task improves the trunk during training without changing the exported artifact or legal TTT eval path

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202604031017_trainonly-mtp-ttt
SEED=1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_LOSS_DECAY=0.5 \
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable MTP entirely
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
MTP_NUM_HEADS=0 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep train-time MTP but skip legal TTT
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
MTP_NUM_HEADS=2 TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- **Training throughput risk:** even small auxiliary heads add extra logits and cross-entropies, so the gain must beat any step-rate loss.
- **Tiny-model horizon risk:** for a ~20M-class model, farther-future targets may regularize too hard; the decay weighting is meant to reduce that risk, but it is still an assumption.
- **Train/eval mismatch risk:** MTP is intentionally training-only here, so any benefit has to survive the handoff into the main-loss-only exported eval model.
- **Validation gap:** this runner can only do lightweight static checks; the real answer still needs an H100-style run under the repository’s normal evaluation setup.

## Validation

Commands run on this runner:

```bash
python -m compileall candidates/202604031017_trainonly-mtp-ttt/train_gpt.py
python - <<'PY'
import torch
PY
```

Outcomes:

- `python -m compileall ...` **passed**
- A minimal CPU forward smoke was **not feasible on this runner** because both `/usr/bin/python` and `/usr/bin/python3` are missing the `torch` package, so even a tiny local model instantiation cannot execute here
- The candidate script still includes a PyTorch SDP attention fallback so that, in an environment with PyTorch but without `flash_attn_interface`, an import-time / small-forward smoke remains possible
