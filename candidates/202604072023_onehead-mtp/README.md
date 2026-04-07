# Candidate: One-Head MTP on the LeakyReLU2 + Legal TTT stack

## Hypothesis

Add a **single multi-token prediction (MTP) auxiliary head** to the current strongest training stack so the trunk learns a slightly richer future-prediction objective during the fixed 10-minute training budget, then **drop that extra head from the exported artifact** so the submission size stays unchanged.

The bet is that this is a better fit for this repository than another large architecture fork: most recent gains came from improving evaluation context or training/sample efficiency, and MTP directly targets sample efficiency.

## Why it is promising for this repository

- The strongest record line already solved many of the obvious compression and architecture wins: int6 export, XSA, EMA/SWA, partial RoPE, GPTQ-lite, LeakyReLU2, and legal TTT.
- The repo history shows repeated gains from getting more value out of the same wallclock budget, while some capacity-only ideas regressed when they reduced effective step count.
- MTP is attractive here because the **auxiliary head is training-time only**. It can improve the trunk while adding **no persistent artifact cost** once excluded from export.
- The record code already had dormant MTP scaffolding, but recent runs kept `MTP_NUM_HEADS=0`, so this is a real unexplored branch rather than a repeat of an existing record.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - Current best overall line.
   - Supplies the base recipe: LeakyReLU2 MLP, parameter banking + Parallel Muon, legal score-first TTT, GPTQ-lite int6 export, XSA, VE, EMA/SWA.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - Confirms the current line is already heavily optimized for quantization/export quality.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - Shows the move to partial RoPE + LN scale and also carries an earlier version of the MTP hook.
4. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
   - Establishes the core 11-layer XSA/EMA branch that later records improved.

## External research

1. **Multi-Token Prediction** — Gloeckle et al., arXiv:2404.19737  
   Training with multiple future-token heads improves sample efficiency and downstream capability while keeping the main autoregressive trunk shared.
2. **Self-Distillation for Multi-Token Prediction** — Zhao et al., arXiv:2603.23911  
   Highlights that MTP heads can be useful but also notes that jointly training many heads can be hard, which is one reason this candidate starts with a conservative **one-head** setup.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate changes only the MTP path:

1. **Turns MTP on by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Fixes optimizer wiring for MTP heads**
   - Earlier record variants optimized MTP heads, but the latest parameter-banking refactor no longer added them to any optimizer path.
   - This candidate restores that path by routing MTP head weights into the Muon matrix optimizer.
3. **Preserves export size behavior**
   - MTP heads are still excluded from the exported artifact, so the candidate keeps the same size discipline as the base record.

## Why this differs from the existing records

Recent records mostly improved:

- attention behavior (`XSA`, partial RoPE),
- averaging/export quality (`EMA`, GPTQ-lite, warmdown tuning),
- or evaluation context (`sliding window`, legal TTT).

This candidate instead targets the **training objective itself** with a **training-time-only auxiliary loss** that should improve trunk quality without consuming artifact budget.

## How to run or evaluate it

From this candidate directory:

```bash
SEED=1337 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the same default architecture family as the 2026-03-23 record and changes only the MTP-related defaults.

## Validation

Local lightweight validation was run after implementation:

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604072023_onehead-mtp/train_gpt.py`
  - **Outcome:** succeeded.
- A static integrity check over `candidates/202604072023_onehead-mtp/train_gpt.py`
  - **Outcome:** confirmed the candidate keeps MTP export exclusion, enables MTP by default, and restores MTP optimizer wiring.
- A CPU-only import/forward smoke test with a stubbed `flash_attn_interface`
  - **Outcome:** **not feasible in this runner** because the repository Python dependencies are not installed here (`torch`, `numpy`, and `sentencepiece` are all missing), so the module cannot be imported deeply enough for a real model forward pass without first installing the full ML stack.

## Main expected risks and tradeoffs

- **Step-time risk:** even one extra auxiliary head adds projection/loss work, so any BPB gain must outweigh the reduced number of optimization steps in 600 seconds.
- **Objective interference:** MTP can improve sample efficiency, but it can also tug the trunk away from the exact next-token objective if the loss weight is too high.
- **TTT interaction uncertainty:** because the exported/evaluated model is rebuilt without MTP heads, any benefit must come entirely through better pre-export trunk training rather than evaluation-time adaptation.
- **Small-model regime uncertainty:** most MTP literature emphasizes larger models, so gains may be smaller here than in the cited papers.
