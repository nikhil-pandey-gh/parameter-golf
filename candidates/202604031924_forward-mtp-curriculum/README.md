# Forward-Curriculum MTP on the Current 11L TTT Stack

## Hypothesis

A **single training-only future-token head** should improve sample efficiency for this 10-minute track if it is introduced with a **forward curriculum** instead of being active from step 0. The extra head does not need to ship in the artifact, so the model can spend temporary training compute on a richer objective without paying a size penalty at export time.

Concretely, the **recommended candidate run** ramps a 2-token-prediction auxiliary loss from **0.0** to **0.12** by **wallclock progress**:

- `MTP_START_FRAC=0.15`
- `MTP_FULL_FRAC=0.45`
- `MTP_NUM_HEADS=1`

This keeps the early trunk focused on next-token modeling, then turns on a modest future-token signal once the basic representation has stabilized.

## Why this is promising here

This repository has already extracted most of the obvious architectural wins from the 11-layer stack: XSA, partial RoPE, LN scaling, EMA/GPTQ-lite, LeakyReLU(0.5)^2, parameter banking, and legal score-first TTT. The remaining room is likely to come from **better sample efficiency under the same 600s wallclock cap**, especially through training-only tricks that do not grow the exported model.

The repo history also showed a useful clue: **MTP code already existed in the 2026-03-20/21/22/23 lineage, but the recorded runs kept `mtp_num_heads:0`**, so that path was effectively dormant. This candidate explicitly activates that direction with a curriculum tuned for small-model training rather than a static always-on auxiliary loss.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - Best current score (`val_bpb: 1.1194` post-TTT).
   - Supplies the actual base implementation: 11L banked stack, LeakyReLU(0.5)^2, partial RoPE, XSA, VE layers, legal TTT, and Parallel Muon.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - Best clean non-TTT line.
   - Reinforced that the frontier favors cheap training/eval improvements on top of the same 11L backbone.
3. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
   - Important because it already contained a training-only MTP export path.
   - The logs show that path was left off in submitted runs (`mtp_num_heads:0`), which made it a good candidate to revisit with a better schedule instead of a blind default flip.

## External research that informed it

1. **Gloeckle et al., 2024 — “Better & Faster Large Language Models via Multi-token Prediction”** (`arXiv:2404.19737`)
   - MTP can improve sample efficiency by training auxiliary future-token heads on top of a shared trunk.
   - That is a good fit for this challenge because training time is tightly capped while the exported artifact only needs the next-token model.
2. **Aynetdinov and Akbik, 2025 — “Pre-Training Curriculum for Multi-Token Prediction in Language Models”** (`arXiv:2505.22757`)
   - Smaller language models do not benefit as reliably from static MTP.
   - A **forward curriculum** helps small models leverage MTP more safely, which is why this candidate ramps the auxiliary loss in instead of enabling it at full strength from the start.

## What changed versus the chosen base implementation

Base: `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Activate one training-only MTP head in the candidate run config**
   - Use `MTP_NUM_HEADS=1` for the candidate experiment.
   - The script default remains `0` so exported artifacts stay load-compatible with the default inference/eval shape.
2. **Add forward-curriculum scheduling**
   - `MTP_LOSS_WEIGHT` now acts as the maximum auxiliary weight.
   - New knobs: `MTP_START_FRAC` and `MTP_FULL_FRAC`.
   - The current MTP weight is updated from wallclock progress each training step.
3. **Make the MTP weight runtime-controlled**
   - The active auxiliary weight is held in a module buffer and updated by the training loop.
   - This avoids treating the curriculum as a fixed codepath choice.
4. **Actually optimize the MTP head**
   - In the copied 03-23 stack, the dormant MTP head existed but was not wired into the optimizer path for this parameter-banked variant.
   - This candidate adds the head weights to a dedicated replicated AdamW path so they are trained correctly under multi-GPU runs.
5. **Keep the artifact clean**
   - `mtp_heads` remain excluded from export and from the final eval model.

## How to run / evaluate

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.12 MTP_START_FRAC=0.15 MTP_FULL_FRAC=0.45 \
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

For a quicker pre-TTT ablation, set `TTT_ENABLED=0`.

## Validation

Commands run in this repo:

1. `python -m compileall candidates/202604031924_forward-mtp-curriculum/train_gpt.py`
   - **Passed**
2. Attempted CPU-only import/forward smoke with a mocked flash-attention shim
   - **Not feasible in this runner as-is** because the Python environment here does not currently have the repo dependencies installed (`torch`, `numpy`, and `sentencepiece` were all missing), so a meaningful import-level smoke test could not be completed without first installing the full training stack.

## Main risks / tradeoffs

1. **Step-time regression**
   - Even one extra future-token head adds another vocab projection and CE term during training, so the model may take fewer optimizer steps within 600 seconds.
2. **Objective mismatch**
   - If the auxiliary weight is too large, the trunk could overfit the training-only future-token task and slightly hurt final next-token BPB.
3. **Curriculum sensitivity**
   - The best `MTP_START_FRAC`, `MTP_FULL_FRAC`, and `MTP_LOSS_WEIGHT` may differ across hardware because the schedule is keyed off wallclock progress.
4. **Interaction with TTT**
   - The hoped-for win is that a better-pretrained trunk improves both pre-TTT and post-TTT metrics, but that interaction still needs a real GPU run to confirm.
