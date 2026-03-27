# Two-head Multi-Token Prediction on the 2026-03-23 stack

## Hypothesis

The strongest underexplored lever in this repo is **multi-token prediction (MTP) as a training-only auxiliary loss**: predict the next token as usual, but also predict the next 2 future tokens with lightweight extra heads during training, then **drop those heads at export time** so the artifact budget is essentially unchanged.

The bet is that this improves sample efficiency on a 10-minute run without paying ongoing model-byte cost, which is unusually well matched to Parameter Golf's rules.

## Why this is promising for this repository

My review of the repo found a very consistent trend:

- the best runs already converged on an 11-layer, 512-dim, int6-compressed stack with XSA, partial RoPE, EMA/SWA, and strong eval-time tricks;
- quantization/export quality and training efficiency matter more than inventing a completely new large subsystem;
- recurrence / depth reuse looked interesting in external research, but the repo's own non-record exploration reported a clear regression from simple layer recurrence under a fixed wall-clock budget;
- there was **already dormant MTP plumbing in the stronger record code**, including explicit export-time stripping of `mtp_heads`, but the recorded runs kept `MTP_NUM_HEADS=0`.

That makes MTP a good next candidate: it is adjacent to the current best stack, cheap to implement, and still meaningfully untested in this codebase.

## Prior records that influenced this candidate

There were **no prior `candidates/` directories** in the repository when this candidate was created.

The most relevant prior records were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current leaderboard entry at review time;
  - provides the training/eval scaffold I copied as the base;
  - contributes LeakyReLU(0.5)^2, Parallel Muon, legal score-first TTT, VE, partial RoPE, and the final export pipeline.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - clean pre-TTT improvement stack;
  - shows that small post-training and averaging refinements still matter.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms partial RoPE + layerwise normalization scale as stable zero-byte wins.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - already contains the MTP code path and export pruning logic, but the included logs show `mtp_num_heads:0`.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful mainly as a warning: simple layer recurrence was a negative result there, so I did **not** choose block sharing / recurrent depth reuse despite it looking attractive in the literature.

## External research that informed the choice

The main paper motivation is:

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - proposes predicting multiple future tokens with independent output heads on top of a shared trunk;
  - reports improved sample efficiency and stronger downstream capability while keeping inference flexible.

A modern production-scale data point pointing in the same direction is:

- DeepSeek-AI, **"DeepSeek-V3 Technical Report"** (`arXiv:2412.19437`)
  - explicitly states that DeepSeek-V3 uses a **multi-token prediction training objective for stronger performance**.

I also considered more invasive ideas from recent compact-model literature, especially shared-block / recurrent-depth approaches (for example ALBERT and more recent small-model sharing papers), but the repo's own history made MTP the safer next step for this specific codebase.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes four focused changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` now defaults to `2`.
   - `MTP_LOSS_WEIGHT` now defaults to `0.15`.
   - This is intentionally conservative: enough to test the idea without overpaying in step time.

2. **Actually optimize the auxiliary heads**
   - the copied record code already had `mtp_heads`, but they were not attached to any optimizer path in this branch of the stack;
   - this candidate wires them into the existing AdamW path used for other small non-banked weights, so the auxiliary objective can train the heads instead of staying a dead code path.

3. **Keep MTP training-only at export time**
   - The existing export path already excludes `mtp_heads` from `export_sd` and rebuilds the quantized eval model with `mtp_num_heads=0`.
   - That means the extra MTP parameters help only during training; they are not serialized into the final artifact.

4. **Make the candidate easier to run and validate locally**
   - add a FlashAttention fallback to PyTorch SDPA when `flash_attn_interface` is unavailable, including re-enabling non-flash SDPA backends for that path;
   - make default `DATA_PATH` and `TOKENIZER_PATH` resolve relative to the repository root, so the script can be launched directly from this candidate directory.
   - leave late QAT **off by default** (`LATE_QAT_THRESHOLD=0.0`) so this candidate does not depend on the compile-time gating bug already noted in earlier record history; MTP is the variable being tested here.

## How to run

From the repository root:

```bash
cd candidates/202603272114_mtp-aux-heads
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The relative-path fix means `DATA_PATH` and `TOKENIZER_PATH` should work automatically when run from this directory, assuming the repo data has been prepared in the usual place.
- If the MTP heads slow training too much, the first ablations to try are `MTP_NUM_HEADS=1` and then `MTP_LOSS_WEIGHT=0.10`.
- If you want to re-introduce late QAT, do it explicitly with `LATE_QAT_THRESHOLD=...` after verifying the compile path behaves as expected.

## Expected risks and tradeoffs

- **Step-time regression**: the extra logits and cross-entropy computations may cost more wall-clock than the auxiliary objective gives back in sample efficiency.
- **Small-model over-regularization**: this trunk is tiny by frontier standards, so a too-strong MTP weight could hurt the main next-token objective.
- **Interaction with TTT / quantization**: MTP only shapes the training trajectory; its gains could disappear after int6 roundtrip or after score-first TTT.
- **Best setting is unclear**: 1 head vs 2 heads, and 0.10 vs 0.15 vs 0.20 loss weight, are all plausible.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603272114_mtp-aux-heads/train_gpt.py
python -m compileall candidates/202603272114_mtp-aux-heads/train_gpt.py
```

Outcome:

- both compile-only syntax checks passed.

CPU smoke test status:

- I attempted a minimal import-and-forward smoke test for the candidate, but the workflow container's available `python` / `python3` interpreters do **not** have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- I did not install new heavyweight dependencies inside this workflow, so a runtime smoke test was **not feasible in this environment**.
