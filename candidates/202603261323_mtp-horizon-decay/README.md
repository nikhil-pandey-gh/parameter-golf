# Candidate: horizon-decayed training-only MTP on the 2026-03-23 stack

## Hypothesis

Activate the repository's dormant multi-token prediction (MTP) path on top of the current best `LeakyReLU² + Legal TTT + Parallel Muon` stack, but weight farther horizons less aggressively than nearer ones.

The concrete change is:

- enable 2 auxiliary future-token heads during training,
- set a conservative global auxiliary weight (`MTP_LOSS_WEIGHT=0.15`),
- apply geometric horizon decay (`MTP_HORIZON_DECAY=0.5`) so the t+2 target matters most and the t+3 target acts as a lighter regularizer,
- keep export unchanged by dropping `mtp_heads.*` from the saved state dict exactly as the base implementation already does.

This aims to buy better sample efficiency inside the fixed 600-second training budget without spending any artifact bytes on the auxiliary heads.

## Why this is promising for this repository

The record history is already heavily optimized around the same static trunk:

- 11 layers, 512d, 4 KV heads, 3x MLP,
- partial RoPE, XSA, BigramHash, SmearGate, VE, EMA/SWA,
- aggressive int6 export with sliding-window evaluation,
- and, in the current best record, legal score-first TTT.

What the repository has **not** actually explored is running the built-in MTP path with nonzero settings. Multiple strong record-family scripts already contain MTP code, but the published runs leave `MTP_NUM_HEADS=0`.

That makes MTP unusually attractive here:

- it is already wired into the best code path,
- it is training-only and excluded from export,
- it directly targets sample efficiency under a hard wallclock cap,
- and it is orthogonal to the existing gains from quantization, averaging, and TTT.

## Prior records and candidates that influenced this choice

- Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - current best overall score: post-TTT `val_bpb: 1.1194`
  - strongest end-to-end stack in the repo
- Static-model baseline for the same family: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best non-TTT static stack
- Earlier XSA/partial-RoPE stack: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms the architecture lineage and that later gains mostly came from better averaging/export/eval
- Earlier XSA README: `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md`
  - shows the code path already exposed `MTP_NUM_HEADS`, but the recorded run used `MTP_NUM_HEADS=0`

There were no prior experiments under `candidates/` when this candidate was created.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737)
  - argues that independent future-token heads on a shared trunk improve sample efficiency when used as an auxiliary training task
- Somesh Mehra et al., **"On multi-token prediction for efficient LLM inference"** (arXiv:2502.09419)
  - cautions that hidden states are specialized for next-token prediction, so naïvely joint-training MTP heads can create tension with the main objective
- Anastasios Gerontopoulos et al., **"Multi-Token Prediction Needs Registers"** (arXiv:2505.10518)
  - suggests that stronger MTP formulations may need extra structural help; useful as follow-up guidance if plain auxiliary heads show mixed results
- Guoliang Zhao et al., **"Self-Distillation for Multi-Token Prediction"** (arXiv:2603.23911)
  - further motivates being conservative about preserving main-head quality while extracting MTP benefits

This candidate therefore takes the most conservative high-upside slice of the MTP literature: use a small number of auxiliary horizons, and discount farther horizons rather than weighting them equally.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`:

- default `MTP_NUM_HEADS` changed from `0` to `2`
- default `MTP_LOSS_WEIGHT` changed from `0.2` to `0.15`
- new `MTP_HORIZON_DECAY` knob added, default `0.5`
- the auxiliary MTP loss now uses a weighted average across horizons instead of equal weighting
- the MTP heads are explicitly added to the AdamW-managed small-parameter group so the auxiliary task can actually train
- logging now reports `mtp_horizon_decay`

Nothing else in the training trunk, quantization path, or TTT protocol was changed.

## How to run or evaluate it

Example training command, keeping the strongest known record stack and adding this candidate's MTP settings:

```bash
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
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_HORIZON_DECAY=0.5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to isolate the training-only effect before layering on legal TTT, run the same command with `TTT_ENABLED=0`.

## Validation run for this candidate

Commands run while preparing this candidate:

```bash
python -m compileall candidates/202603261323_mtp-horizon-decay/train_gpt.py
```

Observed outcomes:

- `compileall` passed
- a runtime CPU smoke test was attempted next with a temporary FlashAttention compatibility stub, but this workflow container does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), so a meaningful local runtime check was **not feasible in this environment**

## Main expected risks or tradeoffs

- MTP may slightly slow step time; if the slowdown costs too many optimizer steps, the sample-efficiency gain may wash out
- even discounted auxiliary horizons can still interfere with the main next-token objective
- the best repo score currently depends on legal TTT, so any interaction between MTP-shaped representations and TTT adaptation is still uncertain
- if this shows partial but noisy gains, the next follow-up should probably be either:
  - a 1-head-only version,
  - an annealed late-training MTP weight,
  - or a MuToR-style register formulation rather than plain auxiliary heads
