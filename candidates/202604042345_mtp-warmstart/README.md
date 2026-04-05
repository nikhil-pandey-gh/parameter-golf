# Candidate: MTP warm-start on the current 11L frontier

## Hypothesis

The current record stack already looks saturated on obvious artifact-side tricks (sliding eval, GPTQ-lite/int6 export, EMA/SWA, Partial RoPE, XSA, LeakyReLU^2, legal TTT). The cleanest missing lever is **train-only multi-token prediction (MTP)**: add a small auxiliary loss that predicts farther-future tokens during training, then **drop those heads at export** so the artifact budget stays unchanged.

This candidate uses **two future-token heads**, **warm-starts them from the tied embedding/head weights**, and **downweights farther-horizon heads** so the extra supervision is strong early but does not dominate the main next-token objective.

## Why this is promising here

1. The repository's best 10-minute runs already lean heavily on eval-time and compression-time gains, so an **artifact-free sample-efficiency improvement** is more attractive than another compression tweak.
2. Recent late-record code already contains a dormant MTP path, but the documented runs keep it disabled (`MTP_NUM_HEADS=0` or omitted), so this is a real missing experiment rather than a renamed replay.
3. MTP is a better fit than more invasive ideas like SpinQuant/QuaRot-style int4 rotations or BitNet-style ternary training because it reuses the existing PyTorch training stack with only local code changes.

## Prior repo evidence that shaped this choice

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest available record in this checkout (`val_bpb: 1.1194`)
  - already carries the mature 11-layer stack: parameter banking + Parallel Muon, LeakyReLU^2, BigramHash, Partial RoPE, LN scale, VE, GPTQ-lite int6 export, sliding eval, legal TTT
- **Near-best non-TTT training stack:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - shows the current frontier is already highly optimized around quantization/export
- **Dormant MTP evidence:** several late-record `train_gpt.py` snapshots already contain MTP code paths, and `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md` explicitly runs with `MTP_NUM_HEADS=0`
- **Negative-result context:** the non-record 1x5090 sweep found naive layer recurrence actively harmful, so this candidate prefers a **sample-efficiency objective** over parameter-sharing/depth-reuse experiments

At implementation time there were **no prior `candidates/` directories** in the repository, so there was no earlier candidate iteration to inherit from or avoid.

## External research that informed it

- **Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_**  
  https://arxiv.org/abs/2404.19737

Why this paper matters here: it argues that predicting multiple future tokens can improve training efficiency and downstream quality, which matches this challenge's **fixed 600-second training cap** unusually well. This candidate keeps the simplest version of that idea: small auxiliary future-token heads during training only, with no exported artifact cost.

## What changed vs the chosen base

Relative to `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes only MTP-focused changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Warm-start auxiliary heads**
   - new `MTP_INIT_MODE=embed` default copies the tied embedding / LM head weights into each MTP head at init instead of starting them from zeros
3. **Horizon decay for auxiliary losses**
   - new `MTP_HEAD_DECAY=0.5` default geometrically downweights farther-future heads
4. **Logging**
   - logs the full MTP configuration so runs are easier to compare

The existing export path already excludes `mtp_heads.*`, so the added training heads remain **train-only** and do not count toward the saved artifact.

## How to run

From the repository root:

```bash
cd candidates/202604042345_mtp-warmstart

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_HEAD_DECAY=0.5 MTP_INIT_MODE=embed \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64 \
SWA_ENABLED=1 SWA_EVERY=50 LATE_QAT_THRESHOLD=0.15 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a cheaper first pass, keep the same command but set `TTT_ENABLED=0` so you can judge whether MTP helps the base training stack before paying the extra legal-TTT evaluation cost.

## Expected risks / tradeoffs

- **Step-time overhead:** extra vocabulary projections and CE losses reduce throughput, so MTP must beat the lost steps with better sample efficiency
- **Objective mismatch:** improving pre-quant next-token representations does not guarantee a better post-quant + sliding + TTT score
- **Warm-start bias:** copying the next-token head into farther-future heads may help short-run optimization, but it could also over-bias those heads toward the main task
- **Interaction risk:** the repo's frontier already layers EMA/SWA, late fake quant, XSA, and legal TTT; MTP could interact nonlinearly with any of them

## Validation

Commands run from the repository root during implementation:

1. `python -m compileall candidates/202604042345_mtp-warmstart/train_gpt.py`
   - **Passed**
2. Attempted a minimal CPU smoke test by importing the module with a mocked FlashAttention shim and instantiating a tiny GPT
   - **Blocked on this runner**: `ModuleNotFoundError: No module named 'torch'`

Because the current environment does not have PyTorch installed, and the full script also expects the repository's CUDA + FlashAttention training stack, I could not run a true local forward-pass or train-start smoke test here.
