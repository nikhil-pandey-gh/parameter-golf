# Candidate: Single-Head Train-Only MTP on the 03-23 Stack

## Hypothesis

Add a **single auxiliary multi-token prediction (MTP) head** during training to improve sample efficiency at a fixed 600-second budget, while keeping the final artifact unchanged by stripping the auxiliary head before export.

The key bet is that Parameter Golf's frontier is already saturated with quantization, EMA/SWA, XSA, partial RoPE, and evaluation tricks, but it has **not actually tried a live train-only MTP objective** on top of the current best stack. If MTP improves representation learning early enough, the main next-token head should arrive at a better checkpoint before the wallclock cap without paying any submission-byte cost.

## Why this is promising for this repository

Repository review suggested three useful facts:

1. The biggest recent gains came from **stacking small efficiency improvements** on top of a strong 11-layer compressed model, not from replacing the whole architecture.
2. The repo already contains dormant MTP code paths, including logic that **excludes `mtp_heads` from export**, so this idea fits the existing design.
3. The latest 03-23 record script carries MTP code but does **not** wire those heads into an optimizer, so simply turning the flag on there would not actually train them.

That makes train-only MTP unusually attractive here: it is differentiated from prior records, cheap to implement, and aligned with the artifact budget.

## Prior records that influenced this candidate

- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`**
  Chosen as the main base because it is the current best overall stack: LeakyReLU^2, legal score-first TTT, parameter banking, Parallel Muon, partial RoPE, VE, EMA/SWA, and GPTQ-lite int6 export.

- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`**
  Important because it already wired `mtp_heads` into optimization and excluded them from export, which confirmed the train-only MTP path is viable in this repo.

- **`records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`**
  Useful cautionary note: feature toggles that rely on mutable class flags can be optimized away by `torch.compile`, so this candidate avoids that pattern.

- **`records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`**
  Negative evidence against naive layer recurrence/depth reuse, which made a lower-risk objective-side idea more appealing than another parameter-sharing attempt.

## External research that informed it

- **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (2024)**
  <https://arxiv.org/abs/2404.19737>
  Core motivation: predict multiple future tokens with independent heads on top of a shared trunk to improve sample efficiency.

- **Guoliang Zhao et al., "Self-Distillation for Multi-Token Prediction" (2026)**
  <https://arxiv.org/abs/2603.23911>
  Useful design constraint: jointly training many MTP heads can be harder than it looks, so this candidate stays conservative with **one** auxiliary head and a modest loss weight.

I also reviewed more invasive ideas such as ALBERT-style sharing, Universal Transformer recurrence, DeepNorm, and rotation-based quantization. Those remain interesting, but they are materially larger code changes and higher validation risk for a single candidate iteration.

## What changed versus the chosen base

Base implementation: **03-23 LeakyReLU^2 + Legal TTT + Parallel Muon**

Changes:

1. **Enable one train-only MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`

2. **Fix the 03-23 optimizer wiring**
   - The 03-23 script instantiated `mtp_heads` and excluded them from export, but did not add their weights to any optimizer.
   - This candidate routes `mtp_heads.parameters()` into the replicated AdamW path so the auxiliary head is actually trained.

3. **Keep export behavior unchanged**
   - `mtp_heads` are still stripped before serialization, so the auxiliary objective adds **training-time parameters only** and no final artifact bytes.

Everything else stays on the proven 03-23 stack: LeakyReLU^2 MLP, parameter banking, Parallel Muon, partial RoPE, XSA on late layers, VE, EMA/SWA, legal TTT, and GPTQ-lite int6 export.

## How to run

From this candidate directory:

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
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For evaluation, the most important final metrics to inspect are:

- `final_int8_zlib_roundtrip_exact val_bpb:...`
- `legal_ttt_exact val_bpb:...`

The latter is the score-first TTT result on the exported round-tripped model.

## Validation performed

Commands run in this repository:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604012147_single-head-mtp/train_gpt.py
```

Outcome:

- Both compile-only checks passed.
- A CPU smoke launch was **not** feasible in this environment because this script hard-requires CUDA and imports `flash_attn_interface`, so there is no safe no-GPU execution path to validate end-to-end startup here.

## Main risks and tradeoffs

- **Step-time regression:** even one extra MTP head adds logits and loss work every step, so gains from better sample efficiency must exceed the lost steps.
- **Objective interference:** the auxiliary head could slightly hurt the main next-token head if `MTP_LOSS_WEIGHT` is too large.
- **Stack complexity:** this keeps the full 03-23 TTT + Parallel Muon stack, so any win or loss will still be entangled with a fairly sophisticated base.
- **Optimizer choice for MTP heads:** these small train-only matrices now use the replicated AdamW path; that is simple and robust, but it has not yet been tuned specifically for MTP in this repo.

## Expected next experiments if this works

1. Sweep `MTP_LOSS_WEIGHT` in a narrow band such as `0.05`, `0.1`, `0.15`.
2. Try `MTP_NUM_HEADS=2` only if the one-head variant is clearly positive.
3. If MTP helps but step-time is tight, test it on the 03-22 EMA/GPTQ-lite base to measure whether the gain is coming from the objective itself or from interactions with the 03-23 stack.
