# Candidate: Single-Step Multi-Token Prediction on the LeakyTTT Stack

## Hypothesis

Enable a **single auxiliary multi-token prediction (MTP) head** during training on top of the current best `LeakyReLU(0.5)^2 + legal TTT + Parallel Muon` stack. The extra head should improve sample efficiency and representation quality during the fixed 10-minute training budget, while preserving the exported artifact budget because the auxiliary head is excluded before serialization.

## Why this is promising for this repository

Recent winning runs in this repository already stack together:

- stronger compact architectures (`11L`, `3x` MLP, XSA, partial RoPE, VE),
- smoother post-training/export paths (EMA, SWA, GPTQ-lite / int6 roundtrip),
- and legal evaluation-time adaptation (TTT).

What is still missing in the public run history is a **training-only objective improvement** that does not consume permanent artifact bytes. This candidate targets exactly that gap.

The repository also contains an unexploited clue: the MTP code path has existed since the 2026-03-20 line of records, but the public runs we inspected still logged `mtp_num_heads:0`, including the 2026-03-22 and 2026-03-23 record stacks. So this is a real, evidence-based unexplored branch rather than a random new feature.

## Prior records and experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - best current public stack,
  - shows LeakyReLU(0.5)^2, legal score-first TTT, parameter banking, and the 11-layer/XSA/partial-RoPE recipe.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - establishes the strong pre-TTT architecture and the EMA/GPTQ-lite export path.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/train_gpt.py`
  - already contains MTP plumbing plus export-time stripping of `mtp_heads`, which makes the idea especially cheap to try here.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - useful negative result: recurrence hurt because reduced step count dominated any capacity benefit. That pushed this candidate toward a minimal **one-head** MTP variant to keep throughput impact small.

## External research that informed the choice

- **Gloeckle et al., “Better & Faster Large Language Models via Multi-Token Prediction”** (`arXiv:2404.19737`)
  - argues that predicting multiple future tokens from a shared trunk improves sample efficiency and downstream quality.
- **DeepSeek-V3 Technical Report** (`arXiv:2412.19437`)
  - explicitly includes a multi-token prediction training objective in a modern high-performing LLM recipe, which is evidence that the objective still looks attractive in contemporary stacks.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Copied that script into this candidate directory.
2. Switched the default MTP configuration from disabled to:
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`
3. Wired `mtp_heads` into the replicated AdamW path so the enabled auxiliary head is actually optimized during training.
4. Left the export path intact, which already removes `mtp_heads` before saving the final model.

This keeps the change intentionally small:

- no root-script edits,
- no new infrastructure,
- no permanent artifact expansion from the auxiliary head weights,
- and no change to the legal TTT evaluation protocol.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 \
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

## Validation

Commands run in this workflow:

- `python -m compileall candidates/202603301032_single-step-mtp/train_gpt.py`
- `python - <<'PY' ...` dependency/data probe for `torch`, `sentencepiece`, `numpy`, `flash_attn_interface`, tokenizer files, and FineWeb validation shards

Outcome:

- syntax compilation succeeded.
- runtime smoke prerequisites were **not** present in this workflow environment:
  - `torch`, `sentencepiece`, `numpy`, and `flash_attn_interface` were all unavailable,
  - no tokenizer `.model` files were present under `data/`,
  - no `fineweb_val_*.bin` shards were present locally.

Minimal CPU-only runtime smoke test:

- Not feasible to run honestly in this workflow environment because both the required Python runtime dependencies and the local dataset/tokenizer artifacts were missing.

## Main expected risks and tradeoffs

- **Step-time regression**: even one auxiliary head adds extra full-vocab logits during training. If throughput falls too much, the extra supervision may not pay for itself.
- **Small-model transfer risk**: the published MTP results are strong, but most of that evidence comes from much larger models than this challenge.
- **Objective interference**: the auxiliary loss could fight with the later quantization-aware/export-tuned behavior or change what TTT adapts to.
- **Diminishing returns under TTT**: if legal TTT already harvests most of the available easy win at eval time, pretraining-side gains from MTP may be partly masked.
