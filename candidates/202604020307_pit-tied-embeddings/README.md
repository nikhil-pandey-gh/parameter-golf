# PIT-lite tied embeddings on the legal-TTT record stack

## Hypothesis

The repository's strongest recent evidence says the tied token embedding / output head interface is still the most fragile part of the model under a 16MB artifact budget:

- `2026-03-18_FP16Embed_WD3600` showed that preserving the embedding in higher precision nearly eliminated the post-quantization gap.
- `2026-03-19_MixedQuant_Int6Int8_SlidingWindow` showed the token embedding needed gentler quantization than the block weights.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` is the current best overall stack, but it still uses ordinary weight tying.

This candidate keeps that winning 2026-03-23 stack intact and changes only the tied token interface: a **PIT-lite** diagonal positive transform is applied inversely on input embeddings and forward on output logits, while training adds a small column-orthogonality regularizer on the shared token memory.

The goal is to make the tied interface behave more like a pseudo-inverse-coupled encoder/decoder pair, without paying the artifact cost of a full dense hidden-space transform.

## Why this is promising here

This repo has repeatedly found that:

1. embedding precision matters disproportionately,
2. quantization-aware improvements are strongest when they target sensitive tensors, and
3. cheap, local changes that preserve throughput tend to win over broad architectural rewrites.

PIT-lite fits those constraints:

- it adds only one extra per-dimension control vector (`pit_log_scale`),
- it preserves the current 11-layer / XSA / legal-TTT training and evaluation flow,
- it keeps export simple because the large shared token table is still quantized the same way,
- and it directly targets the most fragile tensor family instead of adding more depth or slower attention.

## Prior records that informed it

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Embedding sensitivity:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
- **Mixed quantization / embedding protection:** `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
- **Best pre-TTT training stack:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

## External research

- Jian Gu, Aldeida Aleti, Chunyang Chen, Hongyu Zhang, **"Rethinking Weight Tying: Pseudo-Inverse Tying for Stable LM Training and Updates"**, arXiv:2602.04556, 2026. <https://arxiv.org/abs/2602.04556>

This candidate does **not** implement the full paper. Instead it adapts the core idea to this repo's constraints:

- shared token memory,
- a positive hidden-space transform around the tied interface,
- and explicit pressure toward a more stable token basis.

The simplification is deliberate so the candidate stays compatible with the current artifact budget and record script structure.

## What changed vs the chosen base

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`:

1. Added PIT-lite hyperparameters:
   - `PIT_ENABLED=1`
   - `PIT_ORTHO_REG=5e-5`
   - `PIT_SCALE_INIT=1.0`
2. Kept the shared `tok_emb.weight`, but changed the tied interface:
   - input embedding path uses `tok_emb / scale`
   - output logits path uses `(hidden * scale) @ tok_emb^T`
3. Added a small orthogonality penalty on the shared token memory columns during training.
4. Marked `pit_log_scale` as a control tensor so it stays in floating-point through export / roundtrip.
5. Threaded the new PIT-lite settings through training, export, reload, and final evaluation.

Everything else is intentionally inherited from the 2026-03-23 record stack: parameter banking, Parallel Muon, XSA, partial RoPE, bigram features, value embeddings, legal score-first TTT, and int6+lzma export.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
PIT_ENABLED=1 PIT_ORTHO_REG=5e-5 PIT_SCALE_INIT=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a cleaner ablation against the base stack, rerun the same command with PIT disabled:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
PIT_ENABLED=0 PIT_ORTHO_REG=5e-5 PIT_SCALE_INIT=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected risks / tradeoffs

- The diagonal PIT-lite transform is much cheaper than full PIT, but also weaker.
- The orthogonality regularizer may fight the repo's existing weight decay if its coefficient is too high.
- If the main embedding issue is export precision rather than interface drift, this may help less than a more aggressive quantization change.
- Because the legal-TTT base is already strong, gains may be small and require multiple seeds to resolve.

## Validation

Commands run in this repository:

```bash
python -m compileall train_gpt.py
python -m compileall candidates/202604020307_pit-tied-embeddings/train_gpt.py
```

Outcomes:

- both syntax checks completed successfully.
- A deeper runtime smoke test was **not feasible in this runner** because the environment did not have the repo's runtime Python dependencies installed (`torch` was missing), and this record-derived script targets a CUDA + FlashAttention execution path.
