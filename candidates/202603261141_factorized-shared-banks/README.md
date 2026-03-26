# Factorized Embeddings + Mirrored Shared Banks

## Hypothesis

The current best in-tree models already squeeze a lot out of quantization, EMA/SWA, partial RoPE, XSA, LeakyReLU^2, and score-first TTT. The remaining headroom may be in **using the 16 MB artifact budget more efficiently**, not in adding yet another expensive evaluation trick.

This candidate applies an **ALBERT-lite** idea to the strongest in-tree banked backbone:

- factorize the tied token embedding into `tok_emb[vocab, embed_dim]` plus `embed_up[embed_dim, model_dim]`,
- share the large attention/MLP bank weights across mirrored depths,
- keep **per-layer norms, scales, residual mixes, skips, VE scales, and XSA wiring untied** so logical depth and layer specialization remain intact.

The goal is to cut redundant bytes from the heaviest parameters while keeping the same 11-layer compute graph and the same training/eval protocol.

## Why it is promising for this repository

Repository evidence points in two directions:

- Embeddings are a persistent compression bottleneck. Early records improved when embeddings were protected during export, which suggests more efficient embedding parameterization is still a live lever.
- Later winning runs converged on a strong fixed-compute recipe (11 layers, seq2048, banked weights, EMA, XSA, VE, LeakyReLU^2). Reusing that recipe but reducing **unique** parameters is different from the depth-recurrence dead end, because compute depth stays fixed.

Compared with prior negative results on recurrent depth reuse, this candidate does **not** halve steps or add extra logical passes. It only shares the banked core matrices and leaves the per-layer wrappers untied.

## Prior records and candidates that influenced this

There were no prior `candidates/` directories in the repo when this candidate was created.

The main local influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current in-tree score,
  - parameter banking already cleanly separates large core weights from per-layer wrapper parameters,
  - ideal base for core-weight sharing without rewriting the whole trainer.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest pre-TTT-style backbone,
  - confirms the repo's stable winning stack before legal TTT.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - documents that naive layer recurrence was a dead end under fixed wallclock,
  - useful negative evidence for why this candidate keeps logical depth fixed.

## External research that informed it

- **ALBERT**: parameter sharing and factorized embeddings for compact Transformers  
  https://arxiv.org/abs/1909.11942
- **Universal Transformer**: repeated computation with shared transition functions  
  https://arxiv.org/abs/1807.03819
- **Adaptive Input Representations**: factorized input/output embeddings for language models  
  https://arxiv.org/abs/1809.10853

This candidate borrows the compact-model lesson, not the full original architectures:

- share only the big attention/MLP cores,
- keep per-layer wrapper parameters untied,
- keep the repository's existing training loop, quantization path, and evaluation protocol.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Main changes:

1. **Factorized tied embeddings**
   - `tok_emb` now stores `embed_dim` features instead of `model_dim`.
   - A learned `embed_up` projection maps token embeddings into model space.
   - The tied output path mirrors that factorization by projecting hidden states back through `embed_up^T` before the final vocabulary projection.

2. **Mirrored shared banks**
   - The large banked Q/K/V/O and MLP up/down tensors are allocated for `NUM_SHARED_BLOCKS` unique cores instead of one unique core per logical layer.
   - Logical layers index those shared cores through a mirrored schedule by default:
     - for 11 layers and 6 shared blocks: `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]`
   - Per-layer wrapper modules remain unique, so only the large matrix weights are shared.

3. **Slightly larger default BigramHash**
   - `BIGRAM_VOCAB_SIZE` defaults to `3072` instead of `2048`, using some of the saved parameter/artifact headroom on a known-good local feature.

The rest of the stack is intentionally inherited:

- LeakyReLU(0.5)^2 MLP,
- parameter banking + parallel Muon,
- partial RoPE,
- XSA on deep layers,
- VE on late layers,
- EMA averaging,
- GPTQ-lite-style int6 export,
- optional legal score-first TTT.

## How to run or evaluate it

From the candidate directory:

```bash
NUM_SHARED_BLOCKS=6 SHARED_BLOCK_MODE=mirror EMBED_DIM=160 \
BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

- disable TTT to measure pure backbone quality: `TTT_ENABLED=0`
- increase sharing pressure in mirror mode: `NUM_SHARED_BLOCKS=4` or `5`
- try lighter, non-mirrored sharing: `SHARED_BLOCK_MODE=group NUM_SHARED_BLOCKS=7` or `8`
- widen the factorized embedding: `EMBED_DIM=192`
- compare sharing layouts: `SHARED_BLOCK_MODE=group`

## Main expected risks or tradeoffs

- **Underfitting from sharing**: if 6 shared cores is too aggressive, the late layers may lose specialization despite untied wrappers.
- **Embedding bottleneck**: `EMBED_DIM=160` may be too small for lexical capacity even if it helps compression.
- **Retuning may be needed**: shared cores receive gradients from multiple logical depths, so the best LR/warmup settings may shift.
- **TTT interaction is uncertain**: a stronger compact backbone may help TTT, but sharing could also make post-training adaptation less stable.

## Validation

Successful local validation:

- `python -m compileall candidates/202603261141_factorized-shared-banks/train_gpt.py`
  - **Passed**

Attempted but not feasible in this environment:

- CPU-only structural smoke test via an inline Python harness with a stubbed `flash_attn_interface`
  - **Blocked** because the local environment does not have `torch` installed.
  - A meaningful runtime smoke test also depends on the normal CUDA + `flash_attn_interface` stack used by the record scripts, which is not available here.
