# Factorized FP16 Lexical Embedding on the 11L TTT Stack

## Hypothesis

The current best stack already squeezes most gains out of the transformer body, but prior records show the tied token embedding is unusually sensitive to quantization because it serves as both the input table and the output head. The candidate factorizes that tied embedding/head path through a smaller lexical subspace (`EMBED_DIM`, default `192`) so the lexical table can stay fp16 at export time while the rest of the model stays on the existing mixed int6/int8 path.

## Why this is promising for this repository

- `2026-03-18_FP16Embed_WD3600` found that keeping `tok_emb.weight` in fp16 nearly erased the quantization gap, but it had to shrink the model to afford the extra bytes.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` and `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` showed the winning direction is now the full 11-layer stack with strong export logic, EMA, and legal TTT rather than broader architecture churn.
- This candidate keeps that stronger 11L base intact and only changes the lexical path, which is one of the few remaining repo-specific weak points called out by the record history.

## Prior work that influenced this candidate

- **Primary base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Embedding sensitivity evidence:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
- **Export/quantization direction:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- **Prior candidates reviewed:** none; there was no `candidates/` directory when this run started.

## External research that informed it

- **ALBERT** (factorized embedding parameterization): https://arxiv.org/abs/1909.11942
- **GPTQ** (accurate post-training quantization for GPT-style models): https://arxiv.org/abs/2210.17323
- **SmoothQuant** (outlier-aware accuracy-preserving PTQ): https://arxiv.org/abs/2211.10438
- **LLM.int8()** (mixed-precision handling for emergent outlier features): https://arxiv.org/abs/2208.07339

ALBERT provides the direct architectural inspiration: decouple lexical embedding width from transformer width. The quantization papers reinforce the repo's own lesson that the most sensitive tensors should not always follow the same export rule as the rest of the model.

## What changed versus the chosen base

1. Added `EMBED_DIM` and `EMBED_EXPORT_FP16` hyperparameters.
2. Replaced the full-width tied embedding with:
   - a lexical table `tok_emb` of shape `[vocab_size, embed_dim]`
   - a learned projection `emb_proj` from `embed_dim -> model_dim`
3. Replaced tied output logits with the mirrored factorized path:
   - hidden state -> `emb_proj.T` -> lexical subspace -> tied `tok_emb.weight`
4. Added `emb_proj.weight` to the embedding optimizer group.
5. Modified mixed export so `tok_emb.weight` can stay fp16 while `emb_proj.weight` still follows the regular int8 path.

## How to run or evaluate it

Use the same high-performing recipe as the 2026-03-23 record, but add the factorized lexical path:

```bash
cd candidates/202604031427_factorized-fp16-embed

BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
EMBED_DIM=192 EMBED_EXPORT_FP16=1 \
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

Recommended first follow-up sweep:

- `EMBED_DIM in {192, 224, 256}`
- `EMBED_EXPORT_FP16 in {0, 1}`

## Main expected risks and tradeoffs

- The factorized lexical subspace may become a low-rank bottleneck if `EMBED_DIM` is too small.
- Keeping the lexical table fp16 may still push the artifact over 16MB if `EMBED_DIM` is increased too aggressively.
- The best post-TTT score may not track the best pre-TTT score perfectly, so this should be judged on both roundtrip export metrics and final legal TTT evaluation.

## Validation

- `python -m compileall candidates/202604031427_factorized-fp16-embed/train_gpt.py` — passed in this workflow runner.
- A truthful CPU smoke test was **not feasible here** because the runner does not have PyTorch installed (`ModuleNotFoundError: No module named 'torch'`), so runtime validation beyond syntax compilation could not be completed without first provisioning the full training stack.
