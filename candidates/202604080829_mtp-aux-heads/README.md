# MTP auxiliary heads on the 11L EMA + GPTQ-lite stack

## Hypothesis

The strongest train-only line in this repo has already saturated most obvious architecture and quantization wins, so the next clean lever is a **training-only sample-efficiency boost** rather than another export-time trick. Enabling a lightweight multi-token prediction (MTP) auxiliary loss should improve representation learning during the fixed 10-minute budget while leaving the exported artifact unchanged, because the extra heads are never serialized.

## Why this is promising for this repository

- The best non-TTT stack (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`) already contains dormant MTP code paths, but every reviewed record kept `MTP_NUM_HEADS=0`.
- Recent repo history says the mature 11-layer line still responds to small train-time improvements (`Partial RoPE`, `LN Scale`, `EMA`, `GPTQ-lite`, `LeakyReLU^2`), so a train-only auxiliary objective is a better fit than revisiting dead-end shared-depth experiments.
- MTP adds no submission bytes because the auxiliary heads are excluded from export, which is unusually attractive under the 16MB artifact cap.

## Prior records and candidates that influenced this

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best train-only stack reviewed: 11L, XSA, Partial RoPE, LN Scale, SmearGate, BigramHash, EMA, warmdown 3500, GPTQ-lite, late QAT.
- **Nearby evidence:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Shows this family can still move from small train-time changes even after the major architecture choices have stabilized.
- **Rejected direction:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Its exploratory sweep reported a strong negative result for naive layer recurrence, which is why this candidate prefers a training-only auxiliary objective over shared-depth reuse.
- **Prior candidates:** none existed before this run.

## External research that informed this

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737)  
  https://arxiv.org/abs/2404.19737  
  Key takeaway: predicting multiple future tokens with auxiliary output heads can improve sample efficiency and downstream quality with little training-time penalty.
- DeepSeek-AI, **"DeepSeek-V3 Technical Report"** (arXiv:2412.19437)  
  https://arxiv.org/abs/2412.19437  
  Key takeaway: MTP remains relevant in modern strong-language-model training recipes, even when the main architecture differs substantially from this repo.

## What changed versus the chosen base implementation

1. **Enabled MTP by default**
   - `MTP_NUM_HEADS` now defaults to `1`.
   - `MTP_LOSS_WEIGHT` now defaults to `0.15`.
   - This adds one future-token auxiliary head during training while keeping next-token evaluation unchanged.
2. **Made MTP explicitly train-only**
   - `mtp_heads.*` are now excluded not just from final export, but also from EMA and SWA bookkeeping.
   - That avoids spending averaging/memory budget on parameters that are never used during evaluation or serialization.
3. **Made the candidate runnable from its own directory**
   - Default `DATA_PATH` and `TOKENIZER_PATH` resolve relative to the repository root instead of the current working directory.
4. **Added validation-friendly fallbacks**
   - Attention falls back to PyTorch SDPA when FlashAttention is unavailable.
   - `SMOKE_TEST=1` runs a tiny synthetic forward/backward pass without CUDA, datasets, or tokenizers.

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202604080829_mtp-aux-heads
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate uses the repo dataset/tokenizer defaults automatically. Override them only if your local paths differ:

```bash
cd candidates/202604080829_mtp-aux-heads
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Minimal CPU smoke check:

```bash
SMOKE_TEST=1 VOCAB_SIZE=128 NUM_LAYERS=2 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2 TRAIN_SEQ_LEN=32 EVAL_SEQ_LEN=32 BIGRAM_VOCAB_SIZE=64 BIGRAM_DIM=32 \
VE_ENABLED=0 XSA_LAST_N=0 ROPE_DIMS=16 MTP_NUM_HEADS=1 python train_gpt.py
```

## Validation

The workflow runner did not have the repo's Python stack preinstalled, so the smoke check used an isolated temporary venv.

| Command | Outcome |
|---|---|
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604080829_mtp-aux-heads/train_gpt.py` | Passed |
| `SMOKE_TEST=1 VOCAB_SIZE=128 NUM_LAYERS=2 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 TRAIN_SEQ_LEN=32 EVAL_SEQ_LEN=32 BIGRAM_VOCAB_SIZE=64 BIGRAM_DIM=32 VE_ENABLED=0 XSA_LAST_N=0 ROPE_DIMS=16 MTP_NUM_HEADS=1 /tmp/gh-aw/agent/pgolf-venv/bin/python candidates/202604080829_mtp-aux-heads/train_gpt.py` | Passed: `smoke_test:ok device:cpu loss:5.5858 logits_shape:(2, 32, 128) mtp_num_heads:1` |

## Main expected risks and tradeoffs

- **Throughput risk:** the extra auxiliary head adds some training compute, so any quality gain must outweigh a possible drop in step count.
- **Objective mismatch:** future-token auxiliary losses can help the trunk, but they can also pull optimization away from the next-token objective if the weight is too high.
- **Small-model uncertainty:** MTP is well-motivated by recent literature, but the gains there were mostly reported on much larger models than this repo's tiny 11-layer stack.
