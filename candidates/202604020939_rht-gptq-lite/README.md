# RHT-GPTQ-lite

## Hypothesis

Applying a self-inverse blockwise randomized Hadamard rotation to large int6 weight matrices before GPTQ-lite clip search should flatten per-row outliers, reduce post-quantization reconstruction error, and improve final validation bits-per-byte without spending extra training-time budget.

## Why it is promising for this repository

- The strongest non-TTT record, [`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`](../../records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md), already showed that small post-training quantization improvements still move BPB on top of the mature 11-layer stack.
- The record progression from [`2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`](../../records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md) to [`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`](../../records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md) to the 2026-03-22 record suggests the training-side architecture is already close to a local frontier, so the next cheap lever is better int6 export.
- The current top run, [`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`](../../records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md), wins mostly by stacking evaluation-time complexity on top of the same family of models. This candidate instead tries to improve the artifact itself while keeping the simpler non-TTT training path.

## Prior experiments that influenced this candidate

- **Chosen base:** [`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`](../../records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md)
  - best non-TTT stack
  - already contains GPTQ-lite percentile search, EMA, warmdown 3500, Partial RoPE, LN scale, XSA4, VE128, SmearGate, and BigramHash
- **Architectural carry-over:** [`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`](../../records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md)
  - confirms Partial RoPE + LN scale are durable zero-parameter wins
- **No prior candidates existed** when this folder was created, so there was no earlier candidate stack to reuse or avoid.

## External research that informed it

- **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)) argues that orthogonal rotations remove outliers without changing the full-precision function, making low-bit quantization easier; notably, it reports essentially lossless 6-bit and 8-bit quantization for LLaMA-family models.
- **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)) shows that some random rotations quantize much better than others, and that rotation choice matters enough to materially change downstream accuracy.

This candidate takes the most repository-friendly slice of those ideas: no learned rotations, no activation/KV machinery, just a deterministic self-inverse randomized Hadamard transform for weight-only int6 export.

## What changed versus the chosen base implementation

1. Added a blockwise randomized Hadamard search for int6 matrices during export.
2. For each large int6 matrix, the script now compares:
   - plain GPTQ-lite percentile search,
   - block-RHT with block size 256,
   - block-RHT with block size 512.
3. The winner is chosen by reconstruction MSE **after** undoing the rotation back into the original weight space.
4. The selected rotation metadata is stored in the serialized artifact and inverted during dequantization.
5. Script defaults for `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root, so `train_gpt.py` can be run directly from this candidate directory without extra path overrides.

## How to run or evaluate it

From this directory:

```bash
cd candidates/202604020939_rht-gptq-lite
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
INT6_ROTATION_BLOCKS=256,512 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The default dataset and tokenizer paths now point at the repository root's `data/` folder.
- If those files are not present yet, populate them first, for example from the repository root with `python3 data/cached_challenge_fineweb.py --variant sp1024`, or from this directory with `python3 ../../data/cached_challenge_fineweb.py --variant sp1024`.
- Override `INT6_ROTATION_BLOCKS` if you want to ablate the idea, for example `INT6_ROTATION_BLOCKS=` to disable the rotation search.

## Validation

### Commands

From the repository root:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604020939_rht-gptq-lite/train_gpt.py
```

### Outcomes

- `compileall` succeeded for the root scripts, `data/`, and this candidate script.
- A minimal CPU smoke run was **not** executed. This script imports FlashAttention and hard-requires CUDA/NCCL in its execution path, so there is no safe CPU-only start command in this environment without changing repository infrastructure.

## Main risks and tradeoffs

- Reconstruction MSE is only a proxy for final BPB, so the chosen rotation can still be neutral or slightly harmful on the full validation metric.
- Export and dequantization become a bit slower because each large int6 matrix may be quantized several times during the search.
- A learned-rotation approach in the style of SpinQuant could outperform this lightweight version, but that would add much more training-time and implementation complexity than this repository-friendly candidate is trying to spend.
