# Boundary-sensitive GPTQ-lite + LeakyReLU(0.5)^2

## Hypothesis

The current repo evidence says the remaining easy wins mostly come from quantization quality and evaluation quality, not from large architectural rewrites. My hypothesis is that the strongest low-risk next step is to start from the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack, keep its training recipe intact, swap in the `LeakyReLU(0.5)^2` activation that helped the 2026-03-23 record, and make the export quantizer layer-sensitive so the shallowest and deepest transformer blocks get more forgiving post-training quantization than the middle blocks.

## Why it is promising for this repository

Three repo patterns point in the same direction:

- Quantization quality has repeatedly been worth real BPB, especially once the model is already near the 16 MB artifact limit.
- The 2026-03-23 record shows that `LeakyReLU(0.5)^2` is a genuinely useful one-line MLP improvement on this family of models.
- The best non-TTT stacks are already close enough that a cleaner export can plausibly matter more than another broad architectural change.

The external research lines up with that:

- **SliderQuant** ([arXiv:2603.25284](https://arxiv.org/abs/2603.25284)) reports that shallow and deep layers are more quantization-sensitive than middle layers, with the first and last layers being the most sensitive.
- **Dissecting Quantization Error: A Concentration-Alignment Perspective** ([arXiv:2603.04359](https://arxiv.org/abs/2603.04359)) argues that reducing quantization error is not only about clipping harder; structure and alignment matter, which supports spending the precision budget where it is most valuable.

This candidate translates that into a repo-native, minimal change: keep the interior `attn` and `mlp` weights on GPTQ-lite int6, but promote the boundary transformer blocks to GPTQ-searched int8 while keeping the same compact serialization format.

## Which records influenced it

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`

Direct influences:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - Borrowed the `LeakyReLU(0.5)^2` activation idea.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - Confirms that the 11-layer partial-RoPE + LN-scale stack is strong before extra eval-time complexity.
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow`
  - Earlier evidence that selective precision allocation is worthwhile in this challenge.

There were no prior local `candidates/` directories at implementation time.

## External research that informed it

- **SliderQuant: Accurate Post-Training Quantization for LLMs** ([arXiv:2603.25284](https://arxiv.org/abs/2603.25284))
  - Key takeaway used here: quantization sensitivity is not uniform across layers; shallow/deep layers deserve different treatment.
- **Dissecting Quantization Error: A Concentration-Alignment Perspective** ([arXiv:2603.04359](https://arxiv.org/abs/2603.04359))
  - Key takeaway used here: quantization error structure matters, so targeted layer-wise choices can be more valuable than a single global rule.

## What changed versus the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **LeakyReLU(0.5)^2 MLP**
   - Replaces `relu^2` with `leaky_relu(x, 0.5)^2`.
   - This imports the strongest low-cost activation improvement already seen in the repo.

2. **Boundary-sensitive GPTQ-lite export**
   - Adds `QUANT_BOUNDARY_LAYERS` (default `1`).
   - Transformer `attn` and `mlp` weights in the first and last `QUANT_BOUNDARY_LAYERS` blocks are quantized with GPTQ-searched int8 (`clip_range=127`).
   - Interior transformer `attn` and `mlp` weights stay on GPTQ-lite int6 (`clip_range=31`).
   - This keeps the repo’s compact serialization structure while making the precision assignment layer-aware.

3. **CPU-compatible smoke path**
   - Adds `SMOKE_TEST=1` mode that skips CUDA/distributed setup, runs a tiny CPU forward pass, and round-trips the new quantizer.
   - Adds an SDPA fallback when FlashAttention is unavailable or the model is running on CPU.

## How to run or evaluate it

### Main training/eval run

Run from this candidate directory on the standard GPU environment:

```bash
MLP_NEGATIVE_SLOPE=0.5 \
QUANT_BOUNDARY_LAYERS=1 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Local smoke check

After installing the repo dependencies:

```bash
SMOKE_TEST=1 python train_gpt.py
```

This does not validate the full 8xH100 training path, but it does validate model construction, CPU forward execution, and the boundary-sensitive quantization/dequantization roundtrip.

## Validation commands and outcomes

Validated locally in this workflow with:

```bash
/tmp/gh-aw/agent/pgolf-venv/bin/python -m compileall candidates/202603292009_boundary-gptq-lrelu2/train_gpt.py
SMOKE_TEST=1 /tmp/gh-aw/agent/pgolf-venv/bin/python candidates/202603292009_boundary-gptq-lrelu2/train_gpt.py
```

Observed outcome:

```text
Compiling '.../candidates/202603292009_boundary-gptq-lrelu2/train_gpt.py'...
smoke_test:ok loss=4.858056 quant_summary={'int8_boundary': 8, 'int6': 8}
```

Notes:

- The smoke check ran in a temporary venv because the base container did not ship with `numpy`, `sentencepiece`, or `torch`.
- No full GPU training run was feasible in this environment.

## Main expected risks or tradeoffs

- The benefit may be smaller than the 2026 papers suggest because this challenge already uses strong per-row quantization and very small models.
- Promoting boundary blocks to int8 may help roundtrip quality but could slightly hurt compression ratio, so the artifact margin needs real GPU-run confirmation.
- This candidate does **not** include test-time training, so even if it improves the clean 03-22 stack, it may still trail the best TTT-enabled record.
- The CPU smoke path is for correctness only; it says nothing about wallclock performance on the actual challenge hardware.
