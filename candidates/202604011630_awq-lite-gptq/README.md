# Candidate: AWQ-lite GPTQ clip search

## Hypothesis

The current 11-layer int6 stack is already very strong on the training side, so the next cheap win is more likely to come from **post-training quantization quality** than from another training-time feature. This candidate keeps the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` architecture and training recipe, but replaces its weight-only GPTQ-lite clip selection with a **training-activation-calibrated clip search**.

Concretely: instead of choosing each per-row int6 clip percentile by minimizing weight reconstruction MSE, this candidate runs a tiny calibration pass on **training tokens only** and picks the clip percentile that minimizes **output reconstruction error** for each quantized linear weight. The expected upside is a smaller int6 roundtrip gap with essentially zero training-step overhead.

## Why this is promising for this repository

Recent records in this repo keep showing the same pattern:

- sliding-window eval was a large early gain,
- 11-layer / 3x-MLP / XSA / EMA / Partial-RoPE stacks are now fairly mature,
- later improvements have come from **compression-aware details** such as fp16 embeddings, int6 mixed quantization, and GPTQ-lite export tweaks.

That makes this repo a good fit for an **export-only** idea. It attacks the same bottleneck as the best recent records without spending the 10-minute training budget on a riskier architecture rewrite or heavier QAT schedule.

## Which records or prior candidates influenced it

There were **no prior `candidates/` folders** in the repo when this candidate was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - strongest no-TTT base for this candidate
  - showed that a better post-training clip search can move BPB with zero training cost
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - confirmed Partial RoPE + LN scale are strong enough to keep as part of the base stack
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - established the 11L/XSA/EMA/int6 direction
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - useful evidence that the frontier has become incremental and that it is reasonable to target exporter quality rather than overhaul the whole trainer

## External research that informed it

This candidate is most directly inspired by the observation from recent PTQ work that **activation statistics matter** for low-bit weight quantization:

- **GPTQ**: one-shot weight quantization using reconstruction-aware criteria rather than naive weight-only scaling  
  <https://arxiv.org/abs/2210.17323>
- **SmoothQuant**: move quantization difficulty using offline, mathematically equivalent transformations driven by activation outliers  
  <https://arxiv.org/abs/2211.10438>
- **AWQ**: protect the most salient channels using activation-aware calibration, without backpropagation through the calibration set  
  <https://arxiv.org/abs/2306.00978>

This implementation is deliberately lighter than full AWQ or SmoothQuant. It keeps the repo's existing int6 export format and simply upgrades the clip search objective from weight MSE to activation-conditioned output MSE.

## What changed versus the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Activation-aware clip search for int6 export**
   - added a small calibration pass over training tokens
   - captured inputs to quantized `CastedLinear` modules using forward pre-hooks
   - for each candidate clip percentile, measured output reconstruction error on the sampled inputs
   - picked the percentile with the lowest activation-conditioned output MSE

2. **New PTQ knobs**
   - `PTQ_CALIBRATION_BATCHES`
   - `PTQ_CALIBRATION_TOKENS`
   - `PTQ_CALIBRATION_ROWS`
   - `PTQ_CLIP_PERCENTILES`

3. **Local validation improvements**
   - added a PyTorch SDPA fallback when `flash_attn_interface` is unavailable
   - added `SMOKE_TEST=1` mode so the script can do a tiny random-token forward pass without dataset or tokenizer setup

The scored training/eval path is otherwise intentionally close to the base record.

## How to run or evaluate it

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
PTQ_CALIBRATION_BATCHES=4 PTQ_CALIBRATION_TOKENS=32768 PTQ_CALIBRATION_ROWS=1024 \
PTQ_CLIP_PERCENTILES=0.995,0.999,0.9995,0.9999,1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- calibration uses **training shards**, not validation shards
- the activation-aware export happens after the usual EMA-based training run
- `SMOKE_TEST=1 python train_gpt.py` is a no-data local sanity check and is not part of the scored path

## Validation commands and outcomes

Validated in this repository with:

```bash
python -m compileall candidates/202604011630_awq-lite-gptq/train_gpt.py
```

Outcome: **passed**

```bash
python -m venv /tmp/pgolf-venv
/tmp/pgolf-venv/bin/pip install numpy torch sentencepiece
CUDA_VISIBLE_DEVICES='' SMOKE_TEST=1 /tmp/pgolf-venv/bin/python train_gpt.py
```

Outcome: **passed** with:

```text
smoke_test_ok loss:4.8494 device:cpu flash_attn:0
```

## Main expected risks or tradeoffs

- The gain may be small if the existing GPTQ-lite row-MSE search already captures most of the available benefit.
- Calibration adds some exporter-time cost, even though it does not change training-step cost.
- This is **not** full AWQ or SmoothQuant: it uses activation-aware selection inside the repo's existing per-row int6 format instead of introducing channel-folding transforms across residual/tied paths.
- If this direction helps, the natural next follow-up is selective channel rescaling or MLP-only smoothing on top of the same calibration pass.
