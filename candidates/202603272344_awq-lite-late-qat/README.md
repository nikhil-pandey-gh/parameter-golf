# AWQ-lite + Real Late QAT

## Hypothesis

The strongest 11-layer stacks in this repository are already close in pre-quant quality, so the next cheap win is likely to come from the **export path**, not from a broad architecture rewrite. This candidate tests whether a small amount of **activation-aware calibration** plus a **real late-QAT path that survives `torch.compile`** can tighten the remaining int6 export gap without adding much infrastructure.

In short: keep the proven 11L stack, add one low-risk training fix and one low-risk export fix, and spend complexity where the repo has repeatedly shown it matters most.

## Why it is promising for this repository

Several repo trends point in the same direction:

- The biggest quality jumps came from **quantization-aware design**, not from pure train-loss improvements.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly documents that its late-QAT switch was **dead under `torch.compile`**.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` got a real win from a better **post-training clip search**, which suggests the export path is still fertile.
- The non-record 4-hour run shows that **training longer alone is not enough**; quantization remains a bottleneck.
- The latest record adds **LeakyReLU(0.5)^2** as a strong, low-cost MLP improvement, so it is worth carrying forward while testing a new export idea.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Base stack for this candidate: 11L, 3x MLP, EMA, XSA4, partial RoPE, LN scale, VE, BigramHash, SmearGate, GPTQ-lite export.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Important warning that the old late-QAT toggle was constant-folded away and therefore not actually active.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Supplies the low-risk `LeakyReLU(0.5)^2` activation change.
- `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/`
  - Reinforces the repo-wide pattern that training for export robustness pays off.

## External research that informed it

- **AWQ** — *Activation-aware Weight Quantization* (`arXiv:2306.00978`)
  - Motivation for using activation statistics, not just raw weight error, when deciding how to quantize a layer.
- **GPTQ** (`arXiv:2210.17323`)
  - Motivation for improving one-shot post-training quantization with a more output-aware error objective.
- **LSQ** — *Learned Step Size Quantization* (`arXiv:1902.08153`)
  - Motivation for treating the quantizer as part of training rather than a hard post-hoc toggle; this candidate uses that idea in a lightweight form via a ramped late-QAT blend.

I considered a QuaRot/SpinQuant-style rotation candidate as well, but that would have required a broader and riskier refactor of the current stack. This candidate keeps the integration surface much smaller while still moving in the same research direction: reduce export error by becoming more activation-aware.

## What changed versus the chosen base implementation

Base: `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Real late QAT on selected late layers**
   - Replaces the compile-foldable class-bool QAT gate with per-module tensor `qat_alpha` buffers.
   - Only targets late transformer attention/MLP matrices (`QAT_LAYER_START`, default `7`) to keep overhead bounded.
   - Ramps `qat_alpha` smoothly once LR scale falls below `LATE_QAT_THRESHOLD` instead of toggling a dead branch.

2. **AWQ-lite export calibration**
   - Runs a tiny post-train calibration pass over a few training batches.
   - Captures linear-layer input activations.
   - For each int6 matrix, picks the best clip percentile from a small candidate set by minimizing **output error on sampled activations**, not plain weight MSE.

3. **LeakyReLU(0.5)^2 MLP**
   - Carries forward the latest record's low-risk activation improvement.

4. **Validation-friendly fallbacks**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA.
   - A small `ALLOW_CPU_SMOKE=1` path exists for startup validation without CUDA.
   - `USE_COMPILE` can be disabled for smoke checks.

## How to run / evaluate

A challenge-shaped run from this candidate directory should look like:

```bash
RUN_ID=awq_lite_late_qat \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
EVAL_STRIDE=64 \
LEAKY_RELU_SLOPE=0.5 \
QAT_LAYER_START=7 \
LATE_QAT_THRESHOLD=0.15 \
ACTIVATION_AWARE_QUANT=1 \
CALIBRATION_BATCHES=2 \
CALIBRATION_BATCH_TOKENS=65536 \
CALIBRATION_SAMPLES_PER_MODULE=256 \
USE_COMPILE=1 \
python train_gpt.py
```

If FlashAttention 3 is installed in the target environment, the script will use it automatically on CUDA. Otherwise it will run with SDPA, which is simpler but likely slower.

## Main expected risks / tradeoffs

- **Calibration cost**: the small activation-collection pass adds some post-train work. It is intentionally tiny, but it still needs to pay for itself.
- **Selective QAT overhead**: even selective late-layer fake quant adds some extra compute.
- **Calibration overfitting**: the AWQ-lite percentile choice uses a tiny train calibration subset; it may help some layers and do nothing for others.
- **Unverified on 8xH100**: this candidate is implementation-complete, but the current runner cannot execute a real torch smoke test because the repo's Python deps are absent locally.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603272344_awq-lite-late-qat
python -m compileall candidates/202603272344_awq-lite-late-qat/train_gpt.py
```

Outcome:

- **Passed**.

Attempted next step:

- A minimal CPU smoke test with a synthetic SentencePiece model and shard pair.

Why that smoke test was not completed here:

- The workflow runner does not currently have the repo runtime dependencies installed (`torch`, `numpy`, `sentencepiece` are all missing), and this environment does not provide networked package installation for filling that gap.
