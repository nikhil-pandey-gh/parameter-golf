# Cubic MLP Equalization on the 2026-03-23 SOTA Stack

## Hypothesis

The current best stack already wins by combining a strong 11-layer architecture with better export-time compression. This candidate pushes that same seam one step further: before GPTQ-lite int6 quantization, exactly rebalance each MLP hidden channel so the `fc` row and `proj` column have more similar ranges.

Because this stack uses `LeakyReLU(0.5)^2`, the MLP is positively homogeneous of degree 2. For any positive hidden-channel scale `s`,

```python
leaky_relu(s * z, negative_slope=0.5).square() == s**2 * leaky_relu(z, negative_slope=0.5).square()
```

So we can preserve the pre-quantization function exactly by applying

```python
fc[h, :]   <- s[h] * fc[h, :]
proj[:, h] <- proj[:, h] / s[h]**2
```

This candidate chooses `s[h]` from per-channel max magnitudes with a cubic balancing rule, then runs the existing GPTQ-lite clip search on the rebalanced weights.

## Why this is promising for this repository

Recent records improved mainly by:

1. moving to the 11L / 512d / 3x-MLP / XSA / Partial-RoPE line,
2. smoothing weights before export with EMA / tight SWA,
3. improving low-bit export itself with GPTQ-lite and lzma.

The strongest repository evidence comes from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

This candidate targets the same quantization bottleneck without changing the training loop or adding new infrastructure. It is also complementary to the repo-wide trend toward outlier-aware compression, but cheaper to implement than a full exception-buffer OWQ/SpQR variant.

## Prior repository influences

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Compression baseline:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- **Architectural lineage:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- **Repository review note:** there were **no pre-existing `candidates/` directories** to extend or avoid duplicating.

## External research that informed it

- **Data-Free Quantization through Weight Equalization and Bias Correction** (Nagel et al., ICCV 2019) — motivates exact range equalization before PTQ.  
  https://arxiv.org/abs/1906.04721
- **SmoothQuant** (Xiao et al., ICML 2023) — shows that equivalent offline rescaling can move quantization difficulty to friendlier places.  
  https://arxiv.org/abs/2211.10438
- **AWQ** (Lin et al., MLSys 2024) — reinforces that activation-/channel-aware weight rescaling is a strong low-bit direction for LMs.  
  https://arxiv.org/abs/2306.00978

The broader external-research pass for this workflow also pointed to outlier-aware quantization as the strongest next family for this repo. This candidate is the smallest exact variant that fits the current codebase cleanly.

## What changed vs the chosen base implementation

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`:

1. Added an **export-time cubic MLP equalization pass** on the unbanked state dict before GPTQ-lite int6 quantization.
2. Logged equalization statistics per pass so runs show the scale range actually used.
3. Added a **FlashAttention/SDPA fallback helper** so forward passes still work when FlashAttention is unavailable.
4. Added a **`SMOKE_TEST_ONLY=1`** path that runs a tiny CPU forward pass plus quantize/dequantize/rebank roundtrip without requiring CUDA or dataset files.

The main training recipe, architecture, EMA/SWA behavior, legal TTT path, and quantization codec remain otherwise unchanged.

## How to run or evaluate it

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
MLP_EQ_ENABLED=1 MLP_EQ_MAX_SCALE=4.0 MLP_EQ_PASSES=1 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

EMA stays on by default in this base script, so there is no separate `EMA_ENABLED` toggle to pass here.

Quick CPU smoke path:

```bash
SMOKE_TEST_ONLY=1 SMOKE_TEST_SEQ_LEN=16 SMOKE_TEST_BATCH_SIZE=1 python train_gpt.py
```

## Main expected risks and tradeoffs

- The transform is **function-preserving before quantization**, but it may still hurt if it makes GPTQ-lite clip search or lzma compression less favorable on some matrices.
- It only touches the **MLP** path, not attention/outlier rows elsewhere, so the upside may be smaller than a fuller OWQ/AWQ-style mixed-bit exception scheme.
- The FlashAttention fallback is for portability and smoke testing; if the runtime falls back to SDPA on the real training path, throughput may drop.

## Validation run for this candidate

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604061426_cubic-mlp-equalization/train_gpt.py` — passed
- `SMOKE_TEST_ONLY=1 SMOKE_TEST_SEQ_LEN=16 SMOKE_TEST_BATCH_SIZE=1 python train_gpt.py` — passed in a temporary venv with repo dependencies installed
  - Output: `smoke_test_ok loss:6.951154 roundtrip_loss:6.959777 seq_len:16 batch_size:1`

The smoke test confirms that the new equalization path, quantize/dequantize roundtrip, and re-banked model load all complete without a GPU.
