# Candidate: QuaRot-lite mixed int6 export on the 11L EMA/XSA stack

## Hypothesis

The current record family appears increasingly limited by **quantization gap**, not by a lack of architectural tricks. This candidate tests whether a **deterministic block-Hadamard rotation** applied before low-bit export can reduce outlier-driven reconstruction error enough to improve roundtrip quality at essentially zero training cost.

Concretely: for each eligible 2D weight matrix, the exporter now tries both:

1. the existing direct mixed int6/int8 quantization path, and
2. a rotated path that applies a normalized block-Hadamard transform on the input dimension before quantization, then inverts that transform after dequantization.

It keeps whichever version has lower reconstruction MSE for that tensor family.

## Why this is promising for this repository

The repository history points to the export path as one of the highest-leverage remaining surfaces:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` showed that the recorded gain came from Partial RoPE + LN scale, while the advertised late QAT path was actually dead code under `torch.compile`.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` got a real win from a **better post-training quantizer** plus EMA and warmdown tuning.
- Earlier runs repeatedly called out embeddings and low-bit export quality as a bottleneck, especially `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` and `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md`.

This candidate keeps the strong 11-layer static stack intact and changes only the **basis in which matrices are quantized**, which is a differentiated lever from the existing repo ideas.

## Prior repo work that influenced this candidate

- **Chosen base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
  - strong static 11L / XSA4 / EMA / GPTQ-lite baseline
  - already isolates the quantization path cleanly
- **Nearby stronger record:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - useful for trend reading, but it adds TTT and parameter banking; this candidate intentionally avoids that extra integration cost so the new signal comes from export quality alone
- **Other influential records:**
  - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`
  - `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/README.md`
  - `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`

There was **no pre-existing `candidates/` directory** in the repository when this candidate was created, so this is the first candidate iteration series in that location.

## External research that informed it

The key research thread is that **orthogonal mixing / flattening before quantization** materially improves low-bit robustness:

- **QuIP#** (2024): https://arxiv.org/abs/2402.04396
- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** (2024): https://arxiv.org/abs/2404.00456
- **SpinQuant: LLM quantization with learned rotations** (2024): https://arxiv.org/abs/2405.16406
- **FlatQuant: Flatness Matters for LLM Quantization** (2024): https://arxiv.org/abs/2410.09426
- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (2023): https://arxiv.org/abs/2306.00978
- **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression** (2023): https://arxiv.org/abs/2306.03078
- **BitStack: Any-Size Compression of Large Language Models in Variable Memory Environments** (2024): https://arxiv.org/abs/2410.23918

I also considered more structural ideas such as **ALBERT** cross-layer sharing (https://arxiv.org/abs/1909.11942) and **Universal Transformers** recurrence/shared depth (https://arxiv.org/abs/1807.03819), but those would require materially larger architectural churn than a quantizer-focused candidate.

## What changed versus the chosen base

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. adds a **CPU-safe and non-Hopper-safe attention fallback** using `scaled_dot_product_attention`, while keeping FlashAttention-3 only on Hopper-class GPUs,
2. adds new export knobs:
   - `ROTATION_BLOCK_SIZE` (default `128`)
   - `ROTATION_ENABLE_INT6` (default `1`)
   - `ROTATION_ENABLE_INT8` (default `1`)
3. adds **normalized block-Hadamard rotation helpers** for eligible 2D matrices whose input dimension is divisible by the chosen block size,
4. extends both the mixed int6 path and the int8 fallback path to try **direct vs rotated quantization** and keep the lower-MSE reconstruction,
5. stores enough metadata to invert the chosen rotation during dequantization and roundtrip evaluation,
6. logs the number of tensors that actually used the rotated basis.

No files outside this candidate directory were changed.

## How to run or evaluate it

Run it like the 2026-03-22 static stack, with the rotation flags enabled:

```bash
cd candidates/202604101921_quarot-lite-int6

ROTATION_BLOCK_SIZE=128 \
ROTATION_ENABLE_INT6=1 \
ROTATION_ENABLE_INT8=1 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
LATE_QAT_THRESHOLD=0.15 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

What to watch in the logs:

- `rotation_block_size`, `rotation_int6`, `rotation_int8`
- `rotation_quantized_tensors`
- final `final_int6_roundtrip` and `final_int6_sliding_window_s64` metrics

## Main expected risks and tradeoffs

- The transform is **fixed**, not learned; it is closer to QuaRot than SpinQuant.
- Export becomes slower because each eligible tensor now evaluates multiple candidates.
- The benefit may be concentrated in a subset of large matrices, so gains may be modest if the base stack is already well-conditioned.
- Embeddings remain a known pain point; this candidate lets the int8 path use rotations too, but it does **not** force embeddings down to int6.
- This does not solve the larger structural ideas (shared depth, lighter TTT, etc.); it is intentionally a narrow export-path bet.

## Validation

### Commands run

```bash
python -m compileall candidates/202604101921_quarot-lite-int6/train_gpt.py
```

```bash
python - <<'PY'
import importlib.util
...
PY
```

### Outcomes

- `python -m compileall .../train_gpt.py` **succeeded**
- a minimal CPU import / quantization smoke test was **not feasible in this container** because the local Python environment does not currently have `torch` installed (`ModuleNotFoundError: No module named 'torch'`)

So this candidate is syntax-checked locally, but still needs its first full runtime smoke on an environment with the repository's normal PyTorch stack available.
