# SpQR-lite Outlier Sidecar on the Legal-TTT / Parallel-Muon Stack

## Hypothesis

The current best record line is already strong on training-time quality (`LeakyReLU^2`, EMA/SWA, XSA, VE, partial RoPE) and eval-time quality (legal score-first TTT), so the next cheap win is likely in the **artifact path**, not the training loop. This candidate keeps the current best 11-layer stack intact and changes only the int6 export path: after the existing GPTQ-lite per-row clip search, it stores a tiny **fp16 residual sidecar** for the hardest-to-quantize weights in the deepest layers.

Concretely, the candidate keeps the base int6+lzma artifact format, but for selected 2D block weights it stores:

- one `int16` column index per row,
- one `float16` residual value per row,
- and adds that residual back during dequantization.

The intent is to recover the last few high-error outliers that rowwise int6 still misses, without paying for a broad mixed-precision upgrade.

## Why this is promising for this repository

Recent record progress in this repo repeatedly came from **compression-aware tweaks with low training overhead**:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved over the prior 11-layer stack mostly by improving post-training quantization (`GPTQ-lite` clip search) and averaging.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` pushed lower mostly via a one-line activation change plus legal score-first TTT, but it still exports through the same basic int6 path.
- The deeper winning line already specializes the last few layers with XSA, partial RoPE, LN scaling, and VE, which suggests those layers are a good place to spend a tiny extra artifact budget.

So the bet here is: if the training stack is already near-saturated, the most repo-aligned next move is to make **deep-layer int6 export slightly less lossy** rather than add another high-risk training feature.

## Prior records / candidates that influenced this

There were **no prior `candidates/` directories** in the repository when this candidate was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - base implementation copied for this candidate,
  - current best repo score,
  - establishes the legal-TTT / parameter-banked / LeakyReLU^2 stack.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - showed that small improvements in the int6 export path are worth chasing,
  - specifically motivates building on top of GPTQ-lite rather than replacing it.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - supports the repo trend of specializing the deepest layers,
  - which is why this candidate spends its sidecar budget on late layers first.

## External research that informed it

This candidate is mainly informed by primary-source quantization work that isolates the hardest weights instead of uniformly raising precision everywhere:

- **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression** (arXiv:2306.03078)
  - motivates isolating quantization outliers and storing them in higher precision.

- **PB-LLM: Partially Binarized Large Language Models** (arXiv:2310.00034)
  - motivates the idea that a small set of salient weights can carry disproportionate compression error, so preserving them separately can recover quality.

- **Scaling Law for Quantization-Aware Training** (arXiv:2505.14302)
  - emphasizes that quantization error remains a real bottleneck, and that mixed-precision treatment of the hardest components is often the right lever.

I also reviewed more aggressive basis-change PTQ ideas such as QuaRot / SpinQuant during research, but did **not** implement them here because they require more invasive basis-handling infrastructure than felt appropriate for a surgical candidate.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added three export knobs:
   - `OUTLIER_TOPK` (default `1`)
   - `OUTLIER_LATE_LAYERS` (default `4`)
   - `OUTLIER_MLP_ONLY` (default `0`)

2. Added `_layer_index_from_name(...)` so the quantizer can target the deepest blocks only.

3. Added `quantize_int6_with_outlier_sidecar(...)`:
   - runs the existing GPTQ-lite rowwise int6 quantizer,
   - reconstructs the tensor,
   - finds the largest residuals per row,
   - stores their indices and fp16 residual values.

4. Extended `mixed_quantize_int6(...)` to optionally attach sparse outlier sidecars to late-layer int6 tensors and report sidecar stats.

5. Extended `dequantize_mixed_int6(...)` to scatter the stored residuals back into the dequantized tensor before evaluation.

Nothing else in the training loop, optimizer logic, model architecture, or legal-TTT path was changed.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202603260924_spqr-lite-sidecar
OUTLIER_TOPK=1 OUTLIER_LATE_LAYERS=4 OUTLIER_MLP_ONLY=0 \
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
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script logs a new `outlier_sidecar:` line during export so you can see how many tensors/entries received residual corrections and how many uncompressed payload bytes they consumed before `lzma`.

## Expected risks / tradeoffs

- The sidecar spends extra bytes on artifact quality, so it could help roundtrip/sliding BPB while slightly worsening final compressed size.
- If late-layer residuals are not actually the dominant error source, the improvement may be negligible.
- If the sidecar budget is too aggressive (`OUTLIER_TOPK > 1` or too many layers), it may erase the size advantage.
- This is still a heuristic approximation of SpQR/PB-LLM-style salient-weight preservation, not a full Hessian-aware sparse codec.

## Validation

Commands run during candidate creation:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603260924_spqr-lite-sidecar/train_gpt.py
```

Outcome:

- **Passed**. The repository baseline scripts, `data/` helpers, and the new candidate `train_gpt.py` all compiled successfully.

Attempted extra low-cost smoke check:

- Planned a CPU-only helper smoke test that AST-extracts the new quantization helpers and runs them on random tensors, without touching CUDA / FlashAttention.
- Could not complete it in this workflow environment because `/usr/bin/python` did not have `torch` installed, even though `requirements.txt` lists it.
- A true CPU launch smoke test of `train_gpt.py` was also not feasible here because the script hard-requires CUDA / FlashAttention, matching the existing record implementations.

So the candidate was syntax-validated, but not fully runtime-smoke-tested in this runner.
