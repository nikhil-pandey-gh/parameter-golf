# Candidate: compile-safe bank-aware late QAT

## Hypothesis

The current SOTA stack already has strong architecture, optimizer, and evaluation choices, but its late QAT path does not materially regularize the largest banked weights (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`). A late-phase, bank-aware fake-int6 path with trainable per-row clip multipliers should shrink the train/export quantization gap where most artifact bytes live, improving post-quant `val_bpb` without changing the overall 11-layer budget.

## Why this is promising here

Repository evidence points in the same direction:

- the strongest runs converge on **11 layers + 3x MLP + int6 block-weight export + sliding eval + EMA**;
- the 2026-03-22 record showed that **better post-training quantization alone** was worth about `-0.0013` bpb via GPTQ-lite clip search and tuned late-QAT timing;
- the 2026-03-21 record explicitly documented that its old late-QAT path was effectively dead because `torch.compile` constant-folded the QAT branch;
- the 2026-03-23 SOTA moved the core projections into **parameter banks**, which improved systems performance but also means the old `CastedLinear` late-QAT branch no longer touches the main q/k/v/o and MLP weights.

So the gap is specific and actionable: make late QAT both **compile-safe** and **bank-aware**, then carry the learned row multipliers into export-time GPTQ-lite quantization.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - chosen base implementation;
  - contributes the current best training stack, parameter banking, LeakyReLU(0.5)^2, legal TTT, and lzma-compressed int6 export.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - motivates keeping GPTQ-lite row clip search at export time.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - motivates the one-time recompile at the QAT handoff so the quantized path cannot be dead-code-eliminated.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - reinforces that the best wins come from keeping the 11L compressed stack intact rather than swapping to more speculative architectures.

No prior `candidates/` directory existed when this candidate was created.

## External research that informed it

- **LSQ** — Esser et al., *Learned Step Size Quantization* (arXiv:1902.08153): learn quantizer step sizes jointly with model weights using straight-through gradients.
- **GPTQ** — Frantar et al., *GPTQ* (arXiv:2210.17323): strong one-shot weight quantization for GPT-style models.
- **SmoothQuant** — Xiao et al. (arXiv:2211.10438): quantization difficulty is often concentrated in a subset of channels and can be improved by channelwise rescaling.
- **AWQ** — Lin et al., *Activation-aware Weight Quantization* (arXiv:2306.00978): protecting a small set of salient channels with scaling substantially reduces weight-only quantization error.
- **QDrop** — Wei et al. (arXiv:2203.05740): low-bit quantization benefits from training procedures that make the quantized solution flatter and less brittle.

This candidate is not a full reproduction of any of those methods. The concrete adaptation here is: **LSQ-style trainable row multipliers on the banked weights, folded into the repository's existing GPTQ-lite export path**.

## What changed vs. the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes:

1. Added **repo-root-relative default paths** so the script can be launched directly from this candidate directory.
2. Added trainable **per-row log clip multipliers** for each banked weight family:
   - `qo_bank`
   - `kv_bank`
   - `mlp_up_bank`
   - `mlp_down_bank`
3. Added a **bank-aware fake-int6 path** used only in the late phase of training.
4. Kept the existing late QAT trigger, but now do a **one-time `torch.compile` rebuild** when late QAT turns on, so the quantized branch is actually traced and used.
5. Reused the learned row multipliers during **export-time GPTQ-lite row clip search**.
6. Excluded the QAT-only clip parameters from the final exported artifact.

Everything else stays intentionally close to the SOTA base: parameter banking, Parallel Muon, LeakyReLU(0.5)^2, XSA, partial RoPE, EMA/SWA, sliding eval, and legal TTT.

## How to run

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
BANK_QAT_ENABLED=1 BANK_QAT_RECOMPILE=1 \
BANK_QAT_MULT_MIN=0.50 BANK_QAT_MULT_MAX=1.50 BANK_QAT_GRAD_SCALE=1.0 \
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

By default, `DATA_PATH` and `TOKENIZER_PATH` resolve back to the repository root, so no extra path overrides are required when launching from this directory.

## Validation

- `python -m compileall train_gpt.py train_gpt_mlx.py data` from the repository root — **passed**
- `python -m compileall candidates/202604031447_bank-aware-late-qat/train_gpt.py` — **passed**
- `python - <<'PY' ... import torch ... PY` smoke-feasibility probe — **failed immediately with `ModuleNotFoundError: No module named 'torch'`**
- Minimal CPU-only smoke start — **not feasible in this environment without changing infrastructure**, because the local environment is missing the runtime dependency stack (`torch` already absent here) and the script imports the FlashAttention 3 CUDA interface at module import time.

## Main risks and tradeoffs

- The one-time recompile at late-QAT handoff costs wall-clock time; if the compile hit is too large, the extra quantization robustness may not repay the lost steps.
- The late fake-quant path uses row-max-based LSQ-style adaptation, while export still uses GPTQ-lite percentile search; that mismatch may help or may partly cancel out.
- Learned clip multipliers can over-protect a few rows and slightly hurt pre-quant quality if the late phase is too long or the multipliers drift to the clamp edges.
- The candidate keeps the heavy legal TTT evaluation stack from the base, so training wins still need to survive the full post-quant + post-TTT pipeline.
