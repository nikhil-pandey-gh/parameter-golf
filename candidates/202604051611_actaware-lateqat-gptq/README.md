# Activation-Aware Late QAT + GPTQ-lite

## Hypothesis

The strongest current stack in this repo is already close to the training-time frontier, so the next cheap win is to reduce **weight-only export error** rather than changing the backbone again. This candidate tests whether:

1. **real late QAT on the banked attention/MLP weights**, and
2. **activation-aware clip selection during int6 export**

can shrink the post-training quantization gap enough to improve final validation BPB under the same 16MB artifact cap.

## Why this is promising here

- The repo’s best non-TTT record already showed that **GPTQ-lite clip search + EMA + longer warmdown** were a real gain.
- The overall best record then added **LeakyReLU(0.5)^2 + legal score-first TTT + Parallel Muon**, but it still exports through a weight-only low-bit path.
- Prior records repeatedly show that leaderboard movement now comes from small improvements to **quantization friendliness**, **evaluation**, and **cheap zero-parameter architectural tweaks**, not from large new subsystems.
- One earlier record explicitly documented a **late-QAT path that became a no-op under `torch.compile`**. This candidate tackles that class of problem directly by fake-quantizing the actual banked weights used by the main model body.

No prior `candidates/` directory existed when this candidate was prepared.

## Prior records that informed this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best overall mean score in-repo,
  - provides the base stack: 11L, parameter banks, Parallel Muon, Partial RoPE, XSA, EMA/SWA, LeakyReLU(0.5)^2, legal TTT.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest non-TTT base,
  - showed GPTQ-lite percentile search and EMA mattered,
  - motivated keeping the export path as the main optimization target.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - important negative lesson: the earlier late-QAT path did not actually affect the intended large weights.

## External research that informed it

- **GPTQ** — https://arxiv.org/abs/2210.17323  
  Motivated using output-aware / curvature-aware thinking for post-training low-bit export rather than plain weight MSE.
- **AWQ (Activation-aware Weight Quantization)** — https://arxiv.org/abs/2306.00978  
  Motivated using calibration activations to identify which input channels matter most during weight quantization.
- **SmoothQuant** — https://arxiv.org/abs/2211.10438  
  Reinforced the value of offline activation statistics for making low-bit quantization more faithful.
- **LSQ** — https://arxiv.org/abs/1902.08153  
  Motivated keeping a real fake-quant training path late in optimization instead of relying only on post-training export.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

### 1. Real late QAT for the banked model weights

The base record’s large attention/MLP weights live in parameter banks and are applied through `F.linear(...)`, so `CastedLinear` fake-quant alone does not cover the main model body. This candidate:

- fake-quantizes **banked attention and MLP weights** during forward,
- enables that path only in the late phase via `LATE_QAT_THRESHOLD`,
- recompiles once when late QAT turns on so the compiled graph actually includes the new path.

### 2. Activation-aware GPTQ-lite clip selection

Before int6 export, the script now runs a short calibration pass on training-distribution batches and records per-layer **input second moments** for:

- Q / K / V projections,
- attention output projections,
- MLP up projections,
- MLP down projections.

The exporter then uses those activation statistics to weight reconstruction error during percentile search, favoring clip choices that reduce **expected output distortion**, not just raw weight-space MSE.

### 3. Better alignment between training and export

- `CastedLinear` fake-quant now uses the same int6 helper family as the banked late-QAT path.
- Late QAT uses a **cheap row-max fake-quant pass** so it is practical inside the 10-minute budget, while export keeps the heavier activation-aware percentile search.
- The banked late-QAT path and the export path still target the same low-bit regime.

### 4. FlashAttention fallback for lightweight local smoke checks

If `flash_attn_interface` is unavailable, the module can still run **import-time / toy-shape forwards** through PyTorch SDPA. This does **not** make `main()` CPU-runnable: the training entrypoint still requires CUDA. The fallback is only for small validation/debug contexts.

## How to run / evaluate

From this candidate directory:

```bash
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
QAT_CLIP_PERCENTILE=1.0 \
QUANT_CALIBRATION_STEPS=4 QUANT_CALIBRATION_BATCH_TOKENS=131072 QUANT_CALIBRATION_SEQ_LEN=512 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The key new knobs are:

- `QAT_CLIP_PERCENTILE`
- `QUANT_CALIBRATION_STEPS`
- `QUANT_CALIBRATION_BATCH_TOKENS`
- `QUANT_CALIBRATION_SEQ_LEN`

## Expected risks / tradeoffs

- The extra late-stage compile may slightly perturb the wallclock profile.
- Activation-aware calibration may overfit if the calibration slice is too small or too narrow.
- The new gain may be modest if legal TTT is still the dominant contributor to the final score.
- Weight-space improvements that help roundtrip BPB do not always translate one-for-one to sliding-window or post-TTT BPB.

## Files added

- `train_gpt.py` — self-contained candidate script
- `README.md` — hypothesis, provenance, usage, and validation notes

## Validation

### Commands run

From the repository root:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604051611_actaware-lateqat-gptq/train_gpt.py
```

```bash
python - <<'PY'
import torch
PY
```

### Outcomes

- `compileall` completed successfully for the root baseline scripts, `data/`, and this candidate script.
- A CPU-only smoke check of the candidate runtime path was **not feasible in this workflow environment** because `torch` is not installed here, even though it is declared in `requirements.txt`.
- Because of that environment limitation, the candidate’s runtime path was not exercised end-to-end in this run; the candidate is positioned as a research iteration to be executed in the repository’s normal PyTorch/CUDA environment.
