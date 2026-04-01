# RotGPTQ-lite: blockwise sign-Hadamard PTQ on the current best TTT stack

## Hypothesis

The repository's best runs already extract most of the easy gains from architecture and evaluation, so the next low-risk win is likely in **post-training quantization quality**. A small **blockwise sign-Hadamard rotation search** before GPTQ-lite int6 quantization should reduce row outliers in the banked MLP/attention weights, improve reconstruction MSE after dequantization, and translate into a slightly better post-roundtrip `val_bpb` without touching the 600s training budget.

## Why this is promising here

- The record history shows repeated gains from better export paths rather than wholesale architecture resets:
  - fp16 embeddings reduced the quantization gap
  - mixed int6/int8 export unlocked deeper 11-layer stacks
  - GPTQ-lite clip search bought another zero-training-cost gain
- The current top record already combines the strongest known local ingredients (11L, LeakyReLU^2, XSA, Partial RoPE, LN scale, VE, EMA/SWA, legal TTT, Parallel Muon), so a **surgical PTQ improvement** is a better fit than a new large subsystem.
- Local repo evidence also argues against the main alternative from external research: the non-record 1x5090 sweep reports **layer recurrence/shared-depth as a negative result** under fixed wallclock, which makes MobileLLM-LS-style sharing less attractive as the immediate next experiment here.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest end-to-end local stack and the most relevant base to improve
- **Quantization direction:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - showed that export-only GPTQ-lite clip search still mattered
- **Partial RoPE / LN scale stack:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms the current architecture family is already near a good local optimum

## External research that informed it

- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs**  
  https://arxiv.org/abs/2404.00456  
  Motivated trying orthogonal rotations to smooth quantization outliers before low-bit export.
- **SpinQuant: LLM quantization with learned rotations**  
  https://arxiv.org/abs/2405.16406  
  Reinforced that the choice of rotation matters, so this candidate searches a tiny bank of deterministic sign-Hadamard variants instead of using a single fixed transform.
- **MobileLLM / MobileLLM-LS**  
  https://arxiv.org/abs/2402.14905  
  This was the strongest alternative considered from external research, but I did not choose it because the local repo already has a negative recurrence/shared-depth signal under tight wallclock constraints.
- **BitNet b1.58**  
  https://arxiv.org/abs/2402.17764  
  Relevant to the 16MB budget, but much more invasive than a PTQ-only tweak.

## What changed vs the chosen base

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate only changes the int6 export path:

1. Adds three new knobs:
   - `ROTPTQ_ENABLED=1`
   - `ROTPTQ_CANDIDATES=4`
   - `ROTPTQ_BLOCK_SIZE=512`
2. For each eligible 2D int6 weight matrix, tries:
   - identity GPTQ-lite quantization
   - several deterministic **sign-Hadamard block rotations** on the input dimension
3. Scores each candidate by **reconstruction MSE after inverse rotation**, then stores:
   - quantized weights
   - scales
   - a tiny per-matrix rotation seed/block metadata entry
4. Inverts the chosen rotation during dequantization before roundtrip evaluation.

Everything else is intentionally left aligned with the proven top record stack.

## How to run / evaluate

From this directory. After populating the standard repo `data/` layout from the root README, `train_gpt.py` resolves `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so the usual checkout works without extra path overrides:

```bash
RUN_ID=rotgptq_lite \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
ROTPTQ_ENABLED=1 ROTPTQ_CANDIDATES=4 ROTPTQ_BLOCK_SIZE=512 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script remains self-contained inside this candidate directory.

## Validation

### Commands run

From the repository root:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604011741_rotgptq-lite/train_gpt.py
```

### Outcomes

- `compileall` succeeded for the root training scripts, `data/`, and this candidate trainer.
- A minimal CPU import/forward smoke test was **not feasible in this runner** because the local Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), so only syntax-level validation was possible here.

## Main risks / tradeoffs

- The rotation search increases **post-training export time**, though not training time.
- It only activates for matrices whose input dimension is divisible by a usable power-of-two block size.
- The search target is **weight-space MSE**, not `val_bpb` directly, so the best local reconstruction candidate may not always give the best final metric.
- The base stack already spends a lot of evaluation time on legal TTT, so any extra export overhead needs monitoring on real hardware.
