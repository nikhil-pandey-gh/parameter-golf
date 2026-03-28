# Control-Local Legal TTT

## Hypothesis

The current best archived stack already shows that legal score-first test-time training (TTT) is worth roughly `-0.0025` BPB, but it pays for that gain by adapting the full model after each scored chunk. This candidate tests a more targeted version: keep the strong 11-layer LeakyReLU^2 + XSA + partial-RoPE + EMA/GPTQ-lite recipe, but during legal TTT adapt only the model's small **control tensors** and **local context modules** (`SmearGate`, `BigramHash`, and value-embedding parameters) instead of the full banked weight stack.

The bet is that, in this repository, a large fraction of the useful TTT signal lives in fast-changing routing/gating/context knobs rather than in the quantized core matrices. If that is true, sparse control-local adaptation should preserve much of the TTT gain while lowering overfitting risk and making evaluation-time tuning easier.

## Why this is promising for this repository

Three trends from the archive point in the same direction:

- **Legal TTT is real signal, not noise.** The current best record `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` reports a consistent `-0.0023` to `-0.0027` BPB post-TTT gain across three seeds.
- **Cheap local context modules keep helping.** `SmearGate`, `BigramHash`, and later shared value embeddings recur in the strongest pre-TTT stacks, which suggests these modules are high-leverage places to adapt quickly.
- **Quantization robustness matters.** The archive's best models increasingly protect a small set of control tensors from aggressive low-bit packing while quantizing the large banks. Updating the already-float control/local path is safer than perturbing the quantized backbone during evaluation.

This makes sparse TTT a good fit for the challenge's combination of low-bit artifacts, strong eval-time methods, and a codebase that already exposes small trainable control pathways.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Chosen base implementation. It is the strongest archived stack and demonstrates that legal score-first TTT is already worth using.
- `records/track_10min_16mb/2026-03-17_LoRA_TTT/`
  - Useful negative/partial precedent: doc-wise evaluation mattered, but the specific adapter choice was not obviously optimal.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - Motivates treating local context modules as first-class adaptation targets.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Reinforces that quantization-aware choices around small/control tensors matter in this regime.

There were no pre-existing `candidates/` directories in this repository when this candidate was created.

## External research that informed it

- **Tent** — *Test-Time Adaptation by Entropy Minimization* ([arXiv:2006.10726](https://arxiv.org/abs/2006.10726))
  - Key takeaway: useful test-time adaptation can come from updating only a small affine/statistics subset rather than the whole model.
- **BitFit** — *Simple Parameter-efficient Fine-tuning for Transformer-based Masked Language-models* ([arXiv:2106.10199](https://arxiv.org/abs/2106.10199))
  - Key takeaway: surprisingly small parameter subsets can capture a large share of adaptation benefits.
- **SmoothQuant** ([arXiv:2211.10438](https://arxiv.org/abs/2211.10438)) and **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456))
  - Key takeaway: low-bit performance is especially sensitive to outliers and which tensors are allowed to stay in a friendlier representation. That argues for leaving the large quantized banks alone during TTT and focusing updates on the already low-bit-safe control/local path.

I also considered Primer-style local convolutions ([arXiv:2109.08668](https://arxiv.org/abs/2109.08668)), but chose sparse TTT for this round because it reuses the strongest archived implementation with less code and lower training-time risk.

## What changed versus the base implementation

Base file: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes two surgical changes on top of that script:

1. **New sparse TTT selection modes**
   - Added `TTT_MODE` with the following options:
     - `all` — original full-model TTT behavior
     - `control` — only train control tensors already tracked by the quantization logic
     - `control_local` — control tensors plus local `BigramHash` / value-embedding module weights
     - `custom` — user-provided substring patterns via `TTT_TRAINABLE_PATTERNS`
   - Added `TTT_EXCLUDE_PATTERNS` and `TTT_LOG_PARAMS` for ablation/debugging.
   - The candidate's intended setting is `TTT_MODE=control_local`.
   - Sparse modes intentionally require `TTT_FREEZE_BLOCKS=0`, because top-level local modules (`smear`, `bigram`, shared VE) affect the whole stack.

2. **Candidate-directory usability fix**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root derived from `__file__`, so the script can be run from inside this candidate directory without rewriting paths.

## How to run / evaluate

Run from this directory:

```bash
cd candidates/202603281351_control-local-ttt

RUN_ID=control_local_ttt \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_MODE=control_local TTT_LR=0.002 TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This inherits the base script's always-on EMA behavior (hard-coded decay `0.997`); this candidate does not add a new EMA toggle.

Suggested ablations:

- Compare against the original behavior with `TTT_MODE=all`.
- To isolate just the already-float control tensors, use `TTT_MODE=control`.
- If value embeddings overfit, try `TTT_EXCLUDE_PATTERNS=ve_shared,ve_layer_scales`.

## Main expected risks / tradeoffs

- **Risk: under-adaptation.** Full-model TTT may still be necessary for the last ~0.001-0.002 BPB.
- **Risk: different optimizer sweet spot.** Sparse parameters often want different LR / epoch settings than dense-model TTT, so `TTT_LR` and `TTT_EPOCHS` likely need a dedicated sweep.
- **Risk: local overfitting.** `BigramHash` and value embeddings are intentionally local/contextual, which is good for rapid adaptation but may overfit chunk-specific statistics if pushed too hard.
- **Tradeoff: cleaner evaluation experiments.** In exchange, this mode should make it easier to study which parts of the model actually drive TTT gains, because the update set is explicit and logged.

## Validation

Commands run in this workflow from the **repository root**:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603281351_control-local-ttt/train_gpt.py
```

Outcome:

- Both compileall commands succeeded.
- If you are already inside `candidates/202603281351_control-local-ttt/`, the directory-local syntax check is simply `python -m compileall train_gpt.py`.
- I attempted a lightweight import smoke test for the candidate script, but the workflow Python environment does not currently have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- Because this candidate inherits the CUDA + FlashAttention runtime path from the chosen base implementation, a real CPU-only runtime smoke test was **not feasible** in this session.
