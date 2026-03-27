# Shared-Tail Bigram Candidate

## Hypothesis

The current best non-TTT stack in this repo already extracts a lot from better quantization, EMA, XSA, partial RoPE, and a hashed bigram residual. The next promising move is to spend parameters more efficiently, not add more executed depth.

This candidate shares the **heavy tail transformer blocks** across the last four executed layers while keeping **per-step control tensors** untied. The saved body parameters are then partially reallocated into a larger default `BigramHash` table (`3072` buckets instead of `2048`). The hope is that this keeps the strong 11-step compute path while improving artifact efficiency and lexical capacity under the 16MB limit.

## Why this is promising for this repository

Repository evidence points in two directions:

- Later wins consistently stack **more capacity per byte**: 11 layers, 3x MLP, XSA, BigramHash, VE, GPTQ-lite, EMA.
- A pure “add more repeated steps” recurrence test was already a negative result under fixed wallclock, because it reduced total optimizer steps too much.

So this candidate deliberately does **not** add extra executed layers. Instead, it keeps the same executed depth and shares only the expensive tail weights. That follows the parameter-efficiency logic of ALBERT/Universal Transformer style sharing without paying the wallclock penalty of a deeper unrolled network.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Key influences:

- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` for the 11-layer XSA + EMA backbone.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` for partial RoPE + LN scaling.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` for the observation that a larger BigramHash can still help on top of a strong stack.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` as a warning that naive layer recurrence can lose badly if it increases per-step compute.

## External research that informed it

Primary sources:

- **ALBERT** — cross-layer parameter sharing for parameter-efficient transformers: https://arxiv.org/abs/1909.11942
- **Universal Transformer** — recurrent parameter reuse across depth: https://arxiv.org/abs/1807.03819

These papers motivate the core bet here: under a hard artifact budget, reusing deep transformer weights can be worthwhile if the model still gets multiple computation steps and enough per-step flexibility.

## What changed versus the chosen base implementation

Relative to the `2026-03-22` GPTQ-lite + EMA record script:

- Added **shared-tail execution mapping** with new knobs:
  - `SHARED_TAIL_N=4`
  - `SHARED_TAIL_UNIQUE=2`
- The default executed-layer mapping becomes:
  - `[0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 8]`
  - so the model still executes 11 layers, but only materializes 9 physical `Block`s.
- Moved the main control tensors to **per-step tensors** at the `GPT` level so repeated block uses are not completely identical:
  - attention scales
  - MLP scales
  - residual mix tensors
  - Q-gains
  - layerwise LN scale factors
- Increased the default `BIGRAM_VOCAB_SIZE` from `2048` to `3072`.
- Added a **CPU-safe attention fallback** using `torch.nn.functional.scaled_dot_product_attention` when FlashAttention 3 is unavailable. This is mainly to make import-level smoke tests possible in less specialized environments.
- Switched default `DATA_PATH` and `TOKENIZER_PATH` to resolve from the repository root, so running from inside this candidate directory works without rewriting paths.
- Threaded the configured `TRAIN_SEQ_LEN` into RoPE construction so the partial-RoPE extrapolation threshold follows the real training context instead of a hardcoded `1024`.
- Set `LATE_QAT_THRESHOLD=0.0` by default to isolate the shared-tail hypothesis rather than rely on the repo’s older compile-fragile late-QAT path.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202603272218_shared-tail-bigram
RUN_ID=shared_tail_bigram_seed1337 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script defaults already point back to the repo-level cached dataset/tokenizer:

- `data/datasets/fineweb10B_sp1024/`
- `data/tokenizers/fineweb_1024_bpe.model`

Useful candidate-specific knobs:

```bash
SHARED_TAIL_N=4
SHARED_TAIL_UNIQUE=2
BIGRAM_VOCAB_SIZE=3072
```

For an ablation back toward the original base shape:

```bash
SHARED_TAIL_N=0 BIGRAM_VOCAB_SIZE=2048
```

## Main expected risks / tradeoffs

- Sharing tail weights may reduce expressivity if the repeated tail blocks collapse toward similar behavior.
- The untied per-step controls may not be enough to recover the flexibility lost by sharing the heavy matrices.
- A larger `BigramHash` table may spend artifact budget on shallow lexical memory instead of deeper contextual parameters.
- This was not validated on the target 8xH100 environment here, so throughput and final artifact size remain the biggest uncertainty.

## Validation run in this workflow

Passed:

```bash
python -m compileall candidates/202603272218_shared-tail-bigram/train_gpt.py
```

Attempted but blocked locally:

```bash
python3 - <<'PY'
import torch
PY
```

Outcome:

- local syntax compilation passed
- a minimal CPU forward-pass smoke test was **not feasible in this environment** because the available local Python runtime does not have `torch` installed, even though `requirements.txt` lists it
- because of that, this candidate was validated statically here rather than with a live forward pass
