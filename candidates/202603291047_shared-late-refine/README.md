# Shared Late Refine

## Hypothesis

Share only the late refinement stack instead of the whole network: keep all 11 forward passes, but reuse the heavy attention+MLP weights in the last 4 XSA layers across 2 learned cores while keeping per-layer norms, residual scales, skip connections, and VE scales unique.

The bet is that the deepest layers are already acting like an iterative refinement stack in the best records. ALBERT-style sharing there should improve parameter efficiency without paying the step-time penalty that hurt full recurrence experiments, and the recovered artifact budget can be reinvested into lexical auxiliaries that already looked promising in this repo.

## Why this is promising for this repository

The strongest non-TTT stack in this repo already concentrates most of its sophistication in the deepest layers: XSA on the last 4 layers, Partial RoPE, LN scale, EMA, GPTQ-lite, BigramHash, and VE. That makes the late stack the most natural place to try controlled sharing.

This candidate specifically avoids two dead ends that are already documented here:

- full-depth recurrence / layer looping was net negative under the 10-minute budget, because it cut training steps too aggressively,
- SwiGLU improved per-step quality but was too slow on the 8-GPU budget.

By sharing only existing late layers, this candidate keeps the same depth and roughly the same compute graph shape while reducing stored parameters.

## Prior records and experiments that informed it

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Most relevant repo signals:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - best non-TTT base with 11L + XSA4 + Partial RoPE + LN Scale + VE + GPTQ-lite + EMA.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - showed that increasing `BigramHash` capacity helped the best stack further.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - confirmed that Partial RoPE + LN scale are durable wins in the 11-layer family.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - documented that naive layer recurrence was a bad trade when it changed step throughput, which is why this candidate shares only the already-present late stack instead of adding more unrolled depth.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
  - documented that SwiGLU and full-depth recurrence were not good fits for this budget, pushing this candidate toward a lower-overhead parameter-efficiency change.

## External research that motivated it

- **ALBERT** (`arXiv:1909.11942`) showed that cross-layer parameter sharing can preserve strong performance while dramatically reducing parameter count.
- **Universal Transformer** (`arXiv:1807.03819`) argued that repeated transformation steps can act as iterative refinement rather than requiring every depth step to have fully unique weights.

This candidate adapts those ideas conservatively: it does **not** share the full network, and it does **not** increase depth. It only shares the late refinement stack that already appears most iteration-like in the current record family.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. Added `BlockCore`, which owns the heavy attention and MLP weights.
2. Kept per-layer `Block` wrappers for norms, residual mixing, scales, and optional DTG gates.
3. Added `SHARED_LATE_LAYERS` and `SHARED_LATE_CORES`.
4. Defaulted to sharing the last 4 layers across 2 cores.
   - With `NUM_LAYERS=11`, the default mapping is effectively:
   - unique early cores for layers `0..6`
   - shared late cores for layers `7..10` with mapping `[7, 8, 7, 8]`
5. Increased `BIGRAM_VOCAB_SIZE` from `2048` to `3072` by default.
6. Expanded VE coverage from layers `9,10` to `8,9,10` and increased `VE_DIM` from `128` to `160`.
7. Serialized the sharing config into exported checkpoints so non-default sharing overrides round-trip correctly.
8. Left the rest of the successful stack intact:
   - 11 layers / 512 dim / 8 heads / 4 KV heads
   - MLP 3x
   - Partial RoPE (16 dims)
   - LN scale
   - XSA on the last 4 layers
   - EMA + SWA warmdown behavior
   - GPTQ-lite-style int6 export path
   - sliding-window evaluation

## How to run or evaluate it

From this candidate directory:

```bash
RUN_ID=shared_late_refine \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The built-in final evaluation path still reports:

- diagnostic post-EMA validation,
- int6 roundtrip validation,
- sliding-window validation at `EVAL_STRIDE=64` by default.

Useful override knobs for this candidate:

```bash
SHARED_LATE_LAYERS=4
SHARED_LATE_CORES=2
BIGRAM_VOCAB_SIZE=3072
VE_DIM=160
VE_LAYERS=8,9,10
```

## Main expected risks and tradeoffs

- Sharing the late stack may over-regularize the deepest layers and remove useful specialization.
- The shared cores also share `q_gain` and the heavy late attention/MLP projections, so gains depend on the idea that the late stack is mostly refinement rather than strongly layer-specific computation.
- The larger BigramHash and broader VE coverage may or may not fully convert the recovered bytes into better BPB.
- This candidate has only been syntax-validated locally in this environment; GPU-side training quality remains the main open question.

## Validation run in this environment

Commands run locally in this container:

```bash
python -m compileall train_gpt.py
```

Outcome:

- passed successfully.

Attempted additional smoke validation:

```bash
python - <<'PY'
import torch
PY

python3 - <<'PY'
import torch
PY
```

Outcome:

- both local Python environments in this container were missing `torch`, so a faithful CPU-side start test was not feasible here,
- the candidate runtime also expects PyTorch plus CUDA/FlashAttention 3, so compile-only validation was the strongest low-cost check available in this environment.
