# AWQ-lite activation-aware int6 clip search

## Hypothesis

The strongest current export path in this repo is still leaving quality on the table because it chooses int6 clip ranges from **weight-only reconstruction error** and, in practice, still shares the same percentile candidate across an entire tensor. A small amount of **activation-aware calibration** should improve roundtrip int6 quality at essentially zero training cost by choosing the clip percentile **per row** and weighting the choice by the channels the model actually uses.

## Why this is promising here

The record history keeps reinforcing the same lesson: once the model family moved to the 10L/11L + MLP3x + seq2048 regime, a lot of the remaining gains came from **export quality**, not just training changes.

- `2026-03-18_FP16Embed_WD3600` showed that treating sensitive tensors more carefully during export mattered immediately.
- `2026-03-19_MixedQuant_Int6Int8_SlidingWindow`, `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`, and `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` all pushed on mixed precision / clip selection instead of only changing the architecture.
- The newest record (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) suggests the obvious low-cost architecture knobs are already being harvested, so an orthogonal **post-training export improvement** is a good next bet.

## Prior repo work that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
   Chosen as the base implementation because it is the strongest pre-TTT export-centric stack in the repo.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`  
   Supplies the same strong 11-layer architecture core that the 2026-03-22 record keeps.
3. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`  
   Useful as a reminder that the next robust gain may come from something that stacks with the latest record rather than redoing its training/eval tricks.

## External research that informed it

1. **AWQ** — *Activation-aware Weight Quantization for LLM Compression and Acceleration* (arXiv:2306.00978)  
   Motivated using activation statistics, not just weight magnitudes, when deciding what to protect during low-bit export.
2. **SqueezeLLM** — *Dense-and-Sparse Quantization* (arXiv:2306.07629)  
   Reinforced that sensitivity is highly non-uniform across weights and that selective protection is often better than uniform quantization.
3. **SpQR** — *Sparse-Quantized Representation* (arXiv:2306.03078)  
   Strengthened the case that low-bit error is often dominated by a small subset of problematic channels / weights.

I also reviewed more architecture-heavy ideas like **Hyper-Connections** (arXiv:2409.19606) and **Intra-Layer Recurrence** (arXiv:2505.01855), but the repo history already shows recurrence-like ideas struggling under the 10-minute wallclock cap, so I chose the lower-risk export path first.

## What changed versus the base implementation

Starting from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four targeted changes:

1. **Real per-row clip selection** for int6 export instead of a single tensor-global winner among the candidate percentiles.
2. **Activation-aware calibration** on a small training-token sample (`AWQ_CALIBRATION_TOKENS`, default `131072`) using forward hooks on the large int6-targeted linear layers.
3. **Activation-weighted clip scoring** using per-channel second moments, so rows are judged more harshly when they distort frequently-used activation channels.
4. **FlashAttention fallback to PyTorch SDPA** so the model class can be imported and smoke-tested without Hopper-specific attention bindings.

The architecture, optimizer split, EMA/SWA defaults, and overall training loop are otherwise inherited from the 2026-03-22 base.

## How to run

From this candidate directory:

```bash
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs for the new export path:

```bash
AWQ_ENABLED=1
AWQ_CALIBRATION_TOKENS=131072
AWQ_BATCH_SEQS=8
AWQ_CLIP_CANDIDATES=0.9990,0.9995,0.9999,0.99999,1.0
```

## Expected risks / tradeoffs

- The calibration sample comes from training shards, so it may not perfectly reflect the validation distribution.
- The extra calibration pass adds a short post-training phase before export.
- If the remaining quantization error is dominated by a few true outlier weights rather than bad clip selection, the next follow-up should probably add sparse residual protection instead of more percentile tuning.
- The SDPA fallback is only for portability and smoke checks; the intended fast path remains FlashAttention on CUDA.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604061057_awq-lite-int6/train_gpt.py
```

Outcome: **passed**.

Attempted CPU smoke test:

```bash
python - <<'PY'
import importlib.util
import pathlib
import torch

path = pathlib.Path("candidates/202604061057_awq-lite-int6/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_train", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
...
PY
```

Outcome: **not feasible in this container** because the available Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
