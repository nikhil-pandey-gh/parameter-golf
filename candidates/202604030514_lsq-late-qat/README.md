# Candidate: LSQ-style Late QAT on the 2026-03-22 11L GPTQ-lite Base

## Hypothesis

The strongest missing improvement in this repository is not another large architectural rewrite, but a **learned quantizer** for the existing 11-layer int6 export path. Replacing fixed late-QAT row scales with **learned per-row LSQ-style scales** should reduce the remaining int6 roundtrip gap and improve sliding-window val_bpb without increasing model width or artifact size.

## Why this is promising here

- The best clean non-TTT record, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, already has a strong architecture and a strong export stack. The remaining room is mostly in **quantizer adaptivity**, not in obvious missing depth/width.
- Several repo records already show that quantization quality is a first-order bottleneck:
  - `2026-03-19_MLP3x_QAT_Int6_SlidingWindow` benefited from int6 QAT.
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly notes that its late-QAT path was accidentally compiled away, so there is still unresolved upside in a **working** late-QAT implementation.
  - `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved export quality with better fixed clip selection, which suggests learned scales are the next natural step.
- I intentionally did **not** fork the 2026-03-23 TTT + Parallel Muon record because this candidate is meant to isolate one training/export change on the strongest simpler base, not to entangle the result with evaluation-time adaptation.

## Prior repo influences

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- **Quantization motivation:** `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow`
- **Late-QAT bug to avoid:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
- **Best overall reference point:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
- **Prior candidates reviewed:** none; `candidates/` did not exist before this run.

## External research that informed this candidate

- **LSQ** — Esser et al., *Learned Step Size Quantization* (`arXiv:1902.08153`): motivates learning quantizer step sizes instead of freezing them from max/percentile heuristics.
- **AdaRound** — Nagel et al., *Up or Down? Adaptive Rounding for Post-Training Quantization* (`arXiv:2004.10568`): reinforces that fixed nearest rounding leaves quality on the table when the quantizer is allowed to adapt.
- **QuaRot** — Ashkboos et al. (`arXiv:2404.00456`) and **SpinQuant** — Liu et al. (`arXiv:2405.16406`): both point in the same direction as this candidate: the remaining gains are often in **making low-bit quantization easier**, not in growing the model.

I considered more invasive ideas from the same research pass (ALBERT-style layer sharing, recurrent depth reuse, compression-aware clustering), but the repo history already warns that recurrence is low-ROI under a hard 10-minute wallclock and the others would require much broader retuning.

## What changed vs the chosen base

This candidate keeps the 03-22 architecture, training recipe, EMA/SWA, XSA4, partial RoPE, BigramHash/SmearGate, VE layers, and GPTQ-lite-style export structure. The changes are intentionally narrow:

1. **Learned per-row late-QAT scales** for every `CastedLinear` weight via a log-scale parameter (`qat_log_scale`).
2. **LSQ-style straight-through fake quantization** in `CastedLinear.forward`, using learned row scales rather than recomputing row-max scales every time.
3. **Separate AdamW parameter group** for QAT scales with `QAT_SCALE_LR=0.005` by default and **no weight decay**.
4. **Late activation** still keys off `LATE_QAT_THRESHOLD`, but the enable path now toggles per-module runtime buffers instead of the previous compile-time class flag.
5. **EMA and AdamW state are reset for learned QAT scales at activation time**, so late-QAT runs do not average or optimize from stale pre-QAT scale state.
6. **Export uses the learned int6 scales directly** for int6 tensors and strips the training-only `qat_log_scale` tensors from the **quantized int6 artifact**.
7. **The raw `final_model.pt` checkpoint keeps the QAT scale parameters**, while the int6 roundtrip path reloads cleanly without needing them at inference time.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
SWA_ENABLED=1 SWA_EVERY=50 QAT_ENABLED=0 LATE_QAT_THRESHOLD=0.15 QAT_SCALE_LR=0.005 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

- `LATE_QAT_THRESHOLD=0.10`, `0.15`, `0.20`
- `QAT_SCALE_LR=0.003`, `0.005`, `0.010`
- `QAT_ENABLED=1` to test full-run LSQ-style QAT instead of late activation

## Main risks / tradeoffs

- **Runtime overhead:** the LSQ fake-quant path now exists in the compiled training graph, so even late-QAT adds more per-step math than the fixed-scale baseline.
- **Scale instability:** learned step sizes can collapse or grow too aggressively if the LR is too high.
- **Interaction with GPTQ-lite export:** this candidate trusts learned scales for int6 export instead of running the full percentile search on those tensors, which may help or hurt depending on how well the scales converge.
- **Still needs real GPU validation:** compile-time behavior is fixed locally, but the actual throughput/quality tradeoff needs an 8-GPU run.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604030514_lsq-late-qat/train_gpt.py
python - <<'PY'
import torch
PY
```

Outcomes:

- `python -m compileall ...` **passed**.
- A runtime smoke test was **not feasible** here because the environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), and this script also requires CUDA plus FlashAttention 3 for an actual training start.
