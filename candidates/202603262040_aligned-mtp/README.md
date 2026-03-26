# Aligned train-only MTP on top of the 2026-03-23 stack

## Hypothesis

The strongest current 11-layer stack should benefit from **multi-token prediction (MTP) as a training-only auxiliary loss**. This repository is heavily wallclock-constrained, so sample efficiency matters more than raw asymptotic quality. Because the challenge vocabulary is only `1024`, a small number of auxiliary future-token heads is relatively cheap to train here, and the parent script already strips those heads before export, so the final artifact size should stay anchored to the base model rather than the auxiliary heads.

## Why this is promising for this repository

- The repo's biggest wins already came from techniques that improve **effective context** or **quality-per-step**: sliding-window eval, legal TTT, quantization-aware training, EMA/SWA, and small activation changes.
- No prior record in `records/` used trainable MTP, even though the latest record code already contains dormant support for `mtp_heads`.
- The best current record (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) already excludes `mtp_heads.*` from the exported state dict, which makes MTP unusually attractive here: more training signal, but no required increase in submission bytes.
- Compared with mainstream large-vocab LLMs, the Parameter Golf setting makes extra prediction heads cheaper because the vocabulary and model width are both small.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-17_NaiveBaseline` established the base 9x512 tied-embedding recipe and the post-quant roundtrip metric.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval` showed that evaluation-aware techniques can buy a large BPB gain without touching training.
- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow` and `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` showed that compression-funded capacity and quantization-aware regularization are the winning model-side pattern.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` and `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` established the current 11-layer frontier stack before TTT.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` is the direct base implementation for this candidate.

There was **no pre-existing `candidates/` directory** when this candidate was created.

## External research that informed it

- **Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_ (arXiv:2404.19737)** argues that predicting multiple future tokens improves pretraining sample efficiency through auxiliary heads on a shared trunk.
- **Gerontopoulos et al., _Multi-Token Prediction Needs Registers_ / MuToR (arXiv:2505.10518)** highlights that MTP works best when it stays aligned with the base next-token objective and keeps extra parameter cost low.
- **Mehra et al., _On multi-token prediction for efficient LLM inference_ (arXiv:2502.09419)** cautions that MTP heads can mismatch hidden states specialized for next-token prediction, which motivated a conservative setup here rather than a large MTP horizon.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps the parent stack intact:

- 11-layer 512d banked transformer
- LeakyReLU(0.5)^2 MLP
- BigramHash(1536) + SmearGate
- Partial RoPE + LN scale + VE128
- GPTQ-lite-style mixed int6 export + `lzma`
- optional sliding-window eval and legal score-first TTT
- parameter banking + Parallel Muon

It makes one focused extension:

1. **Turn on MTP by default**
   - `MTP_NUM_HEADS` default: `0 -> 2`
   - `MTP_LOSS_WEIGHT` default: `0.20 -> 0.15`

2. **Make the MTP heads actually train**
   - The parent script defined `mtp_heads`, used them in the loss, and stripped them from export, but did **not** add those head weights to any optimizer.
   - This candidate adds the MTP head weights to the Adam "head" optimizer and the replicated-gradient all-reduce list, so enabling MTP now has a real training effect.

3. **Align MTP heads with the tied next-token head**
   - Added `MTP_INIT_FROM_TIED=1` (default on).
   - When embeddings are tied, the auxiliary MTP heads start from `tok_emb.weight` instead of a dead zero init.
   - The intent is to keep auxiliary prediction closer to the base NTP geometry and reduce early optimization burn-in.

4. **Keep export byte-neutral**
   - The export path still removes all `mtp_heads.*` tensors before save, quantization, roundtrip reload, and final eval.
   - The final artifact therefore stays comparable to the parent stack rather than paying for the auxiliary heads.

## How to run / evaluate

The candidate is self-contained inside this directory.

### Training + roundtrip + sliding-window eval

```bash
RUN_ID=aligned_mtp \
BIGRAM_VOCAB_SIZE=1536 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_INIT_FROM_TIED=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Training + legal TTT evaluation (closer to the best current record)

```bash
RUN_ID=aligned_mtp_ttt \
BIGRAM_VOCAB_SIZE=1536 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_INIT_FROM_TIED=1 \
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If the extra auxiliary loss costs too many training steps, the first sweep I would run is:

- `MTP_NUM_HEADS=1`
- `MTP_LOSS_WEIGHT in {0.10, 0.15, 0.20}`

## Main expected risks / tradeoffs

- **Throughput risk:** even cheap MTP adds extra logits and loss work, so wallclock-limited step count may drop.
- **Objective mismatch risk:** better pretraining loss does not automatically translate to better post-quant + sliding-window + TTT BPB.
- **Initialization risk:** copying from the tied embedding matrix may stabilize training, but it could also over-constrain the auxiliary heads if they need to specialize quickly.
- **Quantization interaction:** MTP is train-only here, but it still shapes the trunk weights that later get quantized; the net effect on quantized BPB is uncertain until run.

## Validation

Commands run in this repository:

```bash
python -m compileall candidates/202603262040_aligned-mtp/train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path("candidates/202603262040_aligned-mtp/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("import_smoke_ok")
PY
```

Outcomes:

- `python -m compileall ...` **passed**
- The import-only smoke test was **not feasible in this runner** because repository runtime dependencies were missing (`numpy` and `torch` were not installed), and this script also expects the CUDA/FlashAttention stack used by the record implementations
