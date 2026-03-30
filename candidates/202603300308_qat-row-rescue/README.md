# Candidate: compile-safe late-QAT + AWQ-style row rescue

## Hypothesis

The strongest clean 11-layer stack in this repository is no longer bottlenecked by core architecture alone; it is bottlenecked by how much quality survives low-bit export. The next plausible gain is to make late QAT actually activate under `torch.compile`, then spend a tiny amount of artifact budget on salvaging the most outlier-heavy rows during int6 export.

## Why this is promising for this repository

The record history points in the same direction:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` is the strongest non-TTT stack and already frames GPTQ-lite plus averaging as worthwhile final-mile gains.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly notes that its late-QAT path was dead-code-eliminated by `torch.compile`, so there is still obvious headroom in making staged fake quantization real.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` reports a meaningful gain from `LeakyReLU(0.5)^2`, and also shows that bumping the bigram table beyond 2048 buckets helped on the same general architecture family.
- `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` shows that spending bits selectively, rather than uniformly, is a strong lever under the 16 MB cap.
- The non-record `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` notes that layer recurrence was actively harmful under fixed wall-clock, so this candidate stays focused on quantization/compression rather than depth reuse.

## External research that informed it

This candidate is mostly inspired by low-bit quantization work that protects the most sensitive channels instead of treating all weights equally:

- **SmoothQuant** (`arXiv:2211.10438`) argues that quantization quality often comes down to taming outliers with mathematically equivalent rescaling rather than only changing the base model.
- **AWQ** (`arXiv:2306.00978`) shows that preserving a small set of salient weights/channels can recover a surprising amount of low-bit quality.

I am not implementing full activation-calibrated AWQ here because this repository strongly prefers minimal, self-contained training scripts. Instead, I use an AWQ-inspired approximation: preserve a tiny fraction of the most outlier-heavy matrix rows in fp16 while keeping the rest of the matrix in GPTQ-lite int6.

## Base implementation

This candidate forks:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

That base already includes the strongest clean stack in the repo:

- 11 layers, 512 dim, GQA
- XSA on the last 4 layers
- partial RoPE + LN scaling
- VE128 on deep layers
- SmearGate + BigramHash
- EMA + tight SWA
- GPTQ-lite int6 export + zstd

## What changed versus the base

1. **LeakyReLU(0.5)^2 MLP activation by default**
   - Replaces `relu^2` with the activation that helped the current best record stack.

2. **Bigger default BigramHash table**
   - Raises the default `BIGRAM_VOCAB_SIZE` from `2048` to `3072` to test the same direction that helped the 2026-03-23 stack.

3. **Compile-safe late QAT gate**
   - Replaces the brittle class-level toggle with a per-layer Python-side QAT flag so staged fake quantization can turn on late in training without relying on a constant-folded class attribute.

4. **AWQ-style row rescue during export**
   - During int6 export, a tiny fraction of the most outlier-heavy rows from selected attention/MLP matrices are stored in their source float dtype and restored exactly after dequantization.
   - This is meant to protect the same kind of rare but important channels that AWQ highlights, while keeping the implementation simple and self-contained.

5. **FlashAttention import fallback for smoke testing**
   - If `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA, which makes lightweight CPU/import smoke tests possible.

6. **Candidate-directory-friendly defaults**
   - Default dataset and tokenizer paths resolve from the repository root based on `__file__`, so the script can be run from inside this candidate directory without editing paths first.

## How to run

From the repository root:

```bash
cd candidates/202603300308_qat-row-rescue
RUN_ID=qat_row_rescue torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs for ablations:

```bash
LEAKY_RELU_SLOPE=0.5 \
ROW_RESCUE_FRAC=0.005 \
ROW_RESCUE_PATTERNS=attn.c_q.weight,attn.c_k.weight,attn.c_v.weight,mlp.fc.weight \
LATE_QAT_THRESHOLD=0.15 \
BIGRAM_VOCAB_SIZE=3072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Ran the following lightweight validation in this container:

- `python -m compileall candidates/202603300308_qat-row-rescue/train_gpt.py`
- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603300308_qat-row-rescue/train_gpt.py`

Outcomes:

- `compileall` passed for the root Python entrypoints, `data/`, and the new candidate script.
- A tiny CPU import/forward smoke test was attempted next, but this workflow container does not have the runtime dependency stack installed — both `python` and `python3` raise `ModuleNotFoundError: No module named 'torch'`. Because of that, a real import-time model smoke test was **not feasible in this environment** without first installing heavyweight repo dependencies.

## Risks and tradeoffs

- The row-rescue path spends extra artifact bytes, so the rescue fraction has to stay tiny.
- The rescue heuristic is **weight-outlier-based**, not full activation-aware calibration, so it is only an AWQ-inspired approximation.
- The compile-safe QAT path may still cost some throughput versus the pure non-QAT 2026-03-22 base.
- Increasing `BIGRAM_VOCAB_SIZE` helps quality in prior runs but also pushes artifact size upward.
- This candidate deliberately avoids recurrence / layer reuse because the repository already has negative evidence for that direction under a fixed wall-clock budget.
