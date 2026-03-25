# LeakyReLU² + Activation-Aware GPTQ-lite Clip Search

## Hypothesis

The strongest non-TTT 11-layer stack in this repo is still using a weight-only GPTQ-lite clip search. Borrowing the activation-awareness idea from AWQ/OmniQuant should make the int6 export track real runtime sensitivity better than plain weight MSE, while LeakyReLU(0.5)^2 brings in the best recent MLP activation change from the current record family.

In short: **keep the proven `1.1233` architecture/training stack, but make the export more activation-aware and the MLP less lossy**.

## Why this is promising for this repository

This repo's frontier has already squeezed large gains out of architecture depth, eval-time context, EMA/SWA, and mixed int6 export. The remaining gap increasingly looks like a **quantization gap** problem:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved the previous record mainly by making the int6 export smarter with row-wise clip search plus EMA.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` showed that **LeakyReLU(0.5)^2** improves even the pre-TTT model, so it is worth testing outside the TTT stack too.
- Recent QAT literature argues that as token counts grow, **weight quantization error matters more**, which matches this repo's 5B+ token / 10-minute regime.

That makes an activation-aware export pass a good next candidate: it is directly aimed at the current bottleneck, adds almost no artifact cost, and adapts cleanly to the existing code.

## Prior repo work that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Base implementation copied here.
  - Strong 11L, XSA4, partial-RoPE, LN-scale, BigramHash, VE, EMA/SWA, GPTQ-lite int6 export stack.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Source of the `LeakyReLU(0.5)^2` activation choice.
  - Shows the activation change is valuable even before TTT.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Useful caution that late QAT can be easy to over-credit if the compiled path is not actually exercising it.

## External research that informed it

- **AWQ: Activation-aware Weight Quantization** ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978))
  - The key takeaway is that activation statistics are better than weight magnitude alone for identifying which channels matter during low-bit quantization.

- **GPTQ** ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323))
  - Motivates spending extra effort on smarter one-shot weight quantization because the compression/quality frontier is often decided there.

- **OmniQuant** ([arXiv:2308.13137](https://arxiv.org/abs/2308.13137))
  - Reinforces the idea that learnable or calibrated clipping thresholds are a productive direction when plain hand-tuned quantization parameters start to saturate.

- **Scaling Law for Quantization-Aware Training** ([arXiv:2505.14302](https://arxiv.org/abs/2505.14302))
  - Especially relevant here: with more training tokens, weight quantization error can dominate, which suggests this repo's long-token 10-minute runs should benefit from more careful weight export.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. **LeakyReLU(0.5)^2 MLP**
   - Replaces `relu^2` with the repo-proven LeakyReLU variant.

2. **Activation-aware calibration during late training**
   - Runs a short, no-grad pass over training batches **before training finishes**, once the LR scale drops into late-training territory.
   - Collects per-input-channel RMS statistics for every `CastedLinear` weight matrix.

3. **Activation-aware GPTQ-lite clip selection**
   - The existing row-wise percentile search is preserved.
   - Percentile selection now minimizes an **activation-weighted reconstruction error** instead of plain weight MSE, approximating output error better.
   - Export only reuses statistics gathered during training; it does not reopen training shards after training ends.

4. **CPU-friendly attention fallback**
   - If `flash_attn_interface` is unavailable, attention falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - This does not change the intended H100 path, but makes local import/smoke checks more practical when FlashAttention is absent.

5. **Candidate-directory-friendly defaults**
   - Default dataset and tokenizer paths are resolved relative to the repository root via `__file__`, so the script can be launched from inside this candidate directory as required.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202603251430_leaky-awq-gptq
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
AQ_CLIP_ENABLED=1
AQ_CALIBRATION_STEPS=8
AQ_CALIBRATION_THRESHOLD=0.2
AQ_PERCENTILES=0.9990,0.9995,0.9999,0.99999,1.0
```

Notes:

- Defaults already point at the repository dataset/tokenizer under `../../data/...` via path resolution from `train_gpt.py`.
- The export remains mixed int6/int8 with zstd when available, following the chosen base.

## Main expected risks / tradeoffs

- **Calibration overhead:** the activation-aware pass adds a little extra post-training time. It should be small, but it is not free.
- **No guarantee weighted clip error matches BPB perfectly:** activation RMS is only a cheap proxy for downstream sensitivity.
- **Interaction with existing late-QAT path remains uncertain:** this candidate targets the export/calibration side first rather than trying to rework the training-time fake-quant path.
- **LeakyReLU transfer may not be additive:** it worked in the current top record family, but its gain on this exact non-TTT stack still needs measurement.

## Validation run for this candidate

Executed in this workflow:

- `python -m compileall candidates/202603251430_leaky-awq-gptq/train_gpt.py`
  - **Passed**

- CPU smoke test attempt: import the candidate module, instantiate a tiny CPU model, run a forward pass, and roundtrip mixed quantization.
  - **Not feasible in this runner** because the workflow environment does not have `torch` installed for Python runtime execution (`ModuleNotFoundError: No module named 'torch'`).
  - Because adding PyTorch would require heavyweight dependency installation outside the repository's existing lightweight checks, this workflow only validated syntax here.
