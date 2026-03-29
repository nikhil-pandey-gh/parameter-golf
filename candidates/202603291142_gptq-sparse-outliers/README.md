# GPTQ-lite + Sparse Outlier Rescue

## Hypothesis

The repository's strongest recent gains have come from reducing post-training quantization damage, not from broad architecture churn. This candidate tests whether a **tiny sparse side channel for the worst GPTQ-lite residuals** can recover more of the pre-quant model quality than plain int6 GPTQ-lite, while still staying inside the 16MB artifact budget.

Concretely, the candidate keeps the strong 11-layer EMA + XSA + Partial RoPE + LN-scale stack from the 2026-03-22 record, still quantizes most block weights to int6, but additionally stores a very small set of the largest residual weights in higher precision. The hope is that a few hundred outlier corrections per large matrix are enough to improve BPB more than their byte cost hurts the final artifact.

## Why this is promising for this repository

Recent repo history points in the same direction:

- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/` showed that better export policy unlocked most of the jump from the naive baseline.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` and `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` suggest the 11-layer training stack is already strong and that small quantization refinements still matter.
- The 4-hour non-record baseline in `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` further highlights that quantization remains a real limiter even when pre-quant quality improves.

That makes a **compression-aware export tweak** more attractive than a riskier architecture rewrite. The candidate avoids broad new infrastructure and only changes the quantization/dequantization path plus a local attention fallback used for smoke testing.

## Prior records that influenced this candidate

The main implementation base is:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

The design was also informed by:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md` for the stabilized 11-layer EMA + XSA stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` for Partial RoPE + LN scale.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` as evidence that the frontier is now being decided by small deltas on already-strong stacks.

I intentionally did **not** base this on the latest legal-TTT record because the goal here is to isolate a new export-side idea with enough artifact headroom to make a sparse side channel realistic.

## External research that informed it

This candidate is grounded in the same family of observations from recent quantization papers:

- **GPTQ**: one-shot weight quantization remains a strong baseline for transformer PTQ. https://arxiv.org/abs/2210.17323
- **AWQ**: only a very small subset of weights/channels tend to dominate quantization error; protecting roughly 1% salient weights can materially reduce degradation. https://arxiv.org/abs/2306.00978
- **SpQR**: explicitly isolating outlier weights into a sparse higher-precision side channel can preserve perplexity at aggressive bitrates. https://arxiv.org/abs/2306.03078
- **SqueezeLLM**: dense-and-sparse decomposition is a practical way to improve ultra-low-bit weight-only quantization under the same memory budget. https://arxiv.org/abs/2306.07629
- **SmoothQuant**: broader motivation that outlier structure is often the root cause of PTQ damage, even though that paper targets W8A8 rather than this repo's weight-only export path. https://arxiv.org/abs/2211.10438

The implementation here is deliberately minimal: it does **not** attempt full activation-aware calibration or Hessian-aware sparse selection. It only keeps the largest post-quant residuals for each int6 matrix.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Sparse outlier rescue on top of GPTQ-lite int6**
   - After the usual per-row GPTQ-lite clip search, the script computes the quantization residual.
   - It stores a capped top-k set of the largest residual entries for each int6 matrix as a sparse side channel.
   - During dequantization, those residuals are scattered back into the reconstructed dense tensor.

2. **New tuning knobs**
   - `INT6_OUTLIER_FRAC` controls the target fraction of residual entries to preserve.
   - `INT6_OUTLIER_MAX_K` hard-caps the number of preserved residuals per tensor so artifact growth stays predictable.

3. **CPU-friendly attention fallback**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch `scaled_dot_product_attention`.
   - This is mainly to make import/smoke validation easier in lighter environments; the intended full run is still the CUDA path.

## Files added

- `candidates/202603291142_gptq-sparse-outliers/train_gpt.py`
- `candidates/202603291142_gptq-sparse-outliers/README.md`

## How to run or evaluate

From the repository root:

```bash
cd candidates/202603291142_gptq-sparse-outliers
RUN_ID=gptq_sparse_outliers \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
INT6_OUTLIER_FRAC=5e-4 \
INT6_OUTLIER_MAX_K=512 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate inherits the 11-layer default stack from the 2026-03-22 record, including EMA, XSA on the last 4 layers, Partial RoPE (16/64), LN scale, and GPTQ-lite-style per-row clip search.

## Validation

The following validation commands were run in this workflow:

```bash
python -m compileall candidates/202603291142_gptq-sparse-outliers/train_gpt.py
```

Outcome: **passed**.

A CPU smoke import/forward test was also attempted with a tiny randomly initialized model, but it was **not feasible in this environment** because the runner does not currently have `torch` installed:

```bash
python - <<'PY'
import importlib.util
# import candidate module, instantiate a tiny GPT, and run a forward pass
PY
```

Outcome: failed immediately with `ModuleNotFoundError: No module named 'torch'`.

`requirements.txt` in the repository root lists `torch`, so the candidate should be smoke-tested again in a repo-typical Python environment before any expensive multi-GPU run.

## Main expected risks and tradeoffs

- **Artifact-size risk**: if the sparse side channel grows faster than the recovered BPB improves, the net submission score could get worse. `INT6_OUTLIER_MAX_K` exists specifically to bound that risk.
- **Naive saliency heuristic**: this version picks the largest residuals by magnitude only. It does not use activation-aware calibration like AWQ or second-order sensitivity like full GPTQ/SqueezeLLM-style methods.
- **Decompression overhead**: the sparse scatter-add path is tiny compared with training, but it still adds some complexity and a bit of eval-time work.
- **Possible diminishing returns**: the repo's current GPTQ-lite path is already fairly good, so the win may be smaller than the papers suggest for much larger models.

## Suggested next experiments

If this candidate is directionally promising, the most natural follow-ups are:

1. Apply sparse rescue only to the most sensitive matrices (for example MLP up/down and attention output projections) instead of all int6 block weights.
2. Tune `INT6_OUTLIER_FRAC` and `INT6_OUTLIER_MAX_K` jointly against artifact bytes.
3. Replace the weight-only residual heuristic with a tiny activation-aware calibration pass, closer to AWQ, using train-stream statistics gathered during training.
