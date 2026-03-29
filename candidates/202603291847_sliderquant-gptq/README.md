# SliderQuant GPTQ-lite profile search

## Hypothesis

The clean 2026-03-22 stack still quantizes every large attention/MLP weight matrix with the same post-training `int6` recipe. Recent PTQ work argues that this is suboptimal: shallow and deep layers are usually more quantization-sensitive than intermediate layers, and the first/last layers are often the most fragile. This candidate tests whether a tiny, repo-native layerwise mixed-precision search can reduce post-quantization damage without changing the proven training recipe or adding new infrastructure.

## Why this is promising for this repository

The repo evidence points to quantization quality as the cheapest remaining lever:

- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/` showed that better post-training quantization almost eliminated the early quantization gap.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` and `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` converged on a stable 11-layer training stack where export quality, not base training instability, looked like the next bottleneck.
- The non-record recurrence experiments were a warning that bigger architectural swings are risky under the 10-minute wallclock budget, while post-training compression changes remain almost free at train time.

So this candidate deliberately spends complexity at export time rather than in the forward or backward pass.

## Prior records and candidates that influenced this idea

There was no pre-existing `candidates/` directory when this folder was created.

The strongest local influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen base implementation. It is the cleanest mature pre-TTT stack in the repository and already includes GPTQ-lite percentile search, EMA, Tight SWA, Partial RoPE, LN scale, XSA, BigramHash, and shared value embeddings.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Confirmed that Partial RoPE + LN scale were real wins while the late-QAT branch was effectively inert under `torch.compile`.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Demonstrated that the latest top score is already consuming a different axis of improvement (activation + TTT + optimizer systems work). This candidate intentionally targets an orthogonal compression-only twist instead of repeating that stack.

## External research

This candidate is primarily informed by three recent PTQ papers:

- **SliderQuant** ([arXiv:2603.25284](https://arxiv.org/abs/2603.25284))
  - Key observation: shallow/deep layers are more sensitive than middle layers, and the first/last layer are usually the most fragile.
  - Relevance here: the 03-22 base still uses uniform block-weight quantization, so a depth-aware schedule is the most natural minimal extension.
- **RAMP** ([arXiv:2603.17891](https://arxiv.org/abs/2603.17891))
  - Key observation: per-layer bit allocation under a global budget beats uniform precision, and sensitivity generalizes architecturally.
  - Relevance here: this repository already has a hard 16MB artifact cap, so choosing the best schedule against the actual serialized artifact size is a direct fit.
- **CoopQ** ([arXiv:2509.15455](https://arxiv.org/abs/2509.15455))
  - Key observation: mixed-precision quantization improves when it accounts for inter-layer effects instead of treating every layer in isolation.
  - Relevance here: the implementation searches whole-model depth profiles rather than greedily tweaking one tensor at a time.

## What changed versus the chosen base implementation

This folder copies the 2026-03-22 script and makes only candidate-local changes:

1. **Budget-aware layerwise quantization profiles**
   - Added a small search over depth profiles such as `edge8-near7-mid6`, `edge7-near6-mid6`, and `6-6-6`.
   - Each profile quantizes the full exported model, compresses it with the same serializer, measures the real submission bytes, and keeps the lowest weighted reconstruction-MSE profile that still fits the byte budget.
   - The attention/MLP stacks inside `blocks.<idx>` get depth-dependent precision; everything else stays on the original float/int8-style path.

2. **Generic per-row quantizer for 2-8 bit search profiles**
   - Generalized the old fixed-`int6` row quantizer into a reusable helper that searches the same percentile grid for different clip ranges.

3. **Run-from-candidate-directory defaults**
   - `DATA_PATH` and `TOKENIZER_PATH` now default relative to the repository root inferred from the candidate file location, so `cd candidates/... && torchrun ... train_gpt.py` works without extra path overrides.

4. **FlashAttention fallback for local smoke/import checks**
   - If `flash_attn_interface` is unavailable, the script falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - This keeps the candidate importable in lightweight environments while preserving the original FlashAttention path for real GPU runs.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603291847_sliderquant-gptq
RUN_ID=sliderquant_gptq \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Because the script resolves the repository root automatically, you do not need to override `DATA_PATH` or `TOKENIZER_PATH` when running from this candidate directory in the standard repo layout.

Optional knobs:

```bash
QUANT_PROFILE_CANDIDATES='8-7-6;8-6-6;7-7-6;7-6-6;6-6-6' \
SUBMISSION_SIZE_BUDGET=16000000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

The following low-cost checks were run in this workflow container:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603291847_sliderquant-gptq/train_gpt.py
```

Outcome: **passed**.

A CPU import/forward smoke test was also attempted with an inline Python snippet that imports the candidate module, instantiates a tiny `GPT`, and exercises the new quantization search path. That runtime smoke check could not complete in this container because the Python environment does not include PyTorch:

```text
ModuleNotFoundError: No module named 'torch'
```

So this candidate has syntax validation in the current environment, but not a true runtime forward-pass validation yet.

## Main expected risks and tradeoffs

- **MSE proxy risk**: the profile search uses weighted reconstruction MSE, which is only a proxy for downstream `val_bpb`.
- **Export-time overhead**: searching several whole-model quantization profiles adds extra post-training time, although it does not affect the training loop.
- **Budget sensitivity**: the best-quality depth profile might miss the 16MB cap after compression, so the search can fall back to a more conservative profile.
- **Still a minimal approximation**: SliderQuant, RAMP, and CoopQ are richer than this implementation. This candidate only captures the simplest repo-native version of their shared idea: not all layers should get the same quantization treatment.
