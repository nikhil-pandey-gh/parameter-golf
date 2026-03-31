# Candidate: 11L EMA + GPTQ-lite + MLP Cubic Equalization

## Hypothesis

The current best non-TTT 11-layer stack already trains well; a remaining weakness is the final **int6 export gap**. This candidate tests whether a **function-preserving, export-time MLP equalization pass** can shrink that gap before the existing GPTQ-lite quantizer runs.

The key observation is that this model's MLP is:

```python
h = relu(W_up x)
y = W_down (h ** 2)
```

Because `relu(.)` is positively homogeneous and the activation is squared, the hidden channel transform is **degree-2 homogeneous**. In exact arithmetic, for any positive per-channel scale `alpha`,

```text
W_up'   = diag(alpha)      W_up
W_down' = W_down diag(alpha^-2)
```

produces the **same real-valued reparameterization** while changing the weight ranges that the exporter later quantizes. In the actual script path, the equalizer additionally snaps `alpha` to powers of two so the paired bf16 casts behave more like exact exponent shifts, and it logs a **post-equalization pre-quant eval** to measure any residual drift before the GPTQ-lite export runs.

## Why this is promising for this repository

The repo history shows a consistent pattern: most of the leaderboard progression came from **better post-training survival under aggressive compression**, not from radical architecture changes. The strongest pre-TTT lineage is:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`

Those records kept the same broad 11L recipe while stacking small quantization-aware wins:

- EMA
- Partial RoPE + LN scaling
- GPTQ-lite clip search
- longer warmdown

This candidate stays in that exact spirit: it leaves the trained model family alone and targets the artifact bottleneck directly. It also focuses on the **MLP pair**, which earlier records repeatedly identified as high-leverage:

- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md` calls MLP 3x the biggest contributor in that run.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` shows that even late in the repo history, MLP-side tweaks can still move the score materially.

No prior `candidates/` directory existed when this candidate was created, so there were no earlier candidate iterations to inherit from or avoid repeating.

## External research that informed it

This candidate is a small, repo-compatible adaptation of a broader quantization idea: **move scale across mathematically equivalent layers to reduce outliers before quantization**.

- **Data-Free Quantization Through Weight Equalization and Bias Correction** (Nagel et al., ICCV 2019, arXiv:1906.04721) showed that scale-equivariant networks can be rebalanced before PTQ to improve quantized accuracy.
- **SmoothQuant** (Xiao et al., ICML 2023, arXiv:2211.10438) showed that offline scale migration can make LLM quantization easier by moving quantization difficulty across equivalent transforms.
- **AWEQ** (Li et al., arXiv:2311.01305) applied activation-weight equalization ideas directly to LLM PTQ.
- **Weight Equalizing Shift Scaler-Coupled Post-training Quantization** (Oh et al., arXiv:2008.05767) is especially relevant to this implementation detail: snapping equalization scales to binary shifts can preserve hardware-friendly behavior while still reducing outlier imbalance.
- **QuaRot** (Ashkboos et al., arXiv:2404.00456) and **SpinQuant** (Liu et al., arXiv:2405.16406) reinforce the same higher-level lesson: reshaping outlier distributions before low-bit quantization can be a first-order lever even when the full-precision model is unchanged.

The twist here is repository-specific: because this stack uses **ReLU-squared MLPs**, the equalization law becomes **cubic** instead of the usual square-root-style rescaling used for ReLU-like pairs.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. Added three export controls:
   - `MLP_EQUALIZE=1`
   - `MLP_EQUALIZE_QUANTILE=0.9995`
   - `MLP_EQUALIZE_MAX_FACTOR=4.0`
   - `MLP_EQUALIZE_POW2=1`

2. Added an **exact MLP channel equalization pass** that runs on the CPU export state dict just before mixed int6/int8 quantization.

3. For each block's MLP pair:
   - compute a robust abs-quantile statistic for `mlp.fc.weight` rows and `mlp.proj.weight` columns,
   - choose
     ```text
     alpha = (down_stat / up_stat)^(1/3)
     ```
   - clamp and geometric-mean normalize `alpha`,
   - optionally snap `alpha` to powers of two to reduce bf16 rounding drift on the executed path,
   - rewrite the pair as:
     - `fc.weight <- alpha * fc.weight`
     - `proj.weight <- proj.weight / alpha^2`

4. Log an estimated pre/post **int6 roundtrip MSE** summary before the actual GPTQ-lite export begins.

5. Run an explicit **post-equalization pre-quant validation pass** inside the script so the export logs reveal whether the bf16 execution path stayed effectively unchanged.

Everything else stays intentionally close to the parent record: 11L stack, XSA, partial RoPE, LN scale, EMA, warmdown 3500, and the same mixed int6/int8 GPTQ-lite export path.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603310633_mlp-cubic-eq
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults inherit the base record's 11L GPTQ-lite recipe and enable MLP equalization automatically.

Useful ablations:

```bash
# Disable the new idea entirely
MLP_EQUALIZE=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# More conservative equalization
MLP_EQUALIZE_MAX_FACTOR=2.0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Slightly more aggressive robust statistic
MLP_EQUALIZE_QUANTILE=0.9999 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks or tradeoffs

- The transform is **exact in full precision**, but it may still worsen compression if the equalized weights become less `zstd`-friendly even while local int6 reconstruction improves.
- The heuristic is **MLP-only**. If the remaining quantization bottleneck is actually in attention or embeddings, this will underperform.
- The robust statistic is intentionally simple and data-free. A calibration-aware variant could do better, but that would be a larger infrastructure jump than this candidate aims for.
- The scale clamp is a safety valve. Too little freedom may mute the effect; too much may create numerically awkward weights despite exact functional equivalence.

## Validation

Commands run while preparing this candidate:

```bash
python -m compileall candidates/202603310633_mlp-cubic-eq/train_gpt.py
```

Outcome:

- `compileall`: **passed**

CPU-only smoke test:

- **Not feasible in this runner**. The environment used for candidate preparation did not have `torch` installed, so it could not execute the training script far enough to validate CUDA / FlashAttention startup. This candidate therefore only received a syntax-level validation here; a real smoke test still needs a Python environment with the repository's training dependencies.
