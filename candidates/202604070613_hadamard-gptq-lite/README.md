# Hadamard-GPTQ-lite

## Hypothesis

The current frontier in this repository is mostly bottlenecked by the **post-training int6 quantization gap**, not by a lack of raw training-time quality. A fixed Walsh-Hadamard input mixing on the largest power-of-two projections should make those weight matrices more isotropic and easier for GPTQ-lite/int6 export to preserve, while keeping the model class unchanged and code size small.

## Why this is promising here

- The strongest training-only line in this repo already converged on **11L / 512d / XSA / Partial RoPE / LN-scale / EMA / GPTQ-lite** and won mostly by reducing export damage rather than by broad architecture churn.
- This candidate keeps that stack and adds a **rotation-inspired quantization aid** that is cheap enough for the 10-minute setting.
- The transform is **parameter-free** and only touches projections whose input dimension is already a power of two, so it fits the existing script without new infrastructure.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`: best training-only base and the immediate fork point.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`: reused the proven `LeakyReLU(0.5)^2` MLP change, but **not** the TTT or Parallel Muon complexity.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`: reinforced that Partial RoPE + LN scaling belong in the base stack.
- Earlier int6 / mixed-quant runs from `2026-03-19` through `2026-03-20`: these established that most durable gains came from better compression-aware capacity allocation.

## External research that informed it

- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** (Ashkboos et al., 2024) showed that orthogonal rotations can preserve function while reducing quantization outliers through the identity `Wx = (WQ^T)(Qx)`.
- **SpinQuant: LLM Quantization with Learned Rotations** (Liu et al., 2024/2025) showed that rotation choice materially affects quantization quality on hard-to-quantize models.
- **OptRot: Mitigating Weight Outliers via Data-Free Rotations for Post-Training Quantization** (Gadhikar et al., 2025) strengthened the case that even lightweight, data-free rotation heuristics can improve weight quantization.

This candidate intentionally implements the **smallest practical version** of that family: fixed Hadamard mixing on selected inputs, not learned rotations or calibration-heavy search.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. Added a normalized **fast Walsh-Hadamard transform** and enabled it on the biggest power-of-two-input projections:
   - attention `q`, `k`, `v`, and output projections,
   - MLP expansion (`fc`),
   - BigramHash projection,
   - ValueEmbedding projection.
2. Switched the MLP nonlinearity from `relu^2` to **`LeakyReLU(0.5)^2`**.
3. Added a **FlashAttention import fallback** to standard SDPA so the module can still be imported when `flash_attn_interface` is unavailable during local smoke/debug runs.
4. Kept the inherited **late-QAT** option but now **rebuild the compiled model** when it turns on, so the fake-quant branch cannot be dead-code-eliminated by an earlier compile trace.
5. Left the rest of the strong March 22 stack intact: XSA, Partial RoPE, LN scaling, SmearGate, BigramHash, VE, EMA, GPTQ-lite export, and the existing training schedule.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202604070613_hadamard-gptq-lite
RUN_ID=hadamard_gptq_seed1337 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves its default dataset and tokenizer paths relative to the repository root, so this works from inside the candidate directory.

Useful ablations:

```bash
# Disable the new rotation path
HADAMARD_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep the new stack but only rotate 512-dim inputs, not 128-dim auxiliaries
HADAMARD_MIN_DIM=512 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- The extra Hadamard mixing adds runtime overhead; if it costs too many steps, the quantization win may not repay it.
- Only the **input-side** power-of-two projections are rotated. The MLP down-projection (`1536 -> 512`) is intentionally left untouched.
- A fixed Hadamard basis is cheaper than learned rotations, but it may underperform a searched/learned rotation on this model family.
- The SDPA fallback is for portability only; real leaderboard-style runs should still use the fast attention path.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202604070613_hadamard-gptq-lite/train_gpt.py
```

Outcome:

- `compileall` **passed**.
- A CPU import/forward smoke test was **not feasible in this workflow environment** because `torch` is not installed here (`ModuleNotFoundError: No module named 'torch'`), so only syntax-level validation was possible locally.
