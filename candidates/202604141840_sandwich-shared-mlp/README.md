# Sandwich-Shared MLP

## Hypothesis

The strongest non-TTT stack in this repo already looks close to saturated on small training-side tweaks, so the next useful lever is **artifact-efficient parameter reallocation**. This candidate shares the **middle-layer MLP weights** in a Subformer-style sandwich pattern, then spends the saved budget on a **wider 3.5x feed-forward** path and a **larger bigram table**, while keeping the proven 11-layer XSA/EMA/GPTQ-lite recipe.

## Why this is promising here

- The repo already found that **naive recurrence / extra reused passes are bad** under a hard wallclock because they cut optimizer steps too much (`records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`).
- The 03-19 to 03-22 records show that the biggest wins came from **compression-funded capacity**, especially wider MLPs, not from tiny optimizer-only tweaks.
- This candidate keeps compute at the same **11 logical layers** and only shares stored weights, so it avoids the “double the depth, halve the steps” failure mode while still freeing artifact bytes.
- Compared to a 03-22-like no-sharing stack, the default config here drops total stored params from **26,993,756** to **24,503,388** while widening the MLP from **3.0x to 3.5x** and increasing `BIGRAM_VOCAB_SIZE` from **2048 to 3072**.

## Prior repo experiments that influenced this

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best pre-TTT training/quant stack in this repo,
  - already includes Partial RoPE, LN scale, XSA4, VE128, EMA, GPTQ-lite, and sliding eval.
- **Capacity trend:** `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - explicitly calls 3x MLP the biggest contributor.
- **Negative result to avoid:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - layer recurrence x2 was the worst sweep result because extra compute cost dominated.
- **Activation evidence:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - LeakyReLU(0.5)^2 gave a clean late-stage gain, so this candidate adopts it as part of the best-known base stack.

## External research

- **Subformer** (Reid et al., 2021, arXiv:2101.00234): sandwich-style sharing in generative transformers can beat naive cross-layer tying while using fewer parameters.
- **ALBERT** (Lan et al., 2019, arXiv:1909.11942): cross-layer sharing is a practical way to scale depth-efficient transformers under tight parameter budgets.
- **Scaling Law for Quantization-Aware Training** (Chen et al., 2025, arXiv:2505.14302): larger models reduce quantization error, which supports spending recovered artifact budget on capacity instead of leaving it unused.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (Nguyen and Lin, 2025, arXiv:2505.01855): targeted reuse is more promising than indiscriminate whole-block recurrence, matching the decision to share only the middle MLP cores.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

1. **Shared middle MLPs**
   - New default pattern: `SHARED_MLP_GROUPS=2-3,4-5,6-7`
   - Layers 2/3, 4/5, and 6/7 each reuse one MLP core.
   - All layers keep their own attention, norms, residual mixing, and per-layer scales.
2. **Wider feed-forward path**
   - Default `MLP_MULT` rises from `3.0` to `3.5` (hidden size `1536 -> 1792` at `MODEL_DIM=512`).
3. **LeakyReLU(0.5)^2**
   - `relu^2` becomes `leaky_relu(0.5)^2`.
4. **Bigger local feature table**
   - Default `BIGRAM_VOCAB_SIZE` rises from `2048` to `3072`.
5. **CPU-safe attention fallback**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA so import/smoke checks can run off-GPU.
6. **Late QAT disabled by default**
   - `LATE_QAT_THRESHOLD=0.0` because earlier repo history showed compiled late-QAT toggles were fragile; this candidate is intended to isolate the sharing/capacity hypothesis instead.

## How to run

From the repo root:

```bash
cd candidates/202604141840_sandwich-shared-mlp
RUN_ID=sandwich_shared_mlp \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves its default `DATA_PATH` and `TOKENIZER_PATH` from the repo root, so this command works even when launched from inside the candidate directory.

Key defaults:

- `NUM_LAYERS=11`
- `MLP_MULT=3.5`
- `MLP_NEGATIVE_SLOPE=0.5`
- `SHARED_MLP_GROUPS=2-3,4-5,6-7`
- `BIGRAM_VOCAB_SIZE=3072`
- `XSA_LAST_N=4`
- EMA shadow averaging is always applied before export
- `SWA_ENABLED=1`
- `ROPE_DIMS=16`
- `LN_SCALE=1`
- `VE_ENABLED=1`
- `LATE_QAT_THRESHOLD=0.0`

To recover a no-sharing baseline inside this file, set:

```bash
SHARED_MLP_GROUPS='' MLP_MULT=3.0 MLP_NEGATIVE_SLOPE=0 BIGRAM_VOCAB_SIZE=2048
```

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall candidates/202604141840_sandwich-shared-mlp/train_gpt.py` | Passed |
| CPU import + tiny forward smoke in an isolated venv | Passed: `loss_shape ()`, `logits_shape (2, 16, 128)`, shared MLP map `[(2,3), (4,5), (6,7)]` |

Notes:

- I did **not** run a full training job here because this environment does not provide the challenge’s CUDA/8xH100 runtime or dataset setup.
- The candidate script now supports a **CPU-only import/forward smoke path** via the SDPA fallback, which was enough to verify the shared-MLP plumbing without GPU infrastructure.

## Expected risks / tradeoffs

- **More MLP FLOPs per step:** widening to 3.5x may cost enough throughput to erase part of the capacity gain.
- **Less middle-layer specialization:** tying only the MLPs is safer than whole-block tying, but it can still overshare.
- **Quantization behavior is uncertain:** the stored model is smaller in parameters, but the wider activations may shift GPTQ-lite clipping optima.
- **Group placement may matter:** `2-3,4-5,6-7` is a research-guided default, not a proven optimum; later work should sweep both placement and group size.
