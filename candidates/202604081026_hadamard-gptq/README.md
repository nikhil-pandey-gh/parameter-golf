# Hadamard-GPTQ-lite export on the 2026-03-22 stack

## Hypothesis

The strongest open gap in this repository is still **post-training quantization quality under the 16MB artifact cap**. The best non-TTT stack already reaches strong pre-export quality, so a small export-time change that reduces outlier-driven int6 error may buy BPB without perturbing the proven training recipe.

This candidate tests a **QuaRot/SpinQuant-inspired fixed Hadamard preconditioner** for int6 block weights: rotate each row's input features in 512-wide blocks before GPTQ-lite percentile search, quantize in that rotated basis, then apply the inverse rotation after dequantization for evaluation. The goal is to make the existing per-row int6 search see flatter, easier-to-clip weight distributions without paying extra artifact bytes for learned rotations.

## Why this looks promising here

- The local record review shows that the biggest durable wins after the baseline came from **compression-aware design**, especially int6/int5 export, EMA/SWA, and small architecture tweaks on top of a stable 11-layer stack.
- The review also shows that the current best simple extension point is **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`**, because it already has the strong 11L + XSA4 + partial RoPE + LN scale + VE128 + GPTQ-lite recipe without the extra complexity of the later score-first TTT path.
- External research points at **rotation-aided low-bit export** as the closest high-upside idea to this codebase:
  - **GPTQ**: https://arxiv.org/abs/2210.17323
  - **QuaRot**: https://arxiv.org/abs/2404.00456
  - **SpinQuant**: https://arxiv.org/abs/2405.16406

The current stack already does percentile search over per-row int6 scales. Replacing "search on raw weights" with "search on a fixed rotated basis" is a small enough code change to be realistic for this repository, while directly targeting the same export bottleneck those papers attack.

## Prior work that influenced this candidate

### Records

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Architecture context:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Why not the 2026-03-23 stack as the base:** the record review found that most of the later gain came from **TTT + LeakyReLU^2**, while parameter banking itself was mostly neutral on BPB, so the simpler 2026-03-22 code is a cleaner place to isolate a quantization export change.

### Prior candidates

None existed in `candidates/` when this candidate was created.

## What changed versus the chosen base

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Hadamard-preconditioned int6 export**
   - Added a fast Walsh-Hadamard transform on the input dimension.
   - For int6 attention/MLP weights whose input dimension is divisible by `HADAMARD_BLOCK_SIZE` (default `512`), the export path now rotates columns before GPTQ-lite percentile search and stores a tiny metadata tag so the inverse rotation is applied after dequantization.
2. **FlashAttention fallback**
   - If `flash_attn_interface` is unavailable, attention falls back to `torch.nn.functional.scaled_dot_product_attention` rather than failing at import time.
3. **Optional activation-aware ablation hook**
   - The candidate also includes an **off-by-default** activation-moment collector (`ACTIVATION_AWARE_QUANT=1`) for future AWQ-style ablations, but it is only meaningful for the **unrotated** export path (`HADAMARD_ROTATE_INT6=0`).

Everything else intentionally stays close to the 2026-03-22 recipe: 11 layers, MLP 3x, XSA on the final 4 layers, partial RoPE, LN scaling, VE128, EMA, tight SWA, late-QAT thresholding, and sliding-window eval.

## How to run

From this candidate directory:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

By default, the candidate resolves `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so it can be launched from inside this candidate directory without rewriting those paths.

Useful knobs for this candidate:

```bash
HADAMARD_ROTATE_INT6=1
HADAMARD_BLOCK_SIZE=512
ACTIVATION_AWARE_QUANT=0
```

To compare against the unrotated export inside the same script:

```bash
HADAMARD_ROTATE_INT6=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To try the optional activation-aware ablation on top of the same code:

```bash
HADAMARD_ROTATE_INT6=0 ACTIVATION_AWARE_QUANT=1 QUANT_CALIB_BATCHES=4 QUANT_CALIB_BATCH_TOKENS=32768 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Risks and tradeoffs

- **Main risk:** a fixed Hadamard rotation may help less than the learned rotations in SpinQuant, so gains could be modest or inconsistent.
- **Export-only idea:** this candidate does not change training dynamics much, so upside is bounded by the current quantization gap.
- **Possible interaction with per-row clipping:** a better basis can still pick a worse percentile if the clip grid is too coarse.
- **Optional activation-aware path is not the primary candidate:** it is included for follow-up ablations, not as the claimed main idea.

## Validation

### Completed here

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604081026_hadamard-gptq/train_gpt.py
```

**Outcome:** passed.

### Not completed here

A minimal CPU end-to-end smoke run was **not feasible in this checkout** because:

1. this candidate intentionally stays on the CUDA-tuned 2026-03-22 training path (`torch.cuda` synchronization/timing, fused optimizers, and multi-GPU launch assumptions), and
2. the repository checkout does not include local FineWeb shard files or the SentencePiece model needed for a realistic start-up run.

The script is therefore validated here at the syntax level, with the export/eval path documented for a proper GPU run.
