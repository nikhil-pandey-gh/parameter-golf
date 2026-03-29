# Mirror-shared core + fp16 tied embedding export

## Hypothesis

The strongest next step for this repository is **partial cross-layer parameter sharing without extra recurrent passes**.

Instead of looping the same layers more times (which prior records show is too slow under a 10-minute wallclock), this candidate keeps the same 11 forward passes but **shares the large attention and MLP cores across mirrored U-Net depths** using the default layout:

```text
0,1,2,3,4,5,4,3,2,1,0
```

Per-layer RMSNorms, residual mixing, skip weights, attention/MLP scales, and `q_gain` stay untied, so the model keeps layer-specific specialization while sharply reducing the number of unique large matrices that must be serialized.

The saved artifact budget is then spent on a safer deployment path: the candidate keeps the **tied token embedding in fp16 during export**, because earlier records showed that this tensor is unusually quantization-sensitive.

## Why this is promising for this repository

This repository's best runs already show a clear pattern:

- strong scores come from **compression-aware training and export**, not just lower pre-quant loss,
- 11-layer U-Net-style stacks are strong,
- small zero-parameter or near-zero-parameter refinements stack well,
- naive depth recurrence is a bad fit for the 10-minute budget because it reduces optimizer steps.

This candidate targets the open space between those findings:

- it borrows the strong **11-layer / XSA / partial-RoPE / LN-scale / EMA / GPTQ-lite** stack,
- it adds **ALBERT-style sharing** of the expensive block cores,
- it does **not** add extra forward passes,
- it uses **alias-aware export** so shared tensors are serialized once instead of paying for repeated keys,
- it uses the newly recovered headroom to keep the tied embedding in fp16.

In short: same depth, same general training recipe, fewer unique artifact bytes, and more precision where prior records found it matters most.

## Prior records and experiments that influenced this candidate

Primary repository influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the base implementation because it is the strongest clean pre-TTT stack in the repo and already contains GPTQ-lite, EMA, BigramHash, VE, XSA, and partial-RoPE-era improvements.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the low-risk **LeakyReLU(0.5)^2** MLP activation change.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - reinforced the value of **partial RoPE** and **layerwise LN scaling**.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - showed that the tied embedding is disproportionately sensitive to quantization and can justify fp16 passthrough.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative result: **layer recurrence x2** hurt badly because it consumed the wallclock budget with fewer optimizer steps.

## External research that informed this candidate

- **ALBERT** (Lan et al., arXiv:1909.11942): strong evidence that cross-layer parameter sharing can preserve performance while cutting parameter count, especially when small layer-specific parameters remain untied.
- **Universal Transformer** (Dehghani et al., arXiv:1807.03819): useful conceptual support for reusing transformation blocks across depth.

Other researched ideas included pruning-aware training, direct low-bit training, and rotation-aware quantization, but those looked more invasive for this repository's current code path. Partial block sharing was the strongest idea that fit the existing script style with minimal infrastructure change.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate changes the base in four main ways:

1. **Mirror-shared block cores**
   - Large attention and MLP weights are shared across mirrored depths by default.
   - Layer-specific norms, gates, scales, `q_gain`, and skip connections remain unique.

2. **Alias-aware export**
   - The quantization/export path detects shared tensors and stores them once with alias metadata.
   - This is important because raw `state_dict()` export can otherwise duplicate shared modules under multiple names.

3. **fp16 tied embedding passthrough during mixed quantization**
   - `tok_emb.weight` is kept in fp16 by default via `INT6_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight`.

4. **LeakyReLU(0.5)^2 MLP**
   - The ReLU^2 MLP is replaced with the LeakyReLU(0.5)^2 variant used successfully in later repo records.

The candidate also adds an explicit FlashAttention import fallback to `scaled_dot_product_attention`, so the script is easier to import in environments that do not have `flash_attn_interface` installed.

## How to run

From the repository root:

```bash
cd /path/to/parameter-golf

RUN_ID=mirror_share_fp16 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SHARE_MIRROR=1 \
INT6_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight \
MLP_NEGATIVE_SLOPE=0.5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 candidates/202603291326_mirror-share-fp16/train_gpt.py
```

Optional ablations:

- disable sharing entirely with `SHARE_MIRROR=0`,
- override the sharing map with `SHARED_LAYOUT=0,1,2,3,4,5,4,3,2,1,0`,
- change fp16 passthrough patterns with `INT6_KEEP_FLOAT_FP16_NAME_PATTERNS=...`.

## Validation

Validation run in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603291326_mirror-share-fp16/train_gpt.py
python -m py_compile candidates/202603291326_mirror-share-fp16/train_gpt.py
```

Outcome:

- syntax checks passed,
- no repository files outside this candidate directory were modified,
- a CPU import/forward smoke test was **not feasible in this environment** because the available Python interpreter did not have `torch`, `numpy`, or `sentencepiece` installed.

## Main expected risks and tradeoffs

- **Too much sharing may reduce depth specialization.** Keeping per-layer norms/scales/gains untied is meant to mitigate this, but it may still cost quality.
- **Artifact savings do not guarantee better BPB.** The point of sharing here is to reallocate bytes to more important tensors, not to shrink size for its own sake.
- **Alias-aware export adds serialization complexity.** If there is a bug in shared-tensor detection, the artifact could undercount or fail to round-trip correctly.
- **LeakyReLU^2 and sharing may interact in unexpected ways.** This candidate deliberately combines a known low-risk activation win with a new architectural change, so attribution will need follow-up ablations.

## Suggested next experiments

1. Compare `SHARE_MIRROR=1` vs `SHARE_MIRROR=0` on the exact same 03-22 stack.
2. Keep only the mirrored **MLP** shared while leaving attention unique.
3. Spend the recovered artifact bytes on `ve_shared.embed.weight` fp16 passthrough too.
4. Try a less aggressive shared layout such as `0,1,2,3,4,5,6,4,3,2,1` or only sharing the deepest half.
