# SpinQuant-lite Int4 Export + Real Late-QAT Tail

## Hypothesis

The strongest non-TTT stack in this repo is already very good *before* evaluation tricks, but it is still artifact-bound: recent 11-layer runs cluster around `15.5-15.9MB` while relying on mixed int6/int8 export. This candidate tests whether a **rotation-aided groupwise int4 export** can shrink the dominant MLP/attention matrices enough to preserve or improve post-quantized `val_bpb`, while spending some of the recovered budget on **fp16 passthrough for quantization-sensitive embedding tables**.

The candidate also fixes a repo-local weakness from the prior late-QAT branch: instead of flipping a class flag after `torch.compile(fullgraph=True)` and risking dead code, it **recompiles once when the late-QAT threshold is crossed**, so the fake-quant branch actually becomes active.

## Why this is promising here

- The repo's recent wins were mostly about **compression-aware modeling**, not just lower pre-quant loss:
  - `2026-03-18_FP16Embed_WD3600` showed that embedding precision can dominate the quantization gap.
  - `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` and `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that small export improvements stack cleanly on the strong 11L core.
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly documented that its late-QAT flag was compile-folded away, so there is still unfinished work on making low-bit tails real.
- External research points in the same direction:
  - **SpinQuant** argues that rotations reduce outliers and make aggressive low-bit export more viable: https://arxiv.org/abs/2405.16406
  - **AQLM** is a stronger but heavier codebook-based version of the same general idea: https://arxiv.org/abs/2401.06118
  - The current repo already benefits from export-side search via **GPTQ-lite** and from context-side architectural improvements via **XSA**: https://arxiv.org/abs/2603.09078

## Chosen base implementation

This forks the strongest clean training-only base in the repo:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

That base already carries the durable late-stack features:

- 11 layers, 512d, 3x MLP
- XSA on the last 4 layers
- partial RoPE (`16/64`) + LN scale
- EMA + warmdown 3500
- SmearGate + BigramHash + VE128
- sliding-window eval

There were **no prior `candidates/` runs** in this repository when this candidate was created.

## What changed vs the base

1. **SpinQuant-lite export for big 2D matrices**
   - attention and MLP matrices now export with **groupwise signed int4**
   - export searches over `group_size in {64, 128}` and clip percentiles in `0.97, 0.99, 0.995, 0.999, 1.0`
   - values are nibble-packed before zstd/zlib compression

2. **Fixed Hadamard rotation before int4 quantization**
   - each group is transformed with a normalized Walsh-Hadamard transform before quantization
   - dequantization applies the same transform again, since the normalized transform is self-inverse
   - this is deliberately the cheap, no-extra-parameter version rather than a learned rotation

3. **Recovered bytes are spent on sensitive tables**
   - `tok_emb.weight`
   - `bigram.embed.weight`
   - `ve_shared.embed.weight`
   stay in `fp16` passthrough instead of being pushed down with the block weights

4. **Late QAT now actually activates**
   - when `lr_scale < LATE_QAT_THRESHOLD`, the script enables the fake-quant branch and recompiles the model once
   - this avoids the earlier "toggle a Python class bool after compile" failure mode noted in the `2026-03-21` record

5. **Metrics/logging switched to the new export path**
   - roundtrip and sliding-window logs now report `int4` artifacts and scores instead of `int6`

## How to run

Defaults are baked into `train_gpt.py`, and the default dataset/tokenizer paths are resolved from the repository root via `__file__`, so the candidate can be launched directly from inside its own directory:

```bash
cd candidates/202604020148_spinquant-lite-int4
RUN_ID=spinquant_lite_int4 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

From the repository root, this equivalent command also works:

```bash
RUN_ID=spinquant_lite_int4 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202604020148_spinquant-lite-int4/train_gpt.py
```

Useful knobs for this candidate:

```bash
INT4_GROUP_SIZES=64,128
INT4_CLIP_PERCENTILES=0.97,0.99,0.995,0.999,1.0
INT4_ROTATE=1
INT4_KEEP_FLOAT_FP16_NAME_PATTERNS=tok_emb.weight,bigram.embed.weight,ve_shared.embed.weight
INT4_QAT_GROUP_SIZE=64
INT4_QAT_CLIP_PERCENTILE=0.995
LATE_QAT_THRESHOLD=0.15
```

## Validation

### Commands run

```bash
python -m compileall candidates/202604020148_spinquant-lite-int4/train_gpt.py
```

### Outcomes

- `python -m compileall ...` **passed**
- A local CPU import/forward smoke check was **not feasible in this runner**:
  - the runner's default Python environment does not have `torch` installed
  - the real training path also requires CUDA plus `flash_attn_interface`

## Main risks and tradeoffs

- **Int4 may still be too aggressive** for some late-layer matrices even with fixed rotations.
- **Fixed Hadamard rotation is weaker than learned rotations** from full SpinQuant-style methods; it is chosen here because it fits the repo's "single self-contained script" constraint.
- **One late recompilation** adds complexity and could cost a few seconds near the end of training.
- **MSE-optimal export settings are not guaranteed to be bpb-optimal**; group-size and clip-search choices may still need empirical tuning.
- **Recovered byte budget is only partly spent** in this candidate. If the export path works, the obvious next follow-up is to cash in more of that headroom on width, depth, or selective higher-precision exemptions.
