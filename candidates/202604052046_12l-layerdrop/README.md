# 12L LayerDrop + LeakyReLU^2

## Hypothesis

This candidate targets a gap in the repo's search space: **deeper unique depth without paying the full training-time cost of that extra depth on every step**.

Across the existing records, better training-side results kept coming from longer context, more layers, and wider MLPs. But the repo also has two clear warnings about naive depth reuse:

- the `2026-03-18_FP16Embed_WD3600` notes call depth recurrence promising but too step-hungry for the 10-minute budget, and
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` explicitly reports a bad result for layer recurrence x2 because it gave up too many optimization steps.

The hypothesis here is that a **12-layer** variant can still be worthwhile if training uses **LayerDrop / stochastic depth** so the deepest blocks are skipped often enough to keep average training cost close to the successful 11-layer stacks, while evaluation still uses the full 12-layer network.

## Why this is promising for this repository

### What the repo history suggests

The record review pointed to a few durable patterns:

- **More effective depth helps**: 10-layer and 11-layer runs consistently beat the 9-layer baseline once compression/export was handled well.
- **Longer context helps**: the seq-2048 and seq-4096 runs were among the strongest pure training-side improvements.
- **The current best reusable stack is the 11-layer XSA/EMA/partial-RoPE line**:
  - `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- **LeakyReLU(0.5)^2 is a strong low-cost win** from `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`.
- **Explicit recurrent/shared-depth attempts were negative** under this challenge's short wall-clock budget.

There were **no prior `candidates/` folders** in the repo, so this candidate is only avoiding overlap with `records/`.

### What external research suggests

The candidate is grounded in three primary sources:

1. **Stochastic Depth** ([Sun et al., 2016](https://arxiv.org/abs/1603.09382)): train short networks and test deep by randomly dropping residual blocks during training.
2. **LayerDrop** ([Fan et al., 2019](https://arxiv.org/abs/1909.11556)): structured layer dropping works for transformers and can preserve useful subnetworks of different depths.
3. **CaiT / LayerScale** ([Touvron et al., 2021](https://arxiv.org/abs/2103.17239)): deeper transformers benefit from explicit depth-stabilizing scale choices. This repo already moved in that direction via layerwise normalization scaling, so LayerDrop is a natural complement rather than a full architectural reset.

I also considered stronger weight-sharing / recurrent-block ideas from ALBERT, Universal Transformer, and recent recurrent latent-depth work, but the repo's own negative recurrence results made **train-short/test-deep** the better fit than **shared-depth reuse** for this challenge.

## Chosen base implementation

This candidate starts from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

That base was the best non-TTT 11-layer stack with:

- seq 2048 / batch 786,432
- efficient late-layer XSA
- partial RoPE (16/64)
- layerwise LN scaling
- shared value embeddings on the deepest layers
- EMA + tight SWA
- GPTQ-lite mixed int6/int8 export

## What changed versus the base

The implementation keeps the 03-22 stack mostly intact and makes a single coherent depth-oriented change set:

1. **Add a 12th transformer layer**
   - `NUM_LAYERS` default: `11 -> 12`

2. **Add LayerDrop / stochastic depth**
   - new env knobs:
     - `LAYERDROP_START` (default `0.0`)
     - `LAYERDROP_END` (default `0.18`)
     - `LAYERDROP_WARMUP_STEPS` (default `1000`)
   - the drop rate increases linearly with depth
   - when a layer is dropped during training, the whole block is skipped
   - evaluation still runs the full model depth

3. **Use LeakyReLU(0.5)^2**
   - new env knob: `MLP_LEAK=0.5`
   - replaces the base relu-squared MLP activation with the strongest simple MLP change seen in the records

4. **Trade a little width for the extra depth**
   - `MLP_MULT` default: `3.0 -> 2.75`
   - this keeps the candidate more realistic under the artifact cap after adding the extra layer

5. **Trim a little auxiliary capacity to pay for depth**
   - `BIGRAM_VOCAB_SIZE`: `2048 -> 1536`
   - `VE_DIM`: `128 -> 96`
   - `VE_LAYERS`: `9,10 -> 10,11`

6. **Make the script runnable from the candidate directory**
   - default dataset/tokenizer paths are resolved relative to the script location, not the current working directory

7. **Add a local smoke-friendly attention fallback**
   - `flash_attn_interface` is now optional
   - if FlashAttention 3 is unavailable, the script falls back to PyTorch SDPA
   - `main()` re-enables the SDPA backends when external FA3 is unavailable
   - this does not change the intended H100 path, but it makes local import/CPU smoke tests possible

## Expected tradeoff

The default shape is intentionally a trade:

- deeper model: **12 layers**
- slightly smaller FFN width: **2.75x instead of 3.0x**
- slightly smaller auxiliary token features

The default candidate instantiates at **27,676,260 parameters** on CPU, which is the depth/width point this candidate is betting on.

## How to run

From this directory:

```bash
cd candidates/202604052046_12l-layerdrop
RUN_ID=layerdrop_trial \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script will:

- train with the built-in wall-clock cap,
- apply EMA before export,
- emit the mixed int6 artifact,
- print standard roundtrip validation metrics, and
- print sliding-window validation metrics at the end.

Useful knobs to sweep first:

```bash
LAYERDROP_END=0.12
LAYERDROP_END=0.18
LAYERDROP_END=0.24
MLP_MULT=2.625
MLP_MULT=2.75
BIGRAM_VOCAB_SIZE=1536
BIGRAM_VOCAB_SIZE=2048
```

## Main risks / tradeoffs

1. **LayerDrop may be too aggressive** and undertrain the deepest layers, especially if `LAYERDROP_END` is pushed above ~0.2.
2. **The extra layer may still be too expensive** in artifact bytes even after shrinking MLP width and auxiliary tables; this needs a real GPU export run to confirm.
3. **Compiled training throughput could shift** because training now uses real block skipping and therefore cannot rely on the original fully static training graph.
4. **The CPU smoke path only validates logic**, not FlashAttention-3 throughput or actual 8xH100 timing.

## Validation

### Syntax check

```bash
python -m compileall candidates/202604052046_12l-layerdrop/train_gpt.py
```

Outcome: **passed**

### Tiny CPU smoke check

Because the workflow runner did not have PyTorch installed in the system interpreter, I created a temporary virtualenv under `/tmp/gh-aw/agent/` and ran a tiny import/forward/backward smoke test there, including a `torch.compile(..., fullgraph=False)` training-mode pass that matches the dynamic LayerDrop training path:

```bash
/tmp/gh-aw/agent/smoke-venv/bin/python - <<'PY'
# imports candidate script, builds a 4-layer toy GPT, runs
# eager forward/backward, a compiled training-mode pass
# (fullgraph=False), and a compiled eval pass (fullgraph=True)
PY
```

Outcome:

```python
{'train_loss': 4.847509, 'compiled_train_loss': 4.847509, 'compiled_eval_loss': 4.847509, 'params': 34547}
```

That smoke path exercises the new LayerDrop logic plus the SDPA fallback successfully.

## Code review notes

A focused review pass initially found two real issues, both now fixed:

1. LayerDrop originally masked residual updates **after** each block ran, which regularized depth but did not save compute. It now performs **whole-block skipping** in the layer loop during training.
2. The FA3 -> SDPA fallback originally left the SDPA backends disabled inside `main()`. The script now re-enables the PyTorch SDPA backends whenever external FlashAttention 3 is unavailable.
