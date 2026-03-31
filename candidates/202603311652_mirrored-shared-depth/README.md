# Mirrored Shared-Depth GPT

## Hypothesis

A better next move for this repository is not another tiny local tweak on the existing 11-layer stack, but a more structural parameter-allocation change: **share the heavy attention+MLP transformer cores across mirrored encoder/decoder depths while keeping small per-layer control tensors distinct**.

The candidate keeps the strong `2026-03-22` recipe (11 logical layers, GPTQ-lite export, EMA/SWA, partial RoPE, LN scaling, XSA on deep layers, VE, BigramHash, Muon/AdamW) but replaces 11 independent block cores with **6 mirrored shared cores** scheduled as `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]`.

That should reduce artifact bytes spent on repeated heavy matrices while preserving layer-specific behavior through separate norms, residual mixing, scale vectors, skip weights, VE scales, and deep-layer XSA placement. The saved bytes are partially reinvested in a larger `BIGRAM_VOCAB_SIZE=3072` default.

## Why this is promising here

Repository evidence suggests the current frontier is already saturated on many small, local improvements:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is already a dense stack of proven wins.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` pushes even further with eval-time TTT.
- Prior records already cover XSA, partial RoPE, LN scale, GPTQ-lite, EMA/SWA, BigramHash, SmearGate, mixed int6/int8 export, and several optimizer sweeps.

At the same time, the repository review shows that **naive recurrence/depth reuse previously underperformed**, so the right next attempt is not a single repeated block with no flexibility. This candidate therefore uses a *relaxed* sharing scheme:

- heavy attention/MLP weights are shared,
- but per-layer control surfaces remain unique,
- and sharing follows the existing encoder/decoder mirror structure instead of flattening the whole model into one repeated block.

## Prior repository work that influenced this candidate

### Base implementation

This candidate is directly based on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`

That base already combines:

- 11 logical layers,
- 3x MLP,
- BigramHash,
- XSA on the deepest layers,
- partial RoPE,
- LN scale,
- VE on selected layers,
- EMA + tight SWA,
- GPTQ-lite style int6 export.

### Other records that shaped the design

- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md`
  - motivated reinvesting saved bytes into a larger bigram table.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - confirmed the strength of partial RoPE + LN scaling, and also reinforced that this candidate should spend its novelty budget on the shared-depth idea rather than another fragile QAT variant.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
  - noted earlier recurrence/depth-reuse misses, which is why this candidate uses *mirrored shared cores with per-layer adapters* instead of a fully naive single-block loop.

## External research that informed it

This candidate is mainly motivated by recent work arguing that parameter sharing can be substantially better than the older “tie everything identically” baseline when the model keeps some depth-specific flexibility:

- **Relaxed Recursive Transformers** (`arXiv:2410.20672`) argues that layer tying can recover much more quality when recurrence is relaxed with small depth-specific adaptation, rather than using a rigid repeated block.
- **ALBERT** (`arXiv:1909.11942`) is a classic result showing cross-layer parameter sharing can reduce model size substantially while keeping good downstream performance.
- **RecurrentGemma / Griffin** (`arXiv:2404.07839`) is another recent signal that recurrence/state reuse can be competitive when the architecture is designed around it rather than bolted on naively.

This candidate does **not** implement the full low-rank relaxed-recursion machinery from those papers. Instead, it adapts the core idea to this repository’s constraints by using existing per-layer control tensors as the “relaxation” mechanism.

## What changed versus the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. The model now separates each logical layer into:
   - a **shared core** containing only the heavy `attn` + `mlp` weights,
   - and a **layer-local adapter** containing norms, residual mix, scale vectors, and optional DTG gate.

2. Instead of 11 unique block cores, the script builds **6 shared cores** and reuses them across 11 logical layers using the mirrored schedule:
   - `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]`

3. XSA remains a **logical-depth** feature, so it is still only applied on the last `XSA_LAST_N` logical layers even when those layers reuse a shared core.

4. Partial RoPE configuration now applies to the shared cores instead of per-layer duplicated attention modules.

5. The default `BIGRAM_VOCAB_SIZE` is increased from `2048` to `3072` to use some of the recovered parameter budget.

6. The script now resolves default dataset/tokenizer paths relative to the repository root via `__file__`, so it can be run directly from the candidate directory as requested.

## How to run

From the candidate directory:

```bash
cd candidates/202603311652_mirrored-shared-depth
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key defaults in this candidate:

- `NUM_LAYERS=11`
- `NUM_SHARED_CORES=6`
- `BIGRAM_VOCAB_SIZE=3072`
- `XSA_LAST_N=4`
- `ROPE_DIMS=16`
- `LN_SCALE=1`
- `WARMDOWN_ITERS=3500`
- `EVAL_STRIDE=64`

The script keeps the same evaluation/export path style as the `2026-03-22` base, including GPTQ-lite-style int6 export and sliding-window evaluation.

## How to evaluate or ablate

The first ablations I would run are:

```bash
# less sharing, safer quality
NUM_SHARED_CORES=7 torchrun --standalone --nproc_per_node=8 train_gpt.py

# more aggressive sharing needs fewer logical layers with this mirror schedule
NUM_LAYERS=9 NUM_SHARED_CORES=5 torchrun --standalone --nproc_per_node=8 train_gpt.py

# isolate whether the larger bigram table helps
BIGRAM_VOCAB_SIZE=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If mirrored sharing looks neutral-to-positive, the next high-value follow-up would be layering in one additional proven repo trick, such as `LeakyReLU^2`, rather than changing many axes at once here.

## Main risks and tradeoffs

- **Quality risk:** even relaxed sharing may still underfit compared with 11 unique blocks if the shared cores become a bottleneck.
- **Interaction risk:** deep-layer XSA and VE may behave differently when the same core is reused at both shallow and deep logical positions.
- **Budget-allocation risk:** the extra bigram capacity may or may not be the best place to spend the saved bytes.
- **Optimization risk:** mirrored sharing changes gradient aggregation onto the reused cores, which could require retuning `MATRIX_LR`, `MUON_MOMENTUM`, or warmdown length.

## Validation

### Commands run

```bash
python -m compileall candidates/202603311652_mirrored-shared-depth/train_gpt.py
```

Attempted an extra import-level smoke check via `importlib`, but the current workflow environment does not have the repo’s runtime Python dependencies installed (`numpy` was missing before any model code ran), so that deeper check could not be completed locally here.

### Outcomes

- `python -m compileall .../train_gpt.py` **passed**.
- Import-level smoke test **blocked by missing runtime dependency in the workflow environment**.
- A true CPU forward-pass smoke test was **not feasible** here because this competitive script depends on CUDA + FlashAttention and does not provide a CPU fallback path.
