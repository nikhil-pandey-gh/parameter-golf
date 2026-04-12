# Positional-Head Partial RoPE

## Hypothesis

The repo already shows that **uniform Partial RoPE** helps the strong 11-layer stack versus full-head/full-dim rotary application. This candidate tests a more targeted version: allocate RoPE to only a subset of KV groups and aligned query heads in early layers, then ramp both the number of positional heads and the rotated dimensions as depth increases.

The expected benefit is better use of a fixed parameter/compression budget: early layers can stay more content-heavy while later layers spend more of their attention bandwidth on durable positional structure.

## Why it is promising for this repository

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` showed that **uniform Partial RoPE (16/64 dims)** was already a real improvement over the prior 11-layer XSA/EMA stack.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120` and `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that the current frontier rewards **late-layer context modeling**, especially in the deepest XSA layers.
- Hong et al., **“On the token distance modeling ability of higher RoPE attention dimension”** (arXiv:2410.08703), identifies **Positional Heads** and argues that long-range modeling is concentrated in a subset of heads and higher-dimensional rotary allocations.
- Zhai et al., **“Exclusive Self Attention”** (arXiv:2603.09078), shows that late-layer context modeling benefits from reducing self-copying pressure, which fits this repo’s XSA-heavy deep-stack trend.

## Prior records that influenced this candidate

| Experiment | Influence |
| --- | --- |
| `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` | Chosen base implementation and training/export recipe |
| `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` | Established that Partial RoPE + LN scale is worth keeping |
| `2026-03-20_11L_EfficientPartialXSA_FA3_SWA120` | Motivated focusing positional changes on the deepest context-heavy layers |
| `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` | Showed the current frontier is squeezing small gains from targeted, low-overhead changes rather than broad rewrites |

There were **no prior `candidates/` experiments** in the repository when this candidate was created.

## External research that informed it

1. Hong et al., **On the token distance modeling ability of higher RoPE attention dimension** — arXiv:2410.08703  
   <https://arxiv.org/abs/2410.08703>
2. Zhai et al., **Exclusive Self Attention** — arXiv:2603.09078  
   <https://arxiv.org/abs/2603.09078>

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. Added a **layer-wise RoPE layout** builder that can allocate:
   - how many KV groups receive rotary position encoding, and
   - how many dimensions per rotated head are rotary.
2. Applied RoPE only to the configured KV prefix and its aligned query-head groups instead of rotating all heads uniformly.
3. Added env overrides for quick ablations:
   - `ROPE_DIMS`
   - `ROPE_KV_HEADS`
   - `ROPE_DIM_SCHEDULE`
   - `ROPE_KV_HEAD_SCHEDULE`
4. Logged the resolved per-layer layout so runs can be compared directly from the training log.

Everything else is intentionally kept close to the base record:

- 11 layers, 512 width, 8 heads / 4 KV heads
- XSA on the last 4 layers
- 3x MLP, SmearGate, BigramHash, VE128, LN scale
- EMA, warmdown 3500, late QAT threshold 0.15
- GPTQ-lite int6 + zstd-22 export

## Default positional-head layout

With the candidate defaults, the resolved 11-layer layout is:

| Layer(s) | Rotated query heads | Rotated KV heads | RoPE dims per rotated head |
| --- | --- | --- | --- |
| 0-2 | 2 | 1 | 8 |
| 3-4 | 4 | 2 | 12 |
| 5-6 | 4 | 2 | 16 |
| 7-8 | 6 | 3 | 24 |
| 9 | 8 | 4 | 24 |
| 10 | 8 | 4 | 32 |

That is the main experimental twist in this candidate.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604120100_positional-head-rope
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate script resolves its default dataset and tokenizer paths relative to the repository root, so the command above works from inside the candidate directory without extra path flags.

To recover a uniform Partial RoPE baseline inside this candidate for ablations:

```bash
cd candidates/202604120100_positional-head-rope
ROPE_DIMS=16 ROPE_KV_HEADS=4 ROPE_DIM_SCHEDULE=16 ROPE_KV_HEAD_SCHEDULE=4 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- The schedule is tuned to the repo’s **11-layer / seq2048 / sliding-eval** regime and may not transfer cleanly to other shapes.
- Over-allocating positional heads late in the stack could make the model too position-dominant and wash out some of the content-heavy gains from XSA, VE, SmearGate, and BigramHash.
- This is still a hand-designed schedule, so the main failure mode is simply that the current uniform `ROPE_DIMS=16` setting was already close to optimal for this budget.

## Validation

Commands run from the repository root:

1. `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604120100_positional-head-rope/train_gpt.py`  
   Outcome: **succeeded**.
2. Minimal CPU smoke import/forward test with a local FlashAttention stub.  
   Outcome: **not feasible in this runner** because the environment did not have the required runtime dependencies installed (`torch`, `numpy`, and `sentencepiece` were all missing), so the script could not be imported for execution without first provisioning the full ML stack.
3. Dependency-free AST extraction of the new schedule helpers plus direct `_build_rope_layout()` assertions.  
   Outcome: **succeeded**; confirmed the default schedule resolves to `(q2/k1/d8) ... (q8/k4/d32)` and that `ROPE_KV_HEADS` / `ROPE_DIMS` now work as independent override knobs.
