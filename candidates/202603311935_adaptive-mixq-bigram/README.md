# Adaptive MixQ + Bigger BigramHash

## Hypothesis

The strongest low-friction next step for this repo is to keep the best current 11-layer training stack mostly intact, then trade export precision **more selectively** so the artifact budget can fund a larger lexical memory.

Concretely: use the 2026-03-23 record trainer as the base, increase `BIGRAM_VOCAB_SIZE` from `2048` to `3072`, and replace the fixed mixed export with a **sensitivity-aware adaptive mixed quantizer**:

- MLP matrices try `int5` vs `int6`
- attention matrices try `int6` vs `int8`
- the large `bigram.embed.weight` and `ve_shared.embed.weight` tables also try `int6` vs `int8`
- the exporter starts from the **cheapest proxy-size** choice, then greedily buys back precision only for the tensors with the best reconstruction-error-per-byte tradeoff
- only a fraction of the saved bytes are reinvested (`MIXQ_REINVEST_FRAC`, default `0.35`); the rest is left as artifact headroom for the bigger BigramHash table

The goal is to preserve most of the strong 11-layer stack while spending compression budget where it matters most.

## Why this is promising for this repository

The record history points in the same direction from two sides:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed that **better post-training quantization alone** still moved the needle on the best 11-layer stack.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` showed that **lower-bit MLP export can buy enough bytes to afford bigger lexical capacity**, and that larger `BigramHash` tables remained useful.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` showed the current strongest overall stack in this repo, but it still used a mostly fixed export recipe.

That combination suggests a natural next candidate: keep the strongest trainer, keep legal TTT optional, and use a smarter exporter to pay for a modestly larger hashed lexical table.

## Prior repo work that influenced this candidate

This candidate is primarily built on:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` (mainly as a warning that one earlier late-QAT path was constant-folded away under `torch.compile`)

There were **no populated prior experiments under `candidates/`** when this candidate was created.

## External research that informed it

This candidate is deliberately closer to the repo’s existing code than a full new training paradigm, but it is informed by the mixed-precision / sensitivity-aware quantization literature:

- **LSQ** — Learned Step Size Quantization argues that low-bit performance depends heavily on step-size selection rather than just nominal bit width. The practical takeaway here is to avoid a one-size-fits-all export policy. Paper: <https://arxiv.org/abs/1902.08153>
- **HAWQ / HAWQ-V2** — Hessian-aware work argues that different layers deserve different precisions because they do not share the same loss sensitivity. This candidate uses a simple reconstruction-error-per-byte proxy instead of second-order machinery, but the design goal is the same. Papers: <https://arxiv.org/abs/1905.03696>, <https://arxiv.org/abs/1911.03852>
- **GPTQ** — showed that careful post-training quantization can preserve quality much better than naive clipping. This candidate reuses the repo’s GPTQ-lite-style per-row clip search as its low-bit building block. Paper: <https://arxiv.org/abs/2210.17323>
- **AWQ** — highlighted the value of selectively protecting the most sensitive channels/weights instead of treating every tensor uniformly. This candidate applies the same spirit at the tensor level. Paper: <https://arxiv.org/abs/2306.00978>

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. `BIGRAM_VOCAB_SIZE` default is now `3072` instead of `2048`.
2. The fixed `mixed_quantize_int6(...)` export path was replaced with `adaptive_mixed_quantize(...)`.
3. The new exporter:
   - keeps the repo’s existing per-row clip search idea,
   - evaluates multiple bit-width candidates per large tensor,
   - estimates a cheap compressed-byte proxy per candidate,
   - greedily upgrades only the tensors with the best error-reduction-per-byte ratio,
   - logs a summary of the selected `int5` / `int6` / `int8` mix.
4. Export artifacts are written as `final_model.mixq.ptz`, and the log labels were updated to `final_mixq_*` while keeping the legacy `final_int8_zlib_roundtrip_exact` alias on sliding-window exact metrics for compatibility.
5. The inherited late-QAT path is left **disabled by default** (`LATE_QAT_THRESHOLD=0.0`) so this candidate isolates the adaptive exporter instead of depending on the previously fragile compiled-QAT behavior.

Everything else stays intentionally close to the 2026-03-23 stack: 11 layers, LeakyReLU² MLPs, XSA on the deepest layers, partial RoPE, LN scaling, VE, EMA/SWA, parameter banking, and optional legal score-first TTT.

## How to run / evaluate

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.0 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MIXQ_REINVEST_FRAC=0.35 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `TTT_ENABLED=1` keeps the strongest record’s evaluation path. Set `TTT_ENABLED=0` if you want to isolate the new export idea first.
- `MIXQ_REINVEST_FRAC` controls how aggressively the exporter buys precision back after starting from the lowest-bit candidates.

## Main expected risks / tradeoffs

- The exporter uses a **proxy compressed-byte estimate per tensor**, not the exact whole-artifact lzma objective. It should be directionally useful, but it is still a heuristic.
- Moving more MLP tensors to `int5` could create a larger training/export mismatch than the current fixed-int6 stack.
- This candidate does **not** attempt to resurrect full late-QAT in the compiled training graph. That was a conscious choice to avoid repeating the earlier constant-folding failure mode without a proper GPU validation cycle.
- `BIGRAM_VOCAB_SIZE=3072` is a moderate guess, not a proven optimum for this exact stack.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202603311935_adaptive-mixq-bigram/train_gpt.py
python - <<'PY'
import importlib.util
print('torch_spec', bool(importlib.util.find_spec('torch')))
print('flash_attn_interface_spec', bool(importlib.util.find_spec('flash_attn_interface')))
PY
```

Observed outcomes:

- `python -m compileall ...` succeeded.
- Dependency probe reported `torch_spec False` and `flash_attn_interface_spec False`.
- Because this environment does not currently have `torch` or the Hopper-specific `flash_attn_interface` module installed, a truthful runtime smoke test was **not feasible here**.
