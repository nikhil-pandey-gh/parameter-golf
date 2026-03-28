# Headwise RoPE Ramp on the PR #414 Stack

## Hypothesis

The repo already showed that **fixed partial RoPE** helps tiny models: moving from full 64-dim rotary to `ROPE_DIMS=16` improved the strong 11-layer XSA/EMA stack from **1.1271** to **1.1248**. My hypothesis is that this same "positional budget" idea should work better if it is allocated **unevenly across depth and heads** rather than applied identically everywhere.

This candidate keeps the strongest current training/eval stack intact, but replaces fixed per-layer RoPE usage with a **headwise depth ramp**:

- early layers rotate only a small subset of heads,
- deeper layers gradually add more rotary-enabled heads,
- `ROPE_DIMS` stays partial within each rotated head.

That preserves more position-agnostic capacity in shallow layers while still restoring strong positional structure deeper in the network.

## Why this is promising for this repository

This repository's best runs repeatedly benefited from **better allocation of a tight representational budget** rather than from broad architectural churn:

- sliding-window eval was a huge free win,
- XSA/EMA/quantization polish gave steady gains,
- fixed **partial RoPE** was explicitly worth another ~`-0.0023` BPB,
- the current SOTA still uses **uniform** partial RoPE (`16/64`) on every layer/head.

So the unexplored opportunity is not "more RoPE" but **smarter RoPE placement**. Under a 16MB artifact cap and a 10-minute training budget, a zero-parameter schedule change is attractive because it preserves the rest of the winning stack and does not require new infrastructure.

## Records and prior experiments that influenced this candidate

Primary implementation base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Most relevant prior evidence:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - showed fixed partial RoPE (`16/64`) beats full RoPE on the strong 11-layer stack.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - showed the post-PR-414 stack still had room from careful quantization/training polish.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - established the 11-layer XSA/EMA/MLP3x recipe that the later partial-RoPE run improved.

There were **no prior `candidates/` directories** in this repository when this candidate was created.

## External research that informed it

This candidate is grounded in three papers:

- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021, arXiv:2104.09864)
  - established RoPE as an efficient way to inject relative position information into attention.
- **Round and Round We Go! What makes Rotary Positional Encodings useful?** (Barbero et al., 2024/2025, arXiv:2410.06205)
  - argues RoPE's usefulness is not just monotonic distance decay; the paper finds models use different RoPE frequencies differently, with high frequencies helping positional patterns and low frequencies carrying more semantic load.
- **Context-aware Rotary Position Embedding** (Veisi et al., 2025, arXiv:2507.23083)
  - reports gains from **head-specific** RoPE behavior, suggesting that uniform positional treatment across heads is suboptimal.

I am not implementing a learned/context-conditioned rotary scheme here because that would add extra code, parameters, and tuning burden. Instead, this candidate takes the same research direction and compresses it into a **static, zero-parameter head schedule** that is more compatible with this challenge.

## What changed versus the chosen base implementation

Base: `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added **layerwise head scheduling** for RoPE.
   - New knobs:
     - `ROPE_HEADS_MIN` (default `2`)
     - `ROPE_HEADS_MAX` (default `8`)
   - With the 11-layer default stack, rotary-enabled query heads ramp from shallow to deep layers in **GQA-aligned groups**.
   - The script derives the schedule in KV-head groups first, then expands to the corresponding query-head blocks so query/KV rotation stays consistent under `8` query heads / `4` KV heads.

2. Added **head masks** so RoPE is applied only to selected heads.
   - Query heads and KV heads have separate masks because this stack uses GQA (`8` query heads, `4` KV heads).
   - The masks are snapped to shared KV groups instead of arbitrary head counts.

3. Kept **partial rotary dimensions** within each rotated head.
   - Default `ROPE_DIMS=16` is unchanged from the best partial-RoPE runs.

4. Made dataset/tokenizer defaults **script-relative**, so the candidate can be run directly from its own directory without depending on the caller's working directory.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603280658_headwise-rope-ramp

RUN_ID=headwise_rope_ramp \
ROPE_DIMS=16 ROPE_HEADS_MIN=2 ROPE_HEADS_MAX=8 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To recover the base model's "all heads rotary" behavior while keeping this candidate script, set:

```bash
ROPE_HEADS_MIN=8 ROPE_HEADS_MAX=8
```

## Main expected risks and tradeoffs

- If shallow layers still need more positional signal than expected, reducing the number of rotary heads early could hurt syntax/local ordering.
- The best head schedule may not be linear; `2 -> 8` is a strong first guess, not a tuned optimum.
- Because head ordering is static, this is only an approximation to the head-specialized behavior suggested by recent RoPE research.
- The current SOTA includes strong legal TTT; any pre-TTT gain from smarter RoPE may look smaller after TTT closes part of the gap.

## Validation

Commands run in this workspace:

```bash
python -m compileall ../../train_gpt.py ../../train_gpt_mlx.py ../../data train_gpt.py
```

Outcome:

- **Passed** for `train_gpt.py`, `train_gpt_mlx.py`, the `data/` utilities, and this candidate's `train_gpt.py`.

Attempted additional smoke check:

- I attempted a lightweight import/runtime smoke test with a FlashAttention stub so the new RoPE head schedule could be exercised on CPU.
- That was **not feasible in this environment** because the workspace Python environment does not currently have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- As a result, validation here is limited to syntax-level compilation rather than an actual model forward pass.

## Suggested next experiments

If this candidate is worth following up on, the next ablations should be:

1. `ROPE_HEADS_MIN/MAX = 4/8` vs `2/8`.
2. `ROPE_HEADS_MIN/MAX = 2/6` to preserve a couple of fully non-rotary heads even in deep layers.
3. Combine the same headwise ramp with the `2026-03-22` non-TTT base to isolate whether the gain is mostly pre-TTT or survives the full evaluation stack.
