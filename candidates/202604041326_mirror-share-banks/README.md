# Mirror-Shared Banks on the PR#414 Stack

## Hypothesis

The current best record already uses contiguous parameter banks for optimizer throughput, but it still keeps a distinct heavy Q/K/V/O and MLP bank for every logical layer. This candidate turns that into **true cross-layer sharing** by tying the heavy banks across mirrored encoder/decoder positions while keeping the existing layer-local control parameters (`attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, skip weights, XSA flags, VE scales, norms) untouched.

For an 11-layer model the shared-bank map is:

```text
[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
```

That keeps the middle layer unique, ties mirrored U-Net positions together, and cuts the physical bank count from 11 to 6. The saved bytes are reinvested into a larger `BigramHash` table (`BIGRAM_VOCAB_SIZE=3072` by default).

## Why this is promising here

Recent repo progress has mostly come from better evaluation, quantization, and incremental refinements on the strong 11-layer stack: XSA, EMA, partial RoPE, LN scale, GPTQ-lite, and LeakyReLU(0.5)^2. The repo has **parameter banking**, but not much **actual parameter sharing**. Since the leaderboard is constrained by a 16 MB artifact, reclaiming bytes from repeated heavy matrices is one of the cleanest ways to change the quality/size tradeoff without spending extra training compute.

This is also a better fit than literal recurrence. A prior non-record run found that doubling depth by reusing layers hurt because it cut the number of optimizer steps too much in a 10-minute wall-clock budget. Mirror sharing keeps the same logical depth and compute graph; it only reduces unique stored weights.

## Prior repo work that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current overall stack,
  - already uses banked heavy weights and LeakyReLU(0.5)^2,
  - showed `BigramHash 2048 -> 3072` helped on the same family of models.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest simpler training/export base,
  - confirmed export-side quality still matters near the top.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - partial RoPE and LN scale remain part of the carried stack.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative result: recurrence hurt because it traded away too many steps.

There were no prior `candidates/` directories in the repository when this candidate was created.

## External research that informed it

- **ALBERT** (cross-layer sharing for parameter efficiency): <https://arxiv.org/abs/1909.11942>
- **Universal Transformers** (shared weights across depth / iterative refinement): <https://arxiv.org/abs/1807.03819>
- **LoRA** (small untied adaptation paths are often enough when large base weights are shared or frozen): <https://arxiv.org/abs/2106.09685>

This candidate only implements the weight-sharing part for now. It does **not** add explicit low-rank per-layer deltas yet; that is the most obvious follow-up if plain mirror sharing is too restrictive.

## What changed vs. the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `BANK_SHARE_MODE` with `mirror` and `none`.
2. Replaced per-layer physical bank allocation with shared-bank allocation derived from `BANK_SHARE_MODE`.
3. Routed each logical layer through `layer_to_bank` instead of indexing a unique bank slice.
4. Updated bank export/import helpers so quantization operates on **shared bank slices**, not duplicated logical-layer copies.
5. Increased the default `BIGRAM_VOCAB_SIZE` from `2048` to `3072` to spend part of the recovered artifact budget.
6. Added a FlashAttention fallback to PyTorch SDPA so the script can be smoke-tested without `flash_attn_interface`.
7. Added `SMOKE_TEST=1`, which runs a tiny CPU-only forward pass plus export/roundtrip reload check.

## How to run

### Main candidate run

```bash
BANK_SHARE_MODE=mirror \
BIGRAM_VOCAB_SIZE=3072 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 candidates/202604041326_mirror-share-banks/train_gpt.py
```

If you want to compare directly against the untied banked stack, rerun with:

```bash
BANK_SHARE_MODE=none \
BIGRAM_VOCAB_SIZE=3072 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 candidates/202604041326_mirror-share-banks/train_gpt.py
```

## Validation

The local environment did not have the repo Python dependencies installed in its system interpreter, so the CPU smoke run was executed in an isolated temp virtualenv instead.

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604041326_mirror-share-banks/train_gpt.py` | Success |
| `SMOKE_TEST=1 NUM_LAYERS=4 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 MLP_MULT=2 BIGRAM_VOCAB_SIZE=128 XSA_LAST_N=1 ROPE_DIMS=16 VE_ENABLED=0 TRAIN_SEQ_LEN=32 EVAL_SEQ_LEN=32 python candidates/202604041326_mirror-share-banks/train_gpt.py` | Success in temp venv; printed `smoke_test:ok loss=6.9315 rt_loss=6.9320 shared_banks=2 bank_share_mode=mirror artifact_bytes=266828` |

## Main risks / tradeoffs

- Mirror sharing may be too restrictive if late decoder layers need meaningfully different heavy matrices than early encoder layers.
- The current implementation relies on existing layer-local control tensors as the adaptation path; if that is not enough, the next step is to add tiny LoRA-style per-layer deltas on shared banks.
- Bigger `BigramHash` may not be the best way to spend the recovered bytes; VE dim or selective higher-precision export may be better reinvestments.
- The strongest record also uses legal TTT. This candidate intentionally isolates the training/export architecture change first instead of coupling it to another evaluation change.
