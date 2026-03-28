# Tail Block Share (202603282143)

## Hypothesis

A stronger next compact-model candidate for this repo is to keep the proven March 22 frontier stack, make it *effectively deeper* than 11 layers, and pay for that extra depth by sharing only the heaviest tail-layer attention/MLP weights.

Concretely, this candidate defaults to `NUM_LAYERS=13`, but the last 4 logical layers are pairwise weight-shared as `[..., 9, 9, 10, 10]`, so the model keeps only 11 unique heavy block-weight sets while still executing 13 logical blocks. The per-layer residual mixing, scaling vectors, skip weights, layer-index-dependent LN scaling, and value-embedding scales stay distinct, which keeps more specialization than a fully looped or fully tied stack.

## Why it is promising for this repository

Recent repo history strongly suggests three things:

1. Deeper 10-11 layer models consistently beat the 9-layer baseline once quantization and evaluation improve.
2. The best stacks are already close to the artifact limit, so adding more *unique* block weights is expensive.
3. Full looped-depth paths were already present in older code and did not emerge as winners, which argues for a narrower, lower-risk sharing strategy rather than fully recurrent depth.

This candidate tries to sit exactly in that gap: keep the successful 11-layer recipe, add depth pressure where tiny LMs often need it, and only share the expensive tail matrices instead of the whole network.

## Prior records and repo evidence that influenced it

This candidate is primarily based on the following runs:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Clean strong base stack: 11L / 512d / GQA / MLP 3x / XSA last 4 / partial RoPE / LN scale / BigramHash / VE / EMA / GPTQ-lite.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Contributed the low-cost `LeakyReLU(0.5)^2` activation change, which was reported as a meaningful gain on top of the frontier stack.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/`
  - Important reminder that evaluation choices matter a lot, so the candidate keeps the repository’s sliding-window evaluation path intact.

There were no prior folders under `candidates/` when this candidate was created.

## External research that informed it

- **MobileLLM / MobileLLM-LS** (`arXiv:2402.14905`)
  - Motivates deep-and-thin small LMs and specifically reports gains from immediate block-wise weight sharing with no model-size increase.
- **ALBERT** (`arXiv:1909.11942`)
  - Classic evidence that cross-layer parameter sharing can preserve much of the benefit of deeper models while reducing parameter count.
- **Universal Transformer** (`arXiv:1807.03819`)
  - Useful conceptual support for reusing computation across depth, but this candidate intentionally uses a more conservative partial-sharing variant instead of full recurrence.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes five targeted changes:

1. **Effective depth increase with pairwise tail sharing**
   - New hyperparameter: `SHARED_TAIL_PAIR_LAYERS`.
   - Default logical depth is now `NUM_LAYERS=13` with `SHARED_TAIL_PAIR_LAYERS=4`.
   - The deepest 4 logical layers share attention + MLP submodules in adjacent pairs, yielding the logical-to-physical map:
     - `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10]`
   - This keeps 11 unique heavy block-weight sets while allowing 13 logical applications.

2. **Per-layer specialization is deliberately retained**
   - The weight sharing only aliases the expensive `attn` and `mlp` submodules.
   - Per-layer residual-mix vectors, attention/MLP scales, skip weights, LN depth scaling, and value-embedding scales remain layer-specific.

3. **LeakyReLU(0.5)^2 MLP activation**
   - Replaces ReLU-squared with the stronger activation used by the March 23 record.

4. **Alias-aware export / quantized roundtrip**
   - Shared tail weights are exported only once.
   - Alias metadata is saved alongside the quantized payload and re-expanded before roundtrip evaluation.
   - This is necessary so artifact size actually reflects the sharing rather than serializing duplicate logical-layer copies.

5. **Safer attention fallback for CUDA environments without `flash_attn_interface`**
   - If `flash_attn_interface` is unavailable, the script falls back to PyTorch `scaled_dot_product_attention`.
   - This is not the intended leaderboard path, and `main()` still expects CUDA, but it makes the candidate easier to smoke-test or adapt in properly provisioned CUDA environments that do not have the external FlashAttention interface installed.

## How to run or evaluate it

The script is written so it can be launched **from inside this candidate directory**. By default it resolves the dataset and tokenizer paths relative to the repository root.

From the repo root:

```bash
cd candidates/202603282143_tail-block-share
SEED=1337 \
NUM_LAYERS=13 \
SHARED_TAIL_PAIR_LAYERS=4 \
VE_LAYERS=11,12 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
SWA_ENABLED=1 \
WARMDOWN_ITERS=3500 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to compare against a no-sharing ablation while keeping this file, set:

```bash
SHARED_TAIL_PAIR_LAYERS=0 NUM_LAYERS=11 VE_LAYERS=9,10
```

## Main risks and tradeoffs

- **More logical depth means more compute**, even if artifact size stays close to the 11-layer baseline.
- **Shared tail weights may overshare late-layer behavior**, especially if the deepest layers want meaningfully different attention patterns.
- **The repo already saw weak results from broader looped-depth ideas**, so this could still land on the wrong side of the compute/quality tradeoff.
- **Quantization interaction is uncertain**: shared matrices may compress well, but the deeper logical stack could still change the optimal clipping / QAT regime.

## Suggested next experiments if this idea is promising

1. Sweep `SHARED_TAIL_PAIR_LAYERS` over `{2, 4, 6}`.
2. Compare `NUM_LAYERS=12/13/14` at fixed unique heavy-layer count.
3. Move VE from `11,12` to `10,11,12` if deeper tail layers seem under-conditioned.
4. Combine this candidate with the newer March 23 lzma export path and/or parameter banking if the architecture itself looks promising.

## Validation

Validated in this workflow environment with:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603282143_tail-block-share/train_gpt.py
```

Outcome:

- `train_gpt.py`, `train_gpt_mlx.py`, `data/`, and `candidates/202603282143_tail-block-share/train_gpt.py` all compiled successfully.

I also attempted to run a tiny CPU-only smoke test by conditionally importing the candidate module and instantiating a very small `GPT`, but this workflow container does not currently have the repository runtime dependencies installed (`torch`, `sentencepiece`, and `numpy` were all unavailable), so a real forward-pass smoke test was **not feasible here**.

That limitation is specific to this workflow environment; the script includes a non-FlashAttention fallback specifically to make adaptation easier in properly provisioned CUDA Python environments that do not have `flash_attn_interface`, but `main()` still requires CUDA.
