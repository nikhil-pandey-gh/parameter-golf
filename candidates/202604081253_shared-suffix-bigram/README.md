# Shared Suffix Bigram Reuse

## Hypothesis

The strongest pre-TTT stacks in this repo already concentrate most of their useful complexity in the deepest layers: XSA is only enabled late, value embeddings only help in the suffix, and the best zero-parameter wins also land on top of deep 11-layer runs. The candidate hypothesis is that **sharing only the deepest block bodies** can recover artifact bytes with much less quality loss than full-model recurrence, and that those saved bytes are better spent on a slightly richer local token-pair signal.

Concretely, this candidate keeps **11 effective layers**, but shares the final two heavy block bodies once so the effective schedule becomes:

`[0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 8]`

The per-layer RMSNorms, residual mixing, attention gain (`q_gain`), scales, skip weights, XSA placement, and VE placement all remain layer-local. Only the large attention/MLP weights are reused.

## Why it is promising for this repository

This repo has already mined obvious gains out of sliding-window eval, mixed quantization, EMA/SWA, partial RoPE, and activation tweaks. The remaining underexplored lane is **parameter sharing that does not also increase training-time compute**.

This design specifically fits the evidence in `records/`:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` is a strong, stable pre-TTT base with EMA, GPTQ-lite clip search, XSA, partial RoPE, VE, and BigramHash.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` shows late-layer, zero-parameter attention tweaks stack cleanly.
- `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` shows that **larger BigramHash tables still help** when the artifact budget allows it.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` shows **LeakyReLU(0.5)^2** is a cheap, transferable MLP upgrade.
- The older `2026-03-19_SlidingWindowEval` codebase had a dormant looped/shared-layer path, but did not ship it; this candidate revisits that direction in a much narrower form.
- The 1x5090 non-record sweep reported that naive layer recurrence hurt, which is why this candidate uses **same-depth sharing** instead of adding more effective layers or extra training-time passes.

## External research that informed it

1. **ALBERT** (Lan et al., arXiv:1909.11942) showed that cross-layer parameter sharing can preserve quality while cutting parameter count substantially.
2. **Universal Transformer** (Dehghani et al., arXiv:1807.03819) showed that repeatedly applying a shared transformation across depth can be useful when the model still keeps enough position-wise specialization.
3. I also considered a quantization-first follow-up based on **LSQ** (Esser et al., arXiv:1902.08153), but the repo has already explored QAT/GPTQ-lite much more aggressively than depth sharing. For this codebase, the cleaner next bet looked like **ALBERT-lite sharing on the suffix**.

## What changed versus the chosen base

Base implementation: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Main changes in this candidate:

1. **Shared deep suffix cores**
   - Adds `SHARED_SUFFIX_LAYERS` and `SHARED_SUFFIX_REPEATS`.
   - Defaults to `SHARED_SUFFIX_LAYERS=2`, `SHARED_SUFFIX_REPEATS=1`.
   - Stores shared attention/MLP weights once in `block_cores`, while each effective layer keeps its own norms/scales/`q_gain`/gates in `blocks`.
2. **Bigger BigramHash default**
   - `BIGRAM_VOCAB_SIZE` default changes from `2048` to `4096`.
3. **LeakyReLU(0.5)^2 MLP**
   - Adds `MLP_LEAK`, default `0.5`.
4. **Path-safe candidate execution**
   - Default data/tokenizer paths now resolve relative to the repository root, so the script can be launched directly from this candidate directory.
5. **CPU-safe attention fallback**
   - If `flash_attn_interface` is unavailable **or the runtime is not Hopper-class CUDA**, the script falls back to PyTorch SDPA and keeps math/mem-efficient SDPA backends enabled. This is mainly to make import/smoke flows less brittle outside the final Hopper environment.

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202604081253_shared-suffix-bigram

RUN_ID=shared_suffix_bigram \
NUM_LAYERS=11 \
SHARED_SUFFIX_LAYERS=2 \
SHARED_SUFFIX_REPEATS=1 \
BIGRAM_VOCAB_SIZE=4096 \
MLP_LEAK=0.5 \
EVAL_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults already point back to the repository root for `data/` and `tokenizers/`, so no extra path overrides are required if the standard dataset layout exists.

## Main risks / tradeoffs

- **Oversharing risk:** even suffix-only sharing may still hurt more than it helps if the final layers really need independent attention/MLP weights.
- **Where to spend the recovered bytes is uncertain:** this version reinvests only part of the savings into a larger BigramHash table; the best reallocation might instead be a wider VE path, a wider MLP, or more buckets.
- **Compiler-sensitive late QAT remains inherited from the base stack:** the primary hypothesis here is the sharing layout, not a rework of the quantization schedule.
- **Compute is unchanged in effective depth but representation changes are structural:** if the shared cores underfit, the candidate may compress well without improving BPB.

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall candidates/202604081253_shared-suffix-bigram/train_gpt.py` | Passed |
| `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604081253_shared-suffix-bigram/train_gpt.py` | Passed |
| Tiny CPU smoke import/forward | Blocked in this workflow environment because `torch` is not installed (`requirements.txt` lists it, but the runtime image here does not provide it), so a no-dependency smoke run was not feasible |
