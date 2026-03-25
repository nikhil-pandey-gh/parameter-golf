# Shared Early MLP Banks + Late FC2 Int8 Protection

## Hypothesis

The current best stack is already strong on evaluation tricks and optimizer systems work, but it still spends a large share of its artifact budget on per-layer MLP weights. My hypothesis is that **sharing only the early/mid MLP banks** while **keeping the last few layers unique and better protected at quantization time** is a better trade than either full recurrence or uniform low-bit quantization.

Concretely, this candidate shares early 3x-MLP weights in contiguous pairs, keeps the last 3 layers' MLP banks unique, and forces the final 2 MLP down-projection banks (`fc2` / `proj`) to export in int8 instead of int6. The saved bytes are reinvested into a larger `BigramHash` default (`3072`) while keeping the rest of the strong 2026-03-23 stack intact.

## Why this is promising for this repository

Two repo patterns motivate this direction:

- The leaderboard repeatedly improved by finding **artifact-budget-neutral or artifact-positive changes** rather than paying large extra compute costs. Sliding eval, EMA, GPTQ-lite clip search, Partial RoPE, LN scale, and LeakyReLU^2 all fit that pattern.
- The repo already contains a negative result for **extra recurrent depth under the 10-minute budget**. The single-GPU non-record run reports layer recurrence as a clear regression because it halves the number of optimizer steps in fixed wall-clock time. This candidate instead keeps compute roughly flat and only changes **how parameters are reused and exported**.

The goal is to borrow the useful inductive bias of depth reuse without repeating layers in the forward pass.

## Influenced by prior records and candidates

There were **no prior populated `candidates/` directories** to build on at review time.

The main record influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current stack
  - parameter banking + parallel Muon already make bank-level sharing a small edit
  - LeakyReLU(0.5)^2 and legal score-first TTT stay in place
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - GPTQ-lite style per-row clip search remains the quantization base
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Partial RoPE + LN scale are retained, while avoiding reliance on the dead Late-QAT path noted in that README
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - both explicitly warn that recurrence / looping layers was a bad trade under a short wall-clock budget

## External research that informed this candidate

- **ALBERT** (`arXiv:1909.11942`) showed that cross-layer parameter sharing can substantially reduce model size while preserving strong language-model quality.
- **Universal Transformer** (`arXiv:1807.03819`) argued that recurrence / reuse can provide a helpful inductive bias, but its direct form increases compute.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (`arXiv:2505.01855`) found that recurrence is most useful when allocated selectively, especially toward earlier layers.
- **Scaling Law for Quantization-Aware Training** (`arXiv:2505.14302`) identifies the FC2 / down-projection path as a quantization bottleneck and motivates protecting that path with mixed precision.
- **LieQ** (`arXiv:2508.03332`) argues that layerwise information effectiveness should inform bit allocation rather than using one uniform low-bit policy everywhere.

This candidate translates those papers into a repository-compatible version: share earlier MLP parameters, keep late layers more specific, and spend saved bytes on protecting the most sensitive late MLP down-projections.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Shared early MLP banks**
   - new defaults: `SHARED_MLP_GROUP_SIZE=2`, `SHARED_MLP_UNSHARED_LATE=3`
   - for the 11-layer stack, that means layers `0-7` share MLP banks in contiguous pairs while layers `8,9,10` keep unique MLP banks
   - attention banks remain per-layer and unchanged

2. **Late MLP mixed precision at export**
   - new default: `LATE_MLP_INT8_LAYERS=2`
   - the final 2 unique MLP down-projection banks export as int8 instead of int6
   - shared early MLP banks still use GPTQ-lite int6 export

3. **Larger default local-context budget**
   - `BIGRAM_VOCAB_SIZE` default raised to `3072`
   - the point is to spend some of the recovered artifact budget on a signal that already helped prior records

4. **CPU-friendly local smoke mode**
   - `SMOKE_TEST=1` runs a model-only path that instantiates the candidate, does a forward/backward pass on random tokens, and round-trips the shared-bank quantization logic
   - it also falls back from FlashAttention to PyTorch SDPA when FlashAttention is unavailable

5. **Candidate-directory-safe defaults**
   - default dataset/tokenizer paths are resolved from the repository root so the script can be run from inside this candidate directory

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603251219_shared-mlp-late-int8

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 \
SHARED_MLP_GROUP_SIZE=2 SHARED_MLP_UNSHARED_LATE=3 LATE_MLP_INT8_LAYERS=2 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a dependency-only local smoke path (no dataset access):

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation

Validated on this workflow runner:

- `python -m compileall candidates/202603251219_shared-mlp-late-int8/train_gpt.py`
  - **passed**
- `SMOKE_TEST=1 python candidates/202603251219_shared-mlp-late-int8/train_gpt.py`
  - **could not run on this runner** because the baseline repository runtime dependency `torch` is not installed here (`ModuleNotFoundError: No module named 'torch'`)

The smoke path is still included because it should be runnable in a normal repository environment where `torch` is available, even if `numpy`, `sentencepiece`, or FlashAttention are absent locally.

## Main expected risks / tradeoffs

- **Too much sharing can underfit.** Early layers may tolerate reuse, but sharing beyond pairs or sharing late layers may erase real depth-specific behavior.
- **Bank sharing changes optimizer statistics.** Parallel Muon now sees fewer unique MLP banks, which may change the stability/speed balance in ways that only a real GPU run will reveal.
- **Mixed precision may spend bytes in the wrong place.** Protecting the final 2 MLP down banks is motivated by quantization literature, but the best budget split for this exact tiny model is still uncertain.
- **TTT interactions are unknown.** Because TTT adapts the quantized/eval model after export, shared early MLPs might help regularize adaptation or might reduce plasticity.

## Suggested next experiments

1. Sweep `SHARED_MLP_GROUP_SIZE` in `{2, 3}` while keeping `SHARED_MLP_UNSHARED_LATE >= LATE_MLP_INT8_LAYERS`.
2. Compare `LATE_MLP_INT8_LAYERS` in `{1, 2, 3}`.
3. If artifact headroom remains, try `BIGRAM_VOCAB_SIZE=4096`.
4. If sharing hurts too much, restrict it to encoder-only MLP banks instead of the whole early stack.
