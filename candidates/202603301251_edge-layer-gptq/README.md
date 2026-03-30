## Candidate: Edge-layer GPTQ-lite (SliderQuant-inspired)

### Hypothesis

The strongest non-TTT stack in this repository is already export-limited: `GPTQ-lite` helped, but the best records still spend a measurable amount of BPB on quantization damage. Recent PTQ work argues that shallow and deep transformer layers are more quantization-sensitive than middle layers. This candidate tests the simplest version of that idea for this repo: keep the interior `attn` and `mlp` matrices on the existing `int6` GPTQ-lite path, but quantize the first and last transformer blocks with a gentler `int8` per-row GPTQ-lite path.

### Why this is promising for this repository

This repository has consistently rewarded selective precision budgeting:

- `2026-03-18_FP16Embed_WD3600` showed that protecting especially sensitive tensors can collapse the quantization gap.
- `2026-03-19_MixedQuant_Int6Int8_SlidingWindow` and related int6/int8 runs showed that mixed precision is competitive under the 16MB artifact cap.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that better per-row clipping alone still buys measurable BPB.

The new twist here is **where** the extra precision goes: not embeddings globally, but the edge blocks that latest PTQ research says are most fragile.

### Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Base implementation for this candidate.
  - Contributed the 11-layer EMA + GPTQ-lite + warmdown3500 + late-QAT recipe.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Motivated keeping the current 11-layer Partial-RoPE/LN-scale/XSA-style architecture family intact rather than changing too many moving parts at once.

- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
  - Evidence that mixed-precision export is one of the highest-leverage knobs in this challenge.

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - Earlier evidence that selective protection of quantization-sensitive tensors can matter more than broad architectural churn.

### External research that informed it

- **SliderQuant** — Wang et al., 2026. [arXiv:2603.25284](https://arxiv.org/abs/2603.25284)
  - Key observation used here: shallow and deep layers are usually more sensitive to PTQ than middle layers, with the first and last layers especially fragile.

- **LLM.int8()** — Dettmers et al., 2022. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
  - Reinforces the mixed-precision idea: transformer quantization quality improves when especially sensitive directions/components get gentler treatment instead of forcing one precision rule everywhere.

### What changed versus the base implementation

This candidate starts from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` and keeps the training architecture and optimizer stack unchanged.

Main changes:

1. **Layer-aware export quantization**
   - Added `EDGE_LAYER_COUNT` (default `1`) so the first and last transformer blocks are treated as an edge band.
   - Added separate clip settings for interior vs edge blocks:
     - `MAIN_LAYER_CLIP_RANGE=31` with the existing GPTQ-lite percentile set.
     - `EDGE_LAYER_CLIP_RANGE=127` with a gentler percentile set.
   - Applied the edge policy only to `mlp` and `attn` matrices, keeping the rest of the export path unchanged.

2. **Generalized GPTQ-lite helper**
   - Refactored the per-row percentile-search quantizer so the same logic can serve both the interior `int6` path and the new edge `int8` path.

3. **Candidate-local usability improvements**
   - Default dataset and tokenizer paths are now resolved relative to the repository root, so the script can be run directly from this candidate directory.
   - Added a FlashAttention fallback for environments without `flash_attn_interface`.
   - Added a `SMOKE_TEST_ONLY=1` path intended for random-token sanity checks without touching the dataset.

### How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603301251_edge-layer-gptq

NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
EDGE_LAYER_COUNT=1 \
EDGE_LAYER_CLIP_RANGE=127 \
MAIN_LAYER_CLIP_RANGE=31 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script defaults already match the intended candidate settings, so the new edge-layer quantization env vars can be omitted unless you want to ablate them explicitly.

For a cheap local sanity check on a machine with PyTorch installed:

```bash
cd candidates/202603301251_edge-layer-gptq
SMOKE_TEST_ONLY=1 python train_gpt.py
```

### Main expected risks or tradeoffs

- The layer-sensitivity result is strongest in larger LLM PTQ papers; this 11-layer compact model may not exhibit the same edge/interior split.
- `int8` on the edge blocks is storage-compatible with this serializer, but broader value ranges may slightly hurt zstd compression.
- The idea targets export quality only; if the current score is dominated by evaluation tricks rather than quantization damage, gains may be small.
- Because the training loop is unchanged, this candidate deliberately leaves open a second step: combining edge-layer quantization with the newer LeakyReLU² and/or legal TTT stacks.

### Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603301251_edge-layer-gptq/train_gpt.py
```

Outcome:

- Passed.

Attempted smoke check:

```bash
SMOKE_TEST_ONLY=1 python candidates/202603301251_edge-layer-gptq/train_gpt.py
```

Outcome:

- Could not be completed in this container because `torch` is not installed here (`ModuleNotFoundError: No module named 'torch'`).
- The candidate script still includes the `SMOKE_TEST_ONLY=1` path so the same sanity check can be run on a machine with the normal repo dependencies installed.
