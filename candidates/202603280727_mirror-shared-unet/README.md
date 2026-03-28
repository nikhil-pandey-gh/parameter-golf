# Mirror-Shared U-Net Core Reuse

## Hypothesis

The current repo has squeezed a lot of value out of the same 11-layer, 512d, 3x-MLP stack through evaluation and quantization tricks, but it has not really explored **structured cross-layer parameter sharing**. This candidate reuses the same transformer block core across mirrored encoder/decoder positions in the existing U-Net-style stack, so the model keeps the strong **11 logical-layer compute graph** while storing only **6 unique block cores**. The saved artifact budget is then reinvested into two repo-proven wins the 2026-03-22 base did not use together: **LeakyReLU(0.5)^2** in the MLP and **fp16 tied-embedding export**.

## Why this is promising for this repository

The strongest training-only record in this repo, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, already converged on a very specific recipe: 11 layers, 3x MLP, BigramHash, SmearGate, XSA on late layers, Partial RoPE, LN scaling, EMA, and GPTQ-lite int6 export. The final leaderboard winner on 2026-03-23 added LeakyReLU(0.5)^2 and legal TTT, suggesting the remaining gains are coming from small but high-leverage changes on top of the same core stack.

This candidate asks a different question: if the U-Net skip layout already pairs encoder and decoder stages, can the repository exploit that symmetry to **share stored block weights without shrinking logical depth**? That is attractive under the challenge's 16 MB artifact budget because the savings come mostly from repeated block matrices rather than from removing compute or context length. Unlike the negative 1x5090 `layer recurrence x2` result in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`, this candidate does **not** add extra logical depth or extra recurrent steps; it keeps the same 11-block execution path and only changes how weights are stored and reused.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen base implementation. Supplies the strongest training-only stack in the repo: 11 layers, EMA, GPTQ-lite int6, Partial RoPE, LN scale, XSA4, VE128, BigramHash, SmearGate.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Motivated the MLP activation change to LeakyReLU(0.5)^2, which that record reports as a meaningful standalone gain.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - Motivated keeping the tied embedding in fp16 at export, since the repo previously found the embedding/output-head matrix to be unusually sensitive to quantization.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Important negative control: naive layer recurrence hurt badly when it increased effective depth/compute. That steered this candidate toward **mirror sharing at fixed logical depth** instead of adding recurrent passes.

## External research that informed it

- **ALBERT** (Lan et al., 2019, arXiv:1909.11942)
  - Classic evidence that cross-layer parameter sharing can reduce model size substantially while preserving strong language-model behavior.
- **Universal Transformer** (Dehghani et al., 2018, arXiv:1807.03819)
  - Motivates repeated application of shared transformer computations across depth rather than treating every layer as fully independent.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (Nguyen and Lin, 2025, arXiv:2505.01855)
  - Relevant because it argues that targeted recurrence/weight reuse within a transformer can be useful, rather than uniform repetition everywhere.
- **SCORE: Replacing Layer Stacking with Contractive Recurrent Depth** (Godin, 2026, arXiv:2603.10544)
  - Recent evidence that replacing fully independent stacked layers with shared recurrent depth can reduce parameters while maintaining or improving optimization behavior.
- **Thinking Deeper, Not Longer: Depth-Recurrent Transformers for Compositional Generalization** (Chen, 2026, arXiv:2603.21676)
  - Reinforces the broader hypothesis that decoupling depth from parameter count is a promising design axis, especially when recurrence is kept stable.

## What changed versus the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Mirror-shared U-Net block cores**
   - New default `SHARE_MIRROR_BLOCKS=1`.
   - The 11 logical layers now map to 6 unique block cores with the pattern:
     - encoder: `0,1,2,3,4`
     - bottleneck/early decoder center: `5`
     - decoder mirrors: `4,3,2,1,0`
   - This keeps the same logical depth and skip topology while shrinking stored block matrices.

2. **Per-logical-layer XSA and LN scaling preserved**
   - XSA usage and LN scale factors are now driven by logical-layer metadata instead of assuming every logical layer owns a unique block instance.
   - This lets the candidate keep late-layer XSA and layerwise LN scaling even when weights are shared.

3. **LeakyReLU(0.5)^2 MLP**
   - Added `MLP_NEGATIVE_SLOPE` with default `0.5`.
   - Replaces ReLU^2 with LeakyReLU(0.5)^2 by default.

4. **fp16 tied-embedding export**
   - Added `FP16_EMBED_EXPORT=1` by default.
   - `tok_emb.weight` is preserved as fp16 passthrough during mixed int6 export when embeddings are tied.

5. **Slightly larger BigramHash budget**
   - Default `BIGRAM_VOCAB_SIZE` is raised from `2048` to `3072`, using some of the saved artifact headroom on a change the repo already saw help in later work.

6. **Self-describing shared-weight exports**
   - The exported checkpoint keeps the compact shared `block_bank.*` weights and stores the logical-layer routing as `__logical_to_block__`, so the artifact carries the mirror-share mapping needed for roundtrip reloads.

## How to run or evaluate it

From the repository root:

```bash
RUN_ID=mirror_shared_unet \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SHARE_MIRROR_BLOCKS=1 \
FP16_EMBED_EXPORT=1 \
BIGRAM_VOCAB_SIZE=3072 \
MLP_NEGATIVE_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 candidates/202603280727_mirror-shared-unet/train_gpt.py
```

The script is self-contained and can also be run from inside the candidate directory, but the dataset and tokenizer paths need to be adjusted relative to that working directory (for example `DATA_PATH=../../data/datasets/fineweb10B_sp1024/` and `TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model`).

From inside `candidates/202603280727_mirror-shared-unet/`:

```bash
RUN_ID=mirror_shared_unet \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SHARE_MIRROR_BLOCKS=1 \
FP16_EMBED_EXPORT=1 \
BIGRAM_VOCAB_SIZE=3072 \
MLP_NEGATIVE_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run while preparing this candidate:

```bash
python -m compileall candidates/202603280727_mirror-shared-unet/train_gpt.py
python -m py_compile candidates/202603280727_mirror-shared-unet/train_gpt.py
python - <<'PY'
import torch
PY
```

Expected outcome:

- `compileall` succeeded.
- `py_compile` succeeded.
- the runtime dependency probe failed in this runner with `ModuleNotFoundError: No module named 'torch'`.

CPU-only smoke test note:

- A real runtime smoke test was **not feasible in this environment** because the current runner is missing the core training dependency stack (`torch` was not importable), and the script also expects the challenge runtime stack used by recent records (CUDA/FlashAttention plus real FineWeb/tokenizer artifacts). Running a fake smoke test without those dependencies would not meaningfully validate the candidate.

## Main risks and tradeoffs

- **Over-sharing may hurt expressivity.** Sharing full block cores across mirrored positions is stronger than just sharing MLPs or attention projections.
- **Quantization dynamics may change.** A smaller set of repeated block matrices can compress very well, but repeated reuse may also make quantization mistakes more correlated.
- **Mirror symmetry might be too restrictive.** Encoder and decoder positions in the U-Net stack are related, but not necessarily identical in role.
- **The best use of saved bytes is uncertain.** This candidate spends them on fp16 embeddings and a larger BigramHash because those are already repo-supported, but future sweeps may show that other reallocations are better.

## Suggested next experiments

- Ablate `SHARE_MIRROR_BLOCKS=0/1` while keeping LeakyReLU^2 and fp16 embedding export fixed.
- Try partial sharing variants, e.g. share only MLPs or only the deepest mirrored layers.
- Sweep `BIGRAM_VOCAB_SIZE` upward now that the artifact budget should be looser.
- If runtime headroom remains acceptable, test whether the saved artifact bytes can fund a wider MLP or a higher-precision embedding/value pathway.
