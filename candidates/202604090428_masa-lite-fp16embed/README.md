# MASA-lite Attention Sharing + FP16 Tied Embedding

## Hypothesis

The current record stack already exploits most of the obvious zero-cost wins (XSA, Partial RoPE, LN scale, EMA, LeakyReLU(0.5)^2, stronger eval). A promising remaining gap is to **reclaim artifact budget from redundant attention weights** and spend those bytes on the **most quantization-sensitive tensor in the repo: the tied embedding / output table**.

This candidate therefore replaces per-layer Q/K/V/O matrices with a small **shared attention atom bank** plus learned per-layer mixing weights, then uses the freed budget to keep `tok_emb.weight` in **fp16** during export instead of quantizing it.

## Why this is promising for this repository

Two repository patterns point in the same direction:

1. `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600` showed that the tied embedding is unusually sensitive to quantization and that preserving it in fp16 dramatically reduces the post-quantization gap.
2. The later 11-layer record stack (`2026-03-21` through `2026-03-23`) improved quality by stacking mostly low-parameter changes, but still quantized embeddings and still carried a full set of per-layer attention matrices.

At the same time, prior local negative results said **depth recurrence / layer recurrence** was a bad trade under a 10-minute cap. This candidate avoids that dead end: it does **not** add more layer applications or more wall-clock compute; it only changes how attention parameters are represented.

## Prior records that influenced this candidate

- **`2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`** — chosen base stack: 11 layers, XSA on late layers, Partial RoPE, LN scale, BigramHash, EMA.
- **`2026-03-18_FP16Embed_WD3600`** — direct motivation for fp16 embedding passthrough.
- **`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`** — motivated the LeakyReLU(0.5)^2 MLP swap.
- **`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`** — motivated the longer warmdown and late-stack defaults.
- **`track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`** and **`2026-03-18_FP16Embed_WD3600`** — both argue against naive layer recurrence under the fixed wall-clock budget.

## External research that informed it

- **ALBERT** (Lan et al., 2019, arXiv:1909.11942): cross-layer parameter sharing can preserve quality while reducing parameters.
- **Subformer** (Reid et al., 2021, arXiv:2101.00234): generative transformers can benefit from structured/sandwich-style sharing instead of naive full-layer tying.
- **Share Your Attention / MASA** (Zhussip et al., 2025, arXiv:2508.04581): attention projections across layers are redundant enough to be represented by shared matrix atoms with learned mixtures.
- **Rethinking Weight Tying / PIT** (Gu et al., 2026, arXiv:2602.04556): compact LMs are particularly sensitive to embedding/unembedding interface quality, reinforcing the value of preserving the shared token table.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train_gpt.py`

This candidate adds:

1. **AttentionAtomBank**: `SHARED_ATTN_ATOMS=4` by default. Q/K/V/O weights are synthesized from a small bank of shared 2D atoms plus learned per-layer softmax mixing weights.
2. **FP16 embedding passthrough**: `tok_emb.weight` is stored as fp16 during mixed int6 export instead of being quantized.
3. **LeakyReLU(0.5)^2** in the MLP path.
4. Candidate defaults aligned to the late strong stack: 11L / 2048 context / XSA4 / Partial RoPE(16) / LN scale / EMA / warmdown3500 / BigramHash3072.
5. **Runtime robustness helpers**:
   - FlashAttention-3 import now has an SDPA fallback.
   - `SMOKE_TEST=1` runs a synthetic forward/backward + quantization roundtrip without dataset access, for environments that have the repo Python deps installed.

## How to run / evaluate

From this candidate directory:

```bash
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable attention sharing
SHARED_ATTN_ATOMS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Disable fp16 embedding passthrough
FP16_EMBED_PASSTHROUGH=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Smaller synthetic smoke mode (no dataset needed; deps still required)
SMOKE_TEST=1 NUM_LAYERS=4 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2 BIGRAM_VOCAB_SIZE=128 BIGRAM_DIM=32 SHARED_ATTN_ATOMS=2 \
TRAIN_SEQ_LEN=32 python train_gpt.py
```

## Validation run here

### Passed

```bash
python -m compileall candidates/202604090428_masa-lite-fp16embed/train_gpt.py
```

### Not feasible in this runner

I attempted a minimal smoke launch, but this environment does not have the repository runtime dependencies installed (`torch`, `numpy`, and `sentencepiece` were all absent), so a real start-up smoke test could not run here. The script includes `SMOKE_TEST=1` specifically to make that validation cheap in an environment where the declared repo dependencies are present.

## Main expected risks / tradeoffs

- **Compile behavior**: shared weight mixing changes the forward graph, so it may not preserve the same compile profile as the original record scripts.
- **Regularization vs capacity**: attention sharing may improve compression and robustness, but too few atoms can underfit.
- **Where to spend saved bytes**: fp16 embeddings are the first use of recovered budget, but later experiments may find better tradeoffs by reallocating some of that room to VE/bigram width or a stronger quantizer.
- **Softmax mixing may be too restrictive**: a future iteration may need unconstrained or low-rank mixing if convex combinations prove too limiting.
