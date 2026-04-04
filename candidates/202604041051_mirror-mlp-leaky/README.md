# Mirror-shared MLP + LeakyReLU^2

## Hypothesis

The strongest next low-risk architecture change is to **share the heavy MLP cores across mirrored encoder/decoder layers** in the existing 11-layer U-Net-style stack, while keeping the per-layer norms, attention blocks, residual mixing, skip weights, and scales unique. That should cut a large chunk of artifact bytes without paying the throughput penalty that hurt naive recurrent-depth experiments, and the saved budget can be spent on a slightly larger hashed n-gram pathway.

This candidate also folds in the repo's most convincing zero-parameter activation tweak, **LeakyReLU(0.5)^2**, to avoid giving back quality while the mirrored layers learn to reuse the same feed-forward subspace.

## Why this is promising for this repository

- The current frontier is already built around an **encoder/decoder U-Net split with mirrored skips**, so symmetry-aware sharing fits the model shape much better than generic recurrence.
- Prior non-record recurrence experiments were negative when they **added extra sequential compute** in a fixed wall-clock budget; this candidate instead keeps the same 11 forward passes and only shares weights.
- The 2026-03-22 pre-TTT record already gets most of the benefit from **EMA + GPTQ-lite + XSA + partial RoPE + LN scaling**, so a new idea should attack **artifact efficiency** rather than rebuild the training stack.
- Sharing five 3x-MLP mirrors removes about **7.9M duplicated MLP weights** from the exported model before compression, which is enough headroom to safely raise the default BigramHash table from 2048 to 3072.

## Prior records that influenced this candidate

- **2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233**: chosen base implementation because it is the strongest pre-TTT stack and already has the export path this candidate needs.
- **2026-03-23_LeakyReLU_LegalTTT_ParallelMuon**: contributed the LeakyReLU(0.5)^2 activation and the evidence that it helps on this family.
- **2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271** and **2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248**: established the current architecture backbone.
- **2026-03-18_FP16Embed_WD3600** and **2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090**: useful negative evidence that naive looped/recurrent depth is not enough under a hard wall-clock cap.

## External research that informed it

- **ALBERT** (Lan et al., 2019; https://arxiv.org/abs/1909.11942): cross-layer parameter sharing is a strong way to trade parameter count for depth/compute.
- **Universal Transformers** (Dehghani et al., 2018; https://arxiv.org/abs/1807.03819): repeated transformer computation can improve parameter efficiency, but should be adapted carefully to preserve parallelism.
- **ReZero** (Bachlechner et al., 2020; https://arxiv.org/abs/2003.04887): supports the idea that keeping lightweight per-layer residual controls unique can stabilize reused cores.

I also reviewed more aggressive compact-model directions such as LSQ-style late low-bit QAT, codebook quantization, and factorized embeddings, but mirrored sharing was the best fit for the current code without introducing new infrastructure.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Mirror-shared MLPs**: blocks `0<->10`, `1<->9`, `2<->8`, `3<->7`, and `4<->6` now reuse the same `MLP` module; block `5` stays unique.
2. **Export deduplication**: the quantized artifact stores one copy of each shared tensor plus an alias map, so sharing actually reduces compressed bytes.
3. **LeakyReLU(0.5)^2**: replaces ReLU^2 inside the MLP.
4. **Default `BIGRAM_VOCAB_SIZE=3072`**: spends a small amount of the recovered artifact budget on a larger hashed bigram table.
5. **Removed dormant late-QAT toggles** from this candidate path so the script does not carry the known `torch.compile`-sensitive fake-quant branch from earlier experiments.

## How to run or evaluate it

From this directory:

```bash
MIRROR_SHARE_MLP=1 \
LEAKY_RELU_SLOPE=0.5 \
BIGRAM_VOCAB_SIZE=3072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The rest of the defaults intentionally follow the 2026-03-22 strong pre-TTT setup: 11 layers, 512 width, 3x MLP, EMA, XSA on the last 4 layers, partial RoPE (16 dims), LN scale, GPTQ-lite export, and zstd-22 compression.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604041051_mirror-mlp-leaky/train_gpt.py
```

Outcome:

- `compileall` completed successfully for the repository baseline files and this candidate script.
- A minimal runtime smoke test was **not feasible in this workflow container** because the required runtime dependencies (`torch`, `sentencepiece`, and `flash_attn_interface`) are not installed here, and this candidate targets the existing CUDA/FlashAttention training path.

## Main expected risks or tradeoffs

- **Over-sharing risk**: mirrored layers may want different MLP subspaces even if they benefit from shared attention/context structure.
- **Quantized export complexity**: the alias-aware export path is simple, but it is still custom serialization logic and should be watched closely on a real GPU run.
- **Budget allocation risk**: this candidate uses the recovered bytes for a larger bigram table, but the best use of the saved artifact headroom may turn out to be something else (for example a wider VE path or a larger hash table).
