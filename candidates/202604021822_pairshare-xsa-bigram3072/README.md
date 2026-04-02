# Pair-shared 11L cores + XSA + Bigram3072

## Hypothesis

The current 11-layer Parameter Golf stack is strong enough that a distinct next step is to improve **artifact efficiency**, not just add another small quantization tweak. This candidate shares the **attention + MLP core weights across adjacent layers** while keeping **layer-specific RMSNorms, residual mix, per-layer scales, skip weights, XSA placement, and value-embedding scales** untied.

The goal is to keep the successful 11-layer training recipe and fixed compute path, but reduce unique weight bytes enough that the model can afford a slightly richer token-pair feature budget (`BIGRAM_VOCAB_SIZE=3072`) and more artifact headroom.

## Why this is promising here

- Recent wins in this repo mostly come from squeezing more quality out of the same artifact budget: mixed quantization, GPTQ-lite clip search, EMA/SWA, partial RoPE, and small attention/MLP changes.
- The repository already explored **naive recurrence** and found it too costly under the 10-minute budget, but this candidate is **not** extra recurrent depth: it keeps the same 11 forward passes and instead shares parameters across pairs of layers.
- Small models pay a high price for every extra unique matrix. Sharing the heavy attention/MLP weights while leaving cheap per-layer control parameters untied is a natural parameter-golf move.

## Prior repository work that informed it

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Strong clean pre-TTT stack with 11 layers, EMA, XSA on late layers, partial RoPE, VE, and GPTQ-lite int6 export.
- **Relevant architectural lineage:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Reinforces that the 11-layer XSA/partial-RoPE path is the best current non-TTT backbone.
- **Feature-budget clue:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - Its ablation notes that pushing BigramHash to 3072 helped on a stronger stack.
- **Negative control:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - Naive layer recurrence underperformed, which is why this candidate uses parameter sharing without adding depth.

## External research that informed it

- **ALBERT**: cross-layer parameter sharing can preserve depth while sharply reducing unique parameters, which is a strong fit for a strict artifact budget.  
  https://arxiv.org/abs/1909.11942
- **Universal Transformer**: useful as a cautionary adjacent idea; recurrence-like reuse can help, but the repo's own negative recurrence result suggests sticking to fixed-depth sharing first.  
  https://arxiv.org/abs/1807.03819

## What changed vs the base implementation

| Area | Base | This candidate |
| --- | --- | --- |
| Transformer cores | 11 fully independent blocks | 11 blocks grouped into adjacent pairs that share the attention + MLP core |
| Per-layer parameters | Untied | Still untied: norms, `resid_mix`, `attn_scale`, `mlp_scale`, skip weights, XSA placement, VE scales |
| XSA control | Stored on attention modules | Moved to the layer wrapper so paired layers can still differ |
| Export | Quantizes full `state_dict()` as-is | Deduplicates aliased shared tensors before quantized export so shared cores only count once |
| Bigram hash | 2048 buckets | 3072 buckets by default |

## Implementation notes

- `SHARED_CORE_GROUP_SIZE=2` is the default. Set it to `1` to disable sharing and recover the original per-layer-core layout.
- The sharing is implemented at the module level: each layer still has its own wrapper block, but adjacent blocks point at the same `SharedBlockCore`.
- Export dedup is required because shared-module aliases can otherwise appear multiple times in a `state_dict`-style export pipeline.

## How to run

From this directory:

```bash
RUN_ID=pairshare_xsa_bigram3072 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable sharing
SHARED_CORE_GROUP_SIZE=1 BIGRAM_VOCAB_SIZE=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep sharing but spend less of the saved budget on bigrams
BIGRAM_VOCAB_SIZE=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Executed in this workflow:

```bash
python -m compileall candidates/202604021822_pairshare-xsa-bigram3072/train_gpt.py
```

Outcome: **passed**.

A true runtime smoke test was **not feasible in this workflow environment** because:

- Python packages required by the script were not installed (`torch_installed False`, `sentencepiece_installed False`)
- the checked-in workspace did not contain the expected `data/datasets/...` or `data/tokenizers/...` artifacts

## Main risks / tradeoffs

- Sharing adjacent layers may hurt specialization enough to offset the byte savings.
- Pair boundaries do not align perfectly with the last-4-layer XSA region, so this candidate keeps XSA as a per-layer flag instead of a shared-core property.
- The export alias dedup path is new and could still need edge-case hardening if additional shared modules are introduced later.
- If the saved bytes materially exceed what Bigram3072 uses, the next follow-up should probably test where to reinvest them: larger bigram tables, wider VE, or mild MLP widening.
