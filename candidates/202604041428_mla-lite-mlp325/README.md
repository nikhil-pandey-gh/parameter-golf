# MLA-lite latent K/V + MLP 3.25x

## Hypothesis

The best unexplored trade in this repo is to compress **all-layer key/value projection weights** instead of spending more bytes on another quantization-only tweak. Replacing each attention block's full-rank K/V projections with a small shared latent bottleneck should free enough parameters and export bytes to slightly widen the MLP, while keeping the proven 11-layer seq2048 recipe intact.

## Why this looks promising here

- The archive shows that recent gains mostly came from **compression-aware capacity reallocation**: MLP 3x, deeper 11-layer stacks, smarter quantization, and evaluation-aware recipes all beat the root starter by large margins.
- The repo already uses **GQA**, **partial RoPE**, **XSA**, **EMA/SWA**, and **GPTQ-lite/mixed int6** in the strongest non-TTT training stack, but it does **not** yet compress K/V projections across all layers.
- The records review also turned up a clear dead end: **layer recurrence** regressed, so parameter savings should be reinvested through a cheaper structural change than depth reuse.

## Prior records that shaped this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen as the main base because it is the strongest pre-TTT training stack in the repo.
  - Reused: 11L/512d seq2048 scaffold, BigramHash, XSA, partial RoPE, LN scaling, EMA, and the mixed-int6 export path.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Contributed the proven **LeakyReLU(0.5)^2** activation change.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - Reinforced the core lesson that widening the MLP is still one of the highest-value uses of recovered bytes.
- `train_gpt.py`
  - Used as the reference for a self-contained SDPA attention path and repo-conventional export/eval structure.

## External research that informed it

1. **DeepSeek-V2** — Multi-head Latent Attention shows that compressing KV representations can buy large efficiency gains with limited quality loss.  
   <https://arxiv.org/abs/2405.04434>
2. **Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs** — partial-RoPE plus low-rank K/V conversion can preserve quality with little adaptation data.  
   <https://arxiv.org/abs/2502.14837>
3. **CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling Multi-Head Latent Attention** — rank choice matters, which motivates exposing `KV_LATENT_DIM` for future sweeps.  
   <https://arxiv.org/abs/2603.17946>
4. **Thin Keys, Full Values** — attention selection appears to need much less dimensionality than value transfer, supporting K/V bottleneck experiments in tiny models.  
   <https://arxiv.org/abs/2603.04427>
5. **Stable Language Model Pre-training by Reducing Embedding Variability** — low-rank attention can improve stability, especially in deeper models.  
   <https://arxiv.org/abs/2409.07787>

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **MLA-lite latent K/V bottleneck**
   - Replaced full-rank `c_k` and `c_v` with:
     - `kv_down: model_dim -> kv_latent_dim`
     - `k_up: kv_latent_dim -> kv_dim`
     - `v_up: kv_latent_dim -> kv_dim`
   - Default `KV_LATENT_DIM=96`.
2. **Slightly wider MLP**
   - Increased default `MLP_MULT` from `3.0` to `3.25`, spending part of the recovered K/V budget on extra feed-forward capacity.
3. **LeakyReLU(0.5)^2**
   - Swapped the base ReLU^2 MLP activation for the stronger recent LeakyReLU^2 variant.
4. **PyTorch SDPA instead of `flash_attn_interface`**
   - Keeps the script self-contained and importable from the candidate directory without an extra FlashAttention Python module.
5. **Robust repo-root discovery**
   - `DATA_PATH` and `TOKENIZER_PATH` now resolve by searching upward for the repository root, so the script keeps working from `candidates/` and future archival locations.
6. **Native seq-length RoPE anchor**
   - The copied base script hardcoded a `1024` RoPE anchor; this candidate now threads `TRAIN_SEQ_LEN` into Rotary so the default seq2048 recipe uses a native 2048 anchor.
7. **Pinned zlib export compression**
   - Export compression is fixed to zlib for reproducible artifact behavior across environments instead of switching based on optional `zstandard` availability.

## Why it differs from existing records

- No archived record in this repo applies **all-layer latent K/V compression**.
- The closest in spirit is the shared late-layer value embedding from the March 22 record, but that only augments values in selected deep layers; this candidate compresses the actual **K/V projection path in every layer**.
- This is also intentionally different from the recurrence route that already failed in the non-record exploration.

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs for follow-up sweeps:

- `KV_LATENT_DIM=64|96|128`
- `MLP_MULT=3.0|3.25|3.5`
- `BIGRAM_VOCAB_SIZE=2048|3072`
- `EVAL_STRIDE=64`
- `REPO_ROOT=/absolute/path/to/repo` if automatic repo-root discovery is ever ambiguous

## Validation

Commands run during implementation:

```bash
python -m compileall candidates/202604041428_mla-lite-mlp325/train_gpt.py
python -m compileall train_gpt.py train_gpt_mlx.py data
```

Outcomes:

- `python -m compileall candidates/202604041428_mla-lite-mlp325/train_gpt.py` **passed**.
- Baseline repo compileall on `train_gpt.py`, `train_gpt_mlx.py`, and `data/` **passed**.
- A minimal CPU forward smoke test was **not feasible in this runner** because `/usr/bin/python` does not currently have the repo's `torch` dependency installed, so the candidate could not be imported for an actual tensor pass without first bootstrapping the full training environment.

## Main risks / tradeoffs

- `KV_LATENT_DIM=96` may be too aggressive for some layers; the obvious next sweep is a rank scan or a shallow/mid/deep schedule.
- Replacing `flash_attn_interface` with PyTorch SDPA keeps the script simpler, but it may shift kernel selection and slightly change throughput on H100-class GPUs.
- Pinning compression to zlib improves reproducibility, but it may leave some artifact bytes on the table versus zstd-based record scripts.
- The latent K/V factors are new matrices that will be quantized by the existing mixed-int6 export path; quantization behavior may improve or regress depending on how smooth the learned bottleneck becomes.
- This candidate does **not** port the March 23 legal TTT / parameter-banking path yet. If the core training stack improves, that is the highest-value place to layer it next.
