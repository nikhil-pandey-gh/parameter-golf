# GQA-2 + VE Rescue

## Hypothesis

The recent winning stack has already squeezed a lot out of 11 layers, XSA, partial RoPE, EMA/GPTQ-lite style compression, and local lexical priors. One thing it has **not** explored is reducing the number of KV heads below 4.

This candidate tests a simple version of that idea: move from **4 KV heads to 2 KV heads globally**, then spend some of the recovered attention capacity/bytes on slightly stronger token-identity rescue paths:

- larger **BigramHash** defaults (`2048 -> 3072`)
- wider shared **Value Embedding** (`128 -> 160`)
- broader VE placement (`layers 9,10 -> 8,9,10`)

The core bet is that the repo's existing VE + bigram machinery can compensate for the information bottleneck introduced by fewer KV heads, while lower KV bandwidth/parameter count helps the model fit the challenge's tight 16MB artifact limit more comfortably.

## Why this is promising for this repository

The records show a very clear pattern:

- **compression-aware capacity** wins
- **small architectural biases** win
- **sliding-window eval** and later **legal TTT** add extra BPB gains on top

The best recent models already use:

- 11 layers / 512 width / 3x MLP
- XSA on deep layers
- partial RoPE + LN scaling
- EMA / SWA / GPTQ-lite style post-training compression
- SmearGate + BigramHash
- shared Value Embedding on late layers

What is still relatively untouched is **more aggressive KV compression inside attention itself**. Research on MQA/GQA suggests this is often a good trade when compensated carefully, and the repository already has the exact kind of rescue mechanisms that should help: lexical priors and token-identity injection on value paths.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest recent base stack
  - showed the current best recipe for parameter banking, XSA, VE, and legal TTT

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - especially important because it highlights **VE on top layers** as part of the pre-TTT architecture

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - partial RoPE + LN scaling are carried forward unchanged

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - evidence that lightweight attention-side structural tweaks can matter

- `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/`
  - motivated continuing to lean on **SmearGate + BigramHash** as rescue paths for a more compressed attention core

There were no pre-existing `candidates/` directories when this candidate was created.

## External research that informed it

- Noam Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*  
  <https://arxiv.org/abs/1911.02150>

- Joshua Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*  
  <https://arxiv.org/abs/2305.13245>

- DeepSeek-AI, *DeepSeek-V2*  
  <https://arxiv.org/abs/2405.04434>

The candidate deliberately chooses the lower-risk end of that spectrum:

- not a full MLA rewrite
- not a new attention kernel
- just a stronger GQA compression setting (`NUM_KV_HEADS=2`) on top of the repo's best recent stack

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Default KV head count** changed from `4` to `2`.
2. **Default BigramHash size** changed from `2048` to `3072`.
3. **Default shared VE width** changed from `128` to `160`.
4. **Default VE layers** changed from `9,10` to `8,9,10`.
5. Script defaults now resolve dataset/tokenizer paths relative to the **repository root**, so the script can be run directly from this candidate directory.
6. `flash_attn_interface` import is now optional; if it is unavailable, the code falls back to PyTorch SDPA in attention.
7. Added a small `SMOKE_TEST=1` path intended for tiny local shape/gradient checks when `torch` is available.

Everything else stays intentionally close to the strongest recent stack.

## How to run

From this candidate directory:

```bash
cd candidates/202603300859_gqa2-ve-rescue
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Compare against the old KV setting on the same script
NUM_KV_HEADS=4 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Tighten or loosen the rescue path
VE_DIM=192 VE_LAYERS=8,9,10 BIGRAM_VOCAB_SIZE=4096 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to layer eval-side adaptation back on:

```bash
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
  TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
  SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation run for this candidate

I ran the lowest-cost checks available in this runner:

- `python -m compileall candidates/202603300859_gqa2-ve-rescue/train_gpt.py`
  - **passed**

- `SMOKE_TEST=1 python train_gpt.py`
  - **not feasible in this runner**
  - the container's system Python does not have `torch` installed, so even a CPU-only micro-run cannot execute here

For baseline sanity earlier in the workflow, I also ran:

- `python -m compileall train_gpt.py train_gpt_mlx.py data`
  - **passed**

## Expected risks / tradeoffs

- **Main risk:** `NUM_KV_HEADS=2` may be too aggressive globally for a tiny model, especially if the VE rescue is not strong enough.
- If quality drops too much, the right follow-up is probably **late-layer-only KV reduction** rather than reverting the idea entirely.
- The bigger BigramHash / VE defaults may recover local/token-identity signal but could also partially duplicate existing biases.
- Optional legal TTT may still be needed to see the full gain, but this candidate is meant to test the **pre-TTT architecture change first**.
