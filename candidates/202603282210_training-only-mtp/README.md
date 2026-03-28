# Candidate: Training-only multi-token prediction on the banked LeakyReLU² stack

## Hypothesis

The current repo looks increasingly **quantization-limited** rather than purely architecture-limited: the strongest non-record run (`records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3`) reached much better full-precision quality than its final compressed artifact, and the leaderboard has already harvested most of the obvious evaluation-only and compression-only wins. This candidate tests a different axis: **improve sample efficiency during training without paying artifact bytes**.

The concrete idea is to train the current best banked stack with **one auxiliary multi-token prediction (MTP) head** so each hidden state also learns to predict the token after the main next-token target. The auxiliary head is used only during training and is stripped before export, so the final artifact keeps the same deployment shape as the base recipe.

## Why it is promising for this repository

This challenge is constrained by both a **10-minute wallclock** and a **16 MB artifact budget**. Multi-token prediction is attractive because it can buy extra supervision during training while keeping the exported model unchanged.

That tradeoff is especially relevant here because:

- the best record family is already strong on evaluation tricks (`sliding window`, legal score-first `TTT`) and quantization tricks (`EMA`, `GPTQ-lite`, `lzma/zstd`, int6),
- the repo evidence suggests remaining headroom is hard to unlock through another tiny optimizer tweak alone,
- the current best banked script already contains dormant MTP support and export-time stripping, so the implementation risk is much lower than introducing a new recurrent block or rotation-aware quantization stack.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Chosen base implementation. It is the strongest current stack in the repo: banked weights, Parallel Muon, LeakyReLU(0.5)^2, legal TTT, GPTQ-lite int6, XSA, partial RoPE, EMA, and export-time MTP stripping.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Useful because its pre-banked implementation already routed MTP heads into the matrix optimizer, which exposed the missing optimizer plumbing in the banked script.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - Important evidence that this repo can become compression-limited. That makes training-only improvements more attractive than artifact-expanding ones.

There were **no prior `candidates/` directories** in the repository when this candidate was prepared.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"**  
  https://arxiv.org/abs/2404.19737
  - Motivating paper for the core hypothesis: auxiliary future-token heads can improve sample efficiency while keeping the trunk architecture intact.
- Maximilian Croci et al., **"QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs"**  
  https://arxiv.org/abs/2404.00456
- Zechun Liu et al., **"SpinQuant: LLM Quantization with Learned Rotations"**  
  https://arxiv.org/abs/2405.16406

QuaRot and SpinQuant were the strongest quantization-focused alternatives considered for this round. They likely have high upside, but they require broader architectural/export surgery than was justified for the first post-record candidate. MTP was the best fit for a precise, low-infrastructure iteration.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Actually optimize the MTP heads in the banked code path**
   - The banked record script defines `mtp_heads` and excludes them from export, but after the parameter-banking refactor those head weights are not added to any optimizer.
   - This candidate adds the MTP head matrices to a dedicated replicated output-head optimizer so the auxiliary loss is no longer a dead feature while preserving the bank-only Parallel Muon semantics for the main weight banks.

3. **Preserve zero-artifact-cost deployment**
   - `mtp_heads.*` are still removed from `export_sd`.
   - The quantized evaluation model is still rebuilt with `mtp_num_heads=0`.
   - The hypothesis is therefore strictly about better training signal, not about shipping extra parameters.

## How to run or evaluate it

From the repository root:

```bash
RUN_ID=mtp_candidate \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202603282210_training-only-mtp/train_gpt.py
```

For quicker experimentation, start with `TTT_ENABLED=0` so only the training-side MTP change is being measured.

## Main expected risks or tradeoffs

- **Training-step cost may rise.** Even one auxiliary head adds extra projection and loss work during training, so total steps in 600 seconds may fall.
- **Auxiliary-task interference is possible.** MTP may improve sample efficiency early but still harm the final next-token objective if the weighting is too high.
- **Optimizer choice may matter.** This candidate trains MTP heads with a separate head-style Adam optimizer to avoid changing Parallel Muon semantics, but Muon-with-full-matrix updates or a dedicated auxiliary-head learning rate might work better.
- **No structural quantization change yet.** If the real remaining bottleneck is still mostly post-training quantization error, rotation-aware methods may ultimately dominate this direction.

## Validation

- `python -m compileall candidates/202603282210_training-only-mtp/train_gpt.py`
  - **Passed** in this workflow environment.
- Minimal CPU-only runtime smoke test
  - **Not feasible in this workflow environment without changing the candidate itself.**
  - This runner currently lacks the Python runtime dependencies the script expects (`torch`, `sentencepiece`, `numpy` were all absent when checked), and the cached FineWeb assets were also unavailable (`data/tokenizers/fineweb_1024_bpe.model` plus `data/datasets/fineweb10B_sp1024/fineweb_{train,val}_*.bin` were not present).
  - The candidate intentionally preserves the CUDA/FlashAttention-oriented execution path from the chosen base record, so adding a special CPU fallback purely for this workflow would have changed the candidate more than intended.
