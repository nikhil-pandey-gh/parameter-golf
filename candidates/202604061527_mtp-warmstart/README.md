# Candidate: Warm-started auxiliary MTP on the 11L EMA + GPTQ-lite stack

## Hypothesis

The strongest non-TTT training stack in this repository is already near the artifact limit, so the next cheap win is likely **sample efficiency**, not another large architectural addition. Multi-token prediction (MTP) is a good fit because it adds training signal to the existing trunk, can be kept **training-only**, and does not need to change the exported single-head artifact.

My concrete bet is that this repo's 11-layer stack already learns hidden states that are useful for short lookahead, so:

1. enabling **2 auxiliary MTP heads**,
2. **warm-starting** them from the main tied output head / embedding geometry, and
3. **downweighting farther horizons** with a geometric decay

should improve fixed-budget training efficiency without paying extra artifact bytes.

## Why this is promising here

- The current best non-TTT record is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which already has most of the repo's strongest training/export tricks stacked together.
- The 2026-03-21, 2026-03-22, and 2026-03-23 record scripts already support `mtp_heads` during training **and** explicitly export/evaluate with `mtp_num_heads=0`, so this codebase already has the right artifact-safe pattern for auxiliary MTP.
- The current leaderboard trend is that compression-aware export and eval are mature, while fixed-budget training quality is still an open lever.

There were **no prior `candidates/` directories** in the repository at review time, so this does not repeat an earlier candidate iteration.

## Prior repository work that influenced this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Strongest non-TTT training stack reviewed.
  - Already includes EMA, GPTQ-lite clip search, 11L, MLP3x, partial RoPE, XSA, SmearGate, BigramHash, VE128, and artifact-safe MTP export.
- **Supporting record lineage:** `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` and `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - Useful for confirming that auxiliary `mtp_heads` are intentionally dropped from the exported artifact.
  - Show that recent gains keep coming from better training/eval leverage rather than brand-new infrastructure.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737)  
  MTP improves sample efficiency by supervising multiple future tokens from a shared trunk.
- Raghavv Goel et al., **"Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing"** (arXiv:2603.17942)  
  Motivates the warm-start: decoder hidden states already align with multi-token prediction directions in embedding space.
- John Kirchenbauer et al., **"Multi-Token Prediction via Self-Distillation"** (arXiv:2602.06019)  
  Reinforces the goal of keeping the final model deployable without a special auxiliary inference pipeline.

I also considered more invasive lookahead-style training from Lorenzo Noci et al., **"Thinking into the Future: Latent Lookahead Training for Transformers"** (arXiv:2603.20219), but it looked too infrastructure-heavy for a minimal candidate in this repo.

## What changed vs the 2026-03-22 base

1. **Default MTP is enabled:** `MTP_NUM_HEADS=2`, `MTP_LOSS_WEIGHT=0.15`.
2. **New horizon decay:** `MTP_HORIZON_DECAY=0.5`, so the +1-token auxiliary head matters more than the +2-token head.
3. **New MTP warm-start:** `MTP_INIT_FROM_MAIN_HEAD=1`, which copies the main tied output geometry into each auxiliary MTP head instead of learning from zero.
4. **Candidate usability fix:** default dataset/tokenizer paths resolve from the repository root via `__file__`, so the script can be run from inside this candidate directory or from elsewhere in the repo without resetting those paths.
5. **Artifact behavior stays unchanged:** auxiliary `mtp_heads` are still excluded from `export_sd`, and roundtrip eval still instantiates `GPT(..., mtp_num_heads=0, mtp_loss_weight=0.0)`.

## How to run

From this candidate directory:

```bash
cd candidates/202604061527_mtp-warmstart
RUN_ID=mtp_warmstart \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides if you want to ablate the idea:

```bash
MTP_NUM_HEADS=0                 # disable the candidate idea entirely
MTP_LOSS_WEIGHT=0.10            # softer auxiliary loss
MTP_HORIZON_DECAY=0.25          # even more emphasis on the first lookahead token
MTP_INIT_FROM_MAIN_HEAD=0       # revert to zero-init auxiliary heads
```

If your dataset/tokenizer live in the repository's standard `data/` layout, the defaults should work whether you launch from this directory or elsewhere. Override `DATA_PATH` / `TOKENIZER_PATH` only when you want a non-default location.

## Validation

- From the repository root: `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604061527_mtp-warmstart/train_gpt.py`  
  **Passed**.
- Minimal CPU runtime smoke was **not feasible in this workflow environment**:
  - `python` had no local `torch` module available,
  - `sentencepiece` was also absent,
  - and the real runtime path expects CUDA + FlashAttention anyway.

## Main risks / tradeoffs

- MTP may improve sample efficiency but still hurt final next-token loss if the auxiliary weight is too high.
- Warm-starting from the main output head could reduce useful diversity across MTP heads if the heads need more horizon-specific specialization.
- Even though artifact size is unaffected, training-time FLOPs still increase.
- This candidate is intentionally conservative: it does **not** also stack the idea onto the full legal-TTT 2026-03-23 pipeline, so the first result should be easier to interpret but may leave some upside on the table.
