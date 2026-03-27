# Scheduled MTP Curriculum on the 2026-03-23 Top Stack

## Hypothesis

The current best record family already appears close to saturated on evaluation tricks, post-training quantization tweaks, and the known local-context modules. A stronger next lever is to improve **sample efficiency during the 600s training window** without paying extra artifact bytes at export.

This candidate enables **multi-token prediction (MTP)** as an auxiliary loss on top of the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` stack, but uses a **small-model-friendly curriculum**:

- start MTP with a low effective weight,
- ramp it up early,
- then decay it back to zero before the late quantization-sensitive portion of training,
- and exclude all MTP heads from the exported artifact.

The goal is to get the representation-learning benefits of MTP early, while keeping the final trained trunk aligned with the single-token, compression-aware export path that already wins in this repository.

## Why this is promising for this repository

Repository review suggests three things:

- The biggest “free” evaluation gain, sliding-window eval, is already fully exploited.
- The strongest records now come from the same 11-layer compression-aware stack family with XSA, partial RoPE, EMA/SWA, and strong export logic.
- The late-record scripts already contain **working MTP code paths**, but the reviewed runs keep them disabled with `MTP_NUM_HEADS=0`.

That makes MTP one of the cleanest open directions here: it changes the **training target**, not the exported model, and it composes naturally with the current best stack.

## Prior records and dead ends that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Key supporting lineage:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md`

Important negative evidence:

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` documents that global layer recurrence was a bad trade under the fixed 10-minute budget.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` notes that one late-QAT path became a no-op under `torch.compile`, so this candidate avoids depending on that for its main hypothesis.

## External research that informed it

Primary sources:

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`).
  The paper argues that predicting multiple future tokens with lightweight auxiliary heads can improve sample efficiency while leaving the main trunk architecture intact.

- Ansar Aynetdinov and Alan Akbik, **"Pre-Training Curriculum for Multi-Token Prediction in Language Models"** (`arXiv:2505.22757`).
  This is especially relevant here because it reports that **smaller language models benefit more from curriculum-style MTP scheduling** than from naive always-on MTP.

- Xiaohao Liu et al., **"L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models"** (`arXiv:2505.17505`).
  I am not implementing leap prediction here, but it reinforces the broader theme that richer future-token supervision can improve efficiency without changing the exported autoregressive core.

## What changed versus the chosen base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes the following focused changes:

1. **Actually turns on MTP by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.10`

2. **Adds a small-model-friendly MTP curriculum**
   - `MTP_WARMUP_FRAC=0.10`
   - `MTP_END_FRAC=0.75`
   - `MTP_DECAY_FRAC=0.10`

   The effective MTP loss weight:
   - ramps up over the first 10% of training,
   - stays active through the middle of the run,
   - then decays to zero by 75% progress.

3. **Makes the MTP scale compile-safe and step-dynamic**
   - The training loop passes a per-step scalar into `forward(...)` rather than relying on a static Python attribute.
   - That keeps the schedule explicit and avoids tying the candidate to a fragile compile-time constant path.

4. **Keeps export bytes unchanged in spirit**
   - MTP heads are still excluded from `export_sd`, exactly as in the base script.
   - The final artifact remains the normal single-token model plus quantized weights. EMA remains inherited from the base script; this candidate does not add a new EMA toggle.

5. **Adds a synthetic CPU smoke path**
   - `SMOKE_TEST=1` instantiates a tiny model, runs one forward/backward step, quantizes/dequantizes the banked weights, reloads them into an eval model, and checks `forward_logits` shape.
   - This is only for local validation, not leaderboard evaluation.

6. **Adds an attention fallback for environments without FlashAttention 3**
   - CUDA+FA3 still uses the original fast path.
   - Otherwise the script falls back to `torch.nn.functional.scaled_dot_product_attention` and explicitly expands grouped K/V heads when GQA is active, which keeps the smoke path aligned with the default 8Q/4KV architecture.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603271424_mtp-curriculum

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 MTP_WARMUP_FRAC=0.10 MTP_END_FRAC=0.75 MTP_DECAY_FRAC=0.10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a dependency-light local smoke check in an environment that already has PyTorch installed:

```bash
cd candidates/202603271424_mtp-curriculum
SMOKE_TEST=1 python train_gpt.py
```

## Validation run for this candidate

Commands run in this workflow runner:

```bash
python -m compileall candidates/202603271424_mtp-curriculum/train_gpt.py
SMOKE_TEST=1 python candidates/202603271424_mtp-curriculum/train_gpt.py
```

Outcomes:

- `python -m compileall ...` **passed**.
- The synthetic smoke test could **not** be completed in this runner because the environment does not have `torch` installed. The script now includes a smoke path specifically for environments where PyTorch is available.

## Main expected risks and tradeoffs

- **Tiny-model MTP can be unstable** if the auxiliary weight is too high or stays active too long. That is why this candidate uses a curriculum instead of always-on MTP.
- **Training-time overhead is not free.** Even with one extra head, the auxiliary loss adds compute and optimizer state during training.
- **The benefit may be smaller than in larger LLM papers.** The external literature is encouraging, but this repository operates at a much smaller scale and under a very tight wall-clock budget.
- **Interactions with TTT are uncertain.** MTP is intended to improve the trunk before export; the final post-TTT effect still needs real H100 validation.

## Expected next experiments if this looks promising

- Sweep `MTP_NUM_HEADS` in `{1, 2}`.
- Sweep peak loss weight in `{0.05, 0.10, 0.15}`.
- Compare always-on MTP vs this curriculum on the same 8xH100 stack.
- If one-head MTP helps, try enabling it only on the pre-TTT stack first to isolate whether the gain survives export before paying the full TTT eval cost.
