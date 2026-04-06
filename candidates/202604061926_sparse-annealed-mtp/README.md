# Sparse Annealed MTP on the 2026-03-23 SOTA Stack

## Hypothesis

The current best stack is already close to saturating the repo's usual low-cost wins (XSA, partial RoPE, LN scaling, EMA/SWA, GPTQ-lite, legal TTT, LeakyReLU²). The next promising lever is **sample efficiency per training step** rather than another artifact-compression tweak. This candidate tests whether **training-only multi-token prediction (MTP)** can make the 11-layer trunk learn more from the same 10-minute budget, while **sparsifying** the auxiliary loss and **annealing it away during warmdown** so the final phase stays focused on next-token quality and quantization-friendly averaging.

## Why this is promising here

Repository review across `records/` showed a few clear patterns:

- Repeated wins came from **cheap inductive biases** and **quantization-aware training/export**, not from adding large new infrastructure.
- The strongest recent lineage is `2026-03-20` XSA4 + EMA -> `2026-03-21` partial RoPE + LN scale -> `2026-03-22` GPTQ-lite + warmdown3500 -> `2026-03-23` LeakyReLU² + legal TTT + Parallel Muon.
- Negative results matter too: **depth recurrence/layer looping regressed**, **SwiGLU often lost on wallclock throughput**, and early **full QAT overhead** was hard to justify.

That makes MTP a better fit than another recurrent/depth-reuse experiment here: it targets **learning efficiency** without changing the exported trunk or repeating a dead end the repo already explored.

## Records and prior work that informed this candidate

### Most influential repository runs

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - Chosen base implementation.
   - Best current score and already includes the strongest known recipe in this repo.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - Confirms EMA + GPTQ-lite + warmdown3500 is the best non-TTT base.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - Partial RoPE + LN scale stayed in the stack.
4. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
   - XSA4 + EMA established the modern 11L branch.
5. Negative-result references:
   - `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
   - `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
   - These are why this candidate does **not** pursue recurrence or SwiGLU.

No prior `candidates/` directory existed when this candidate was created.

## External research

Primary sources that motivated the idea:

1. Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** (2024-04-30)  
   https://arxiv.org/abs/2404.19737
2. Somesh Mehra et al., **On multi-token prediction for efficient LLM inference** (2025-02-13)  
   https://arxiv.org/abs/2502.09419
3. Anastasios Gerontopoulos et al., **Multi-Token Prediction Needs Registers** (2025-05-15)  
   https://arxiv.org/abs/2505.10518
4. John Kirchenbauer et al., **Multi-Token Prediction via Self-Distillation** (2026-02-05)  
   https://arxiv.org/abs/2602.06019

Why these papers point in the same direction:

- MTP repeatedly shows up as a way to improve **sample efficiency** or future-token reasoning.
- The repo's constraint is not just artifact size; it is also **what improves BPB under a fixed wallclock**.
- A training-only MTP head is especially attractive here because it can be **excluded from export**, so the artifact cost is mostly code bytes rather than model bytes.

I did **not** implement register-token MTP or self-distilled standalone MTP decoding here because those require broader architectural or evaluation changes than this repo's current scripts benefit from.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes four targeted changes:

1. **Enable training-only MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.2`
   - Export still strips `mtp_heads` before serialization and quantization.
2. **Sparse auxiliary supervision**
   - `MTP_STRIDE=2`
   - The extra heads only score every other valid position, reducing wallclock overhead relative to dense MTP.
3. **Warmdown annealing for the auxiliary loss**
   - Full weight while LR scale is high.
   - Linearly fade from `MTP_FADE_FROM_SCALE=0.35` to `MTP_FADE_TO_SCALE=0.15`.
   - Zero auxiliary weight in the lowest-LR tail so the last phase remains next-token-focused.
4. **FlashAttention fallback for importability**
   - If `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA.
   - This is mainly for lightweight local smoke imports; `main()` still fails fast unless `flash_attn_interface` is present for the real CUDA path.

## Expected artifact / runtime effect

- **Artifact size:** the new MTP heads are **not exported**, so the main artifact effect should be a small code-size increase only.
- **Training throughput:** slower than the pure base because of auxiliary logits, but the stride-2 sparsification is intended to keep that overhead modest.
- **Evaluation path:** unchanged except for the import fallback; TTT, quantization, and sliding-window eval all remain the base behavior.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 MTP_STRIDE=2 \
MTP_FADE_FROM_SCALE=0.35 MTP_FADE_TO_SCALE=0.15 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The MTP defaults are already baked into this candidate script, so the `MTP_*` variables above are included mainly to make the intended configuration explicit.

## Validation

Commands run in this workspace:

1. `python -m compileall candidates/202604061926_sparse-annealed-mtp/train_gpt.py`
   - **Passed**
2. Toy CPU import/forward smoke:
   - attempted via `python - <<'PY' ... import torch ... GPT(...) ... PY`
   - **Blocked by environment**: this container does not currently have `torch` installed, even though `requirements.txt` lists it.

Why there is no full runtime smoke here:

- `main()` requires CUDA plus the repo runtime stack.
- This shared runner is missing `torch`, so a real start-up smoke without installing heavyweight dependencies was not feasible.

## Main risks / tradeoffs

1. **Throughput tradeoff**: even sparse MTP may still reduce total steps enough to erase any sample-efficiency gain.
2. **Late-phase objective mismatch**: if the fade window is wrong, auxiliary prediction could still interfere with quantization-sensitive warmdown behavior.
3. **Tiny-model regime uncertainty**: MTP evidence is strongest at larger scales; the repo's small model may get a weaker benefit.
4. **Interaction with legal TTT**: better representations could help TTT, but there is also a chance the training objective shifts the model toward a slightly worse next-token optimum before adaptation.

## Suggested next experiments if this shows promise

1. Sweep `MTP_NUM_HEADS` in `{1, 2}`.
2. Sweep `MTP_STRIDE` in `{1, 2, 4}` to trade sample efficiency against steps/sec.
3. Keep dense MTP early, then disable it even earlier in warmdown if throughput or final quantized BPB degrades.
