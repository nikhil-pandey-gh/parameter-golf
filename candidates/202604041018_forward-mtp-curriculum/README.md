# 202604041018 Forward MTP Curriculum

## Hypothesis

The current best stack is already close to saturated on architecture and export tricks, so the highest-leverage missing idea is a **training-time-only auxiliary objective** that improves sample efficiency without adding bytes to the shipped artifact. This candidate adds a **forward multi-token-prediction (MTP) curriculum** on top of the current best record: start as plain next-token prediction, then unlock extra future-token heads over the course of the 600s run.

The bet is that this repository's 11-layer, 16MB-constrained model behaves much more like a small language model than the 7B+ models in early MTP work, so a **curriculum** matters more than naive always-on MTP.

## Why this is promising here

- The strongest record lineage keeps winning by stacking many small efficiency gains on the same 11L/512d backbone rather than replacing the backbone outright:
  - `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`
  - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- Repository review showed **no existing candidate directory** and no record README built around MTP despite the record scripts already carrying dormant MTP scaffolding.
- MTP is unusually attractive for Parameter Golf because the extra heads are **training-only** and can be **excluded from export**, so the artifact budget is unaffected while training gets a denser supervisory signal.
- The main alternatives looked less attractive under this repo's constraints:
  - recurrence/depth reuse was already a negative result in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - late-QAT toggles were explicitly fragile in `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - more invasive MTP variants like register tokens would require larger sequence/architecture changes than this repo currently uses.

## Prior experiments that influenced this candidate

### Chosen base

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- Why this base:
  - best recorded score in this repo review
  - already includes parameter banking, LeakyReLU^2, partial RoPE, XSA, VE, EMA/SWA, and legal TTT
  - already excludes `mtp_heads` from export, making MTP a natural training-only extension

### Relevant trends from records

- **Keep the frontier backbone intact:** 11 layers, MLP3x, BigramHash/SmearGate family, partial RoPE, XSA on deep layers, EMA/SWA, aggressive low-bit export.
- **Evaluation matters a lot:** sliding-window eval and legal TTT remain important.
- **Dead ends / caution flags:** recurrence and large architecture pivots have not looked compelling under the 10-minute budget.

## External research that informed the choice

1. **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737)**  
   <https://arxiv.org/abs/2404.19737>  
   MTP improves sample efficiency by predicting multiple future tokens with independent heads on a shared trunk.

2. **Ansar Aynetdinov and Alan Akbik, "Pre-Training Curriculum for Multi-Token Prediction in Language Models" (arXiv:2505.22757)**  
   <https://arxiv.org/abs/2505.22757>  
   The most relevant paper for this repo: smaller language models benefit from a **forward curriculum** that gradually increases the number of active MTP heads instead of training with full MTP from the start.

3. **Anastasios Gerontopoulos et al., "Multi-Token Prediction Needs Registers" (arXiv:2505.10518)**  
   <https://arxiv.org/abs/2505.10518>  
   Confirms the broader MTP direction is active, but its register-token approach is more invasive than we want for this repo, so this candidate keeps the simpler shared-trunk multi-head formulation.

## What changed vs. the chosen base implementation

1. **Enable MTP by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`.
2. Add `MTP_CURRICULUM` with:
   - `forward` (default): unlock extra heads progressively
   - `none`: train with all configured MTP heads active from the start
3. Add a non-persistent `mtp_head_weights` buffer plus `set_mtp_active_heads()` so the compiled model can keep a fixed shape while the active auxiliary heads change over time.
4. Make the curriculum **wallclock-aware**:
   - with the challenge's 600s cap, the first extra head activates at ~1/3 of the run
   - the second extra head activates at ~2/3 of the run
   - if no wallclock cap is set, the schedule falls back to iteration progress
5. Change the auxiliary loss aggregation so only the currently active MTP heads contribute to the average loss.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_CURRICULUM=forward \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `mtp_heads` are still **excluded from `final_model.pt` and the quantized artifact**, so this remains a training-only change.
- For ablations, set `MTP_CURRICULUM=none` to compare against always-on MTP with the same number of heads.
- This base script keeps **EMA always on** with decay `0.997`; the configurable averaging knob here is `SWA_ENABLED`.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202604041018_forward-mtp-curriculum/train_gpt.py
```

Outcome:

- `compileall` passed.
- A CPU-only runtime smoke test was **not feasible** in this environment because the submission script hard-requires CUDA and imports FlashAttention 3 at runtime, matching the leaderboard-oriented GPU path used by the record scripts.

## Main risks and tradeoffs

- Even with export-safe heads, MTP still adds training compute; if step count drops too much, the auxiliary signal may not pay for itself.
- The implementation keeps the compiled graph shape fixed, so zero-weight inactive heads still incur some compute before they are activated.
- MTP may interact with legal TTT, EMA/SWA, and late-stage quantization in non-obvious ways; the first follow-up experiments should be:
  1. `MTP_CURRICULUM=forward` vs `none`
  2. `MTP_NUM_HEADS=1` vs `2`
  3. `MTP_LOSS_WEIGHT=0.10`, `0.15`, `0.20`
