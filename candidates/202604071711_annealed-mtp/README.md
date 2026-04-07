# Annealed Multi-Token Prediction on the 2026-03-23 frontier stack

## Hypothesis

The strongest unexplored low-infrastructure idea in this repo is a **training-only multi-token prediction (MTP) auxiliary loss** on top of the current best stack. The shared trunk should learn slightly better short-horizon representations early in training, while an **annealed warmdown schedule** turns the auxiliary off before the final next-token / quantization-sensitive phase.

## Why this is promising for this repository

- The repo frontier already squeezed a lot out of **sliding-window eval, int6/int5 compression, GPTQ-lite, EMA/SWA, XSA, partial RoPE, SmearGate, BigramHash, and legal TTT**. Another pure quantization tweak looked lower-upside than an unused training objective.
- The best recent scripts already contain dormant MTP support, but the record logs still show **`mtp_num_heads:0`**, so this is a real unexplored axis in-repo rather than a repeated record.
- MTP is especially attractive here because the extra heads are **training-only** and are already excluded from export in the base script, so the candidate can chase better sample efficiency with **zero extra artifact bytes**.
- I did **not** choose cross-layer recurrence/sharing as the first next candidate even though external research makes it tempting, because the repo's own non-record sweep explicitly reports recurrence as a bad fit for this regime.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current stack,
  - LeakyReLU^2 MLP,
  - legal score-first TTT,
  - parameter banks + parallel Muon,
  - GPTQ-lite int6 + lzma export.
- **Direct architectural predecessors:**
  - `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
- **Earlier enabling ideas that remain intact here:**
  - `records/track_10min_16mb/2026-03-19_SlidingWindowEval/`
  - `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/`
  - `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`

## External research that informed it

- **Gloeckle et al., 2024 — "Better & Faster Large Language Models via Multi-token Prediction"**  
  https://arxiv.org/abs/2404.19737  
  Main takeaway used here: MTP can improve sample efficiency by predicting several future tokens with lightweight extra heads on a shared trunk.
- **Mehra et al., 2025 — "On multi-token prediction for efficient LLM inference"**  
  https://arxiv.org/abs/2502.09419  
  Main takeaway used here: MTP is helpful, but hidden states are strongly specialized for next-token prediction, so aggressive or late auxiliary pressure can be counterproductive.
- **Gerontopoulos et al., 2025 — "Multi-Token Prediction Needs Registers"**  
  https://arxiv.org/abs/2505.10518  
  Main takeaway used here: richer MTP variants likely want architectural help, so this candidate stays conservative: simple shared-trunk heads plus late annealing instead of invasive register-token changes.

## What changed versus the chosen base implementation

1. **Enabled MTP by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`.
2. Added a new **`MTP_DECAY_END_SCALE`** hyperparameter (default `0.2`).
3. Added a small runtime buffer `mtp_aux_scale` plus `set_mtp_aux_scale()` so the MTP loss is:
   - full strength through most of training,
   - then linearly annealed toward zero as the main LR warmdown scale drops below `0.2`.
4. Logged the active MTP scale during training so ablations can tell whether late warmdown is still under auxiliary pressure.
5. Kept the existing export behavior where **MTP heads are excluded from the final artifact**, so the candidate remains artifact-budget neutral.

## How to run / evaluate

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_DECAY_END_SCALE=0.2 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to isolate the training-only effect before paying the legal-TTT evaluation cost, set `TTT_ENABLED=0`.

## Validation

- `python -m compileall candidates/202604071711_annealed-mtp/train_gpt.py` **passed**.
- I attempted a tiny CPU smoke test via dynamic import plus a flash-attention stub, but this runner does not have the repo's core Python dependencies installed (`torch`, `numpy`, and `sentencepiece` were all missing), so a meaningful local execution test was **not feasible on this runner**.

## Main expected risks / tradeoffs

- **Extra output-head compute** may reduce steps within the 600-second cap if the MTP heads cost more than the added supervision helps.
- The 2025 MTP follow-up work suggests hidden states are still strongly next-token-specialized, so too much auxiliary weight can hurt the final objective.
- The stronger register-token MTP variants from recent work were intentionally **not** implemented here to keep the candidate self-contained and low-risk.
- Because the final export drops the MTP heads, the entire bet is that the shared trunk improves enough during training to offset the auxiliary mismatch.
