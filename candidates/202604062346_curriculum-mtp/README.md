# Curriculum MTP on the 11L banked stack

## Hypothesis

A small-model-friendly multi-token-prediction (MTP) curriculum can improve final next-token BPB on the current 11-layer banked/XSA/LeakyReLU^2 stack without increasing artifact size.

The key bet is that MTP should be treated as a **train-only auxiliary objective**, not an always-on replacement for next-token prediction:

- start with pure next-token prediction,
- ramp in a small 2-head MTP objective during the middle of training,
- anneal it back out before the final warmdown finishes.

That keeps the late-stage optimization aligned with the actual evaluation target while still using MTP to shape better hidden states earlier in the run.

## Why this is promising for this repository

The public record line has already squeezed a lot out of quantization, export, and evaluation:

- mixed int5/int6/int8 export,
- GPTQ-lite clip search,
- EMA/SWA,
- partial RoPE,
- XSA on deep layers,
- SmearGate + BigramHash,
- legal score-first TTT.

Recent top scripts already carry dormant MTP plumbing, but the public record configs still leave it off. That makes MTP a good next candidate here: it is **minimally invasive**, **artifact-free at export time**, and directly aimed at representation/sample-efficiency rather than another tiny export tweak.

## Prior repo influences

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Immediate stack predecessors:**  
  `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
  `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Earlier architectural wins carried forward:**  
  `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`  
  `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/`

There was **no existing `candidates/` directory** when this candidate was created, so there were no prior candidate iterations to inherit from.

## External research

| Source | Why it matters here |
|---|---|
| Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** (2024), https://arxiv.org/abs/2404.19737 | Establishes MTP as a useful auxiliary objective that improves sample efficiency and downstream capability. |
| Ansar Aynetdinov et al., **Pre-Training Curriculum for Multi-Token Prediction in Language Models** (ACL 2025), https://arxiv.org/abs/2505.22757 | Most relevant paper for this repo: small language models struggle with naive always-on MTP, and curriculum learning helps them benefit from it. |
| Anastasios Gerontopoulos et al., **Multi-Token Prediction Needs Registers** (2025), https://arxiv.org/abs/2505.10518 | Useful cautionary signal that naive MTP can be brittle; this candidate stays conservative (2 heads, scheduled on/off) instead of forcing a large always-on horizon. |

## What changed vs. the chosen base implementation

Starting point: the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` record script.

This candidate adds five focused changes:

1. **Turns on MTP by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`.
2. **Adds a 3-phase curriculum** controlled by:
   - `MTP_START_FRAC` (default `0.10`)
   - `MTP_FULL_FRAC` (default `0.35`)
   - `MTP_END_FRAC` (default `0.75`)
3. **Buckets the runtime MTP weight** with `MTP_WEIGHT_BUCKETS=8`, so `torch.compile(fullgraph=True)` only sees a small discrete set of auxiliary-loss states instead of a new scalar every step.
4. **Wires the curriculum through explicit runtime forward arguments**, so the compiled model can specialize on a small discrete set of states (0-head, 1-head, 2-head) instead of hiding the schedule behind a static config flag.
5. **Explicitly optimizes the MTP heads during training** while still excluding them from export, so the auxiliary objective costs train compute but not artifact bytes.
6. **Adds a lightweight smoke path** (`SMOKE_TEST=1`) plus a FlashAttention fallback so the candidate can be started and locally sanity-checked without a full challenge environment.

## How to run / evaluate

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 \
MTP_START_FRAC=0.10 MTP_FULL_FRAC=0.35 MTP_END_FRAC=0.75 MTP_WEIGHT_BUCKETS=8 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- This keeps the **base 11L banked / XSA / partial-RoPE / VE / LeakyReLU^2** stack intact and changes only the training objective scheduling.
- The script still supports the inherited **legal score-first TTT** path if you want to layer it on top during evaluation with `TTT_ENABLED=1`.
- MTP heads are **not exported** into the final artifact.

## Main expected risks / tradeoffs

- **Extra training compute:** MTP heads add logits + CE work during training, so a fixed 10-minute run may complete fewer steps.
- **Auxiliary-objective mismatch:** exported artifacts do not include MTP heads, so the curriculum has to improve the shared trunk rather than relying on train-only capacity.
- **Schedule sensitivity:** small models can regress if MTP turns on too hard or stays on too long; the default schedule is intentionally conservative.
- **Interaction risk with EMA / warmdown / QAT timing:** even a good auxiliary objective can hurt if it shifts the late-training regime too much.

## Validation

Validated in this runner with lightweight checks only:

1. `python -m compileall train_gpt.py` — **passed**
2. `SMOKE_TEST=1 SMOKE_STEPS=1 TRAIN_LOG_EVERY=1 python train_gpt.py` — **passed**
   - exercises a compiled **MTP-off** step, a compiled **MTP-on** step, and a minimal **export/reload** sanity check
   - observed: `smoke_test:ok logits_shape:(1, 64, 1024) export_reload:ok`

What was **not** validated here:

- full data-path training,
- full validation-set evaluation,
- GPU / multi-GPU throughput,
- artifact size on a real run.

Reason: this runner does not have the challenge dataset/tokenizer cache under `data/datasets/` and `data/tokenizers/`, and it does not expose CUDA hardware.
