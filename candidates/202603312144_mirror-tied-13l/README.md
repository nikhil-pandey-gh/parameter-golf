# Mirror-Tied 13L Shared-Core Candidate

## Hypothesis

A 13-layer logical stack with **mirror-tied heavy projection banks** can outperform the current 11-layer family at the same artifact budget by increasing effective depth **without storing 13 independent block cores**.

The key idea is ALBERT-style parameter sharing applied only to the large attention/MLP matrices. Each logical block still keeps its own:

- RMSNorms,
- residual mixing/scaling vectors,
- skip topology,
- XSA enablement,
- value-embedding usage,
- and deeper-layer stabilization via LN scaling.

So the model shares the expensive parts while preserving most of the per-layer specialization that the repo's recent records have found useful.

## Why this looks promising for this repository

The recent record line has already squeezed a lot from the 11-layer, 512d, 3x-MLP stack through better evaluation, quantization, and lightweight inductive biases:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

At the same time, the repo already shows that **naive full recurrence / repeated-depth reuse** can lose in a fixed 10-minute budget because it raises compute too much:

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`

This candidate tries a narrower version of sharing:

- keep the modern 11L record stack,
- raise the logical depth to 13 layers,
- but store only **7 unique mirrored cores** instead of 13 unique cores.

That means the candidate adds depth while still exporting **fewer heavy matrix cores than an unshared 11-layer banked model would store**.

## Prior records that influenced this candidate

No prior `candidates/` directory existed when this was created, so the main influences are the record folders.

Most relevant influences:

- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - best overall track result (`1.1194` mean post-TTT)
  - provided the implementation base: parameter banking, Parallel Muon, leaky-ReLU-squared MLP, optional legal score-first TTT, lzma export
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - best recent training-only mean (`1.1233`)
  - reinforced that the mature 11-layer architecture is the right substrate for a new structural idea
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - showed partial RoPE + LN scaling were clean, parameter-free wins
- `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
  - specifically warned against naive depth recurrence that burns too much wall-clock budget

## External research that informed it

Primary sources:

- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**
  - https://arxiv.org/abs/1909.11942
  - motivation: cross-layer parameter sharing can preserve model quality while sharply reducing stored parameters
- **Universal Transformers**
  - https://arxiv.org/abs/1807.03819
  - motivation: depth reuse can improve expressivity, but must be balanced against compute
- **Fast Transformer Decoding: One Write-Head is All You Need**
  - https://arxiv.org/abs/1911.02150
  - considered as an alternative parameter-saving direction (more aggressive KV sharing), but not chosen for this candidate because the repo's current 4-KV GQA line is already strong and stable

Why this is different from the repo's existing sharing tricks:

- the records already share some **value-embedding machinery** across late layers,
- but they do **not** share the main attention/MLP projection banks across mirrored layers,
- and they do **not** currently have an ALBERT-style mirrored-core layout.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps that record's strong stack intact where possible:

- LeakyReLU(0.5)^2 MLP
- Parameter Banking + Parallel Muon
- partial RoPE
- LN scaling
- XSA on the last 4 layers
- BigramHash + SmearGate + VE
- EMA / SWA / GPTQ-lite-style int6 export
- optional legal score-first TTT

New changes in this candidate:

1. **13 logical layers by default** instead of 11.
2. **`MIRROR_TIE=1` by default**.
3. Added **`SHARED_LAYERS`** control; default `0` resolves to the mirrored count automatically.
4. Replaced per-layer bank storage with a **mirror map**:
   - logical layer map: `[0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0]`
   - only 7 unique shared cores are stored/exported
5. Updated quantization export helpers so the artifact only serializes the **unique shared banks**, not duplicated logical layers.
6. Made VE placement default to the **last two logical layers** when `VE_LAYERS` is left empty.

In other words, this is a structural compression/depth trade: deeper logical network, but fewer unique heavy matrices.

## How to run / evaluate

Training-only / roundtrip evaluation:

```bash
NUM_LAYERS=13 MIRROR_TIE=1 SHARED_LAYERS=7 VE_LAYERS=11,12 \
BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to compare against the current best overall eval path, add the legal TTT flags from the `2026-03-23` record on top of the command above, for example:

```bash
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0
```

`TTT_FREEZE_BLOCKS` should stay at `0` for this candidate. Because the heavy attention/MLP weights live in bank tensors instead of block-local parameters, partial block freezing would only freeze the light shell parameters and give misleading TTT semantics. The candidate now rejects `TTT_FREEZE_BLOCKS>0` explicitly.

## Main expected risks / tradeoffs

- **Step-count risk:** 13 logical layers increase compute per step, so the model may train fewer steps inside the 600-second cap.
- **Over-sharing risk:** mirrored layers may want different projection weights once the network gets deeper; sharing could underfit or cause early/late-layer interference.
- **TTT interaction risk:** if legal TTT is enabled, shared banks mean one adaptation update affects both sides of the mirrored stack.
- **Evaluation cost risk:** combining a slower 13-layer forward pass with stride-64 evaluation and legal TTT may increase total eval time noticeably.

## Validation run for this candidate

Commands run in this workflow environment:

```bash
python -m compileall candidates/202603312144_mirror-tied-13l/train_gpt.py
```

Outcome:

- ✅ Syntax compilation passed.

CPU smoke test status:

- Not run here.
- This environment's Python did not have `torch` installed, and the inherited record base assumes the full CUDA/FlashAttention training stack.
- Because of that, a real startup smoke test was **not feasible in this workflow run** without adding heavyweight dependencies that are outside the repository's existing lightweight validation path.
