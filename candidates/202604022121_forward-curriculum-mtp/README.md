# Forward-Curriculum MTP on the 11L LeakyReLU^2 Stack

## Hypothesis

Forward-curriculum **multi-token prediction (MTP)** should improve sample efficiency in this repository's 600-second tiny-LM setting because it gives the shared 11-layer trunk richer supervision only after next-token training has stabilized. The main bet is that this can improve the **pre-export model** without increasing submission bytes, since the auxiliary MTP heads are excluded from the serialized artifact.

## Why it is promising for this repository

The record progression in `records/` shows that the repo has already captured most of the obvious wins from:

- sliding-window evaluation,
- int6 plus stronger codecs,
- 11-layer / 3x-MLP stacks,
- XSA, partial RoPE, LN scaling,
- EMA/SWA, GPTQ-lite, and legal TTT.

That makes a **training-signal** improvement more attractive than another narrow codec or schedule tweak. MTP is a good fit because it:

1. targets sample efficiency directly,
2. reuses the current trunk and export path,
3. can be implemented with a surgical change instead of new infrastructure.

## Which records influenced it

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current stack,
  - contributes LeakyReLU(0.5)^2, parameter banking, parallel Muon, and training-only export stripping for `mtp_heads`.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest non-TTT export-aware 11-layer recipe,
  - established the EMA + GPTQ-lite + warmdown core that the latest stack builds on.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - partial RoPE + LN scale stack retained in the modern base.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - showed that deeper-layer attention-side changes still matter after the early quantization wins.

There were no prior `candidates/` directories when this candidate was created.

## Which external research informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (2024), arXiv:2404.19737  
  MTP improves sample efficiency by predicting multiple future tokens with auxiliary heads on a shared trunk.
- Ansar Aynetdinov and Alan Akbik, **"Pre-Training Curriculum for Multi-Token Prediction in Language Models"** (2025), arXiv:2505.22757  
  Small language models benefit much more when MTP is introduced with a **forward curriculum** instead of being enabled fully from step 0.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate adds:

1. **Forward-curriculum MTP defaults**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
   - `MTP_CURRICULUM=1`
   - `MTP_CURRICULUM_START=0.10`
   - `MTP_CURRICULUM_END=0.70`
2. **Sequential head activation**
   - head 1 ramps in first,
   - head 2 ramps in later,
   - early training stays closer to pure next-token prediction,
   - the curriculum follows **wallclock progress** when `MAX_WALLCLOCK_SECONDS` is active, so it stays aligned with the repository's 600-second training budget.
3. **Training-only artifact behavior preserved**
   - `mtp_heads.*` are still excluded from export and roundtrip evaluation.
4. **CPU-safe smoke path**
   - `SMOKE_TEST=1 python train_gpt.py` runs a tiny synthetic forward/logits pass without requiring dataset shards or FlashAttention 3.
5. **Attention fallback**
   - uses PyTorch SDPA when FlashAttention 3 is unavailable so the smoke path can run on CPU.

## How to run or evaluate it

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Explicit curriculum run:

```bash
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 \
MTP_CURRICULUM=1 MTP_CURRICULUM_START=0.10 MTP_CURRICULUM_END=0.70 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To test whether it stacks with the existing legal TTT path:

```bash
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 \
MTP_CURRICULUM=1 MTP_CURRICULUM_START=0.10 MTP_CURRICULUM_END=0.70 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks or tradeoffs

- The auxiliary heads still cost **training compute**, so any MTP gain must beat the reduction in total step count.
- Small models can regress under naive MTP; this candidate depends on the **curriculum** to avoid that failure mode.
- The interaction with legal TTT is unverified: MTP may help the pre-TTT base, the TTT-adapted model, both, or neither.
- The current schedule is a first strong guess, not a sweep. The most likely follow-ups are tuning:
  - `MTP_LOSS_WEIGHT`,
  - head count,
  - curriculum start/end fractions,
  - whether TTT is still worth its eval-time cost once training improves.

## Validation

Commands:

- `python -m compileall candidates/202604022121_forward-curriculum-mtp/train_gpt.py`
- `SMOKE_TEST=1 python candidates/202604022121_forward-curriculum-mtp/train_gpt.py`

Outcomes:

- `python -m compileall candidates/202604022121_forward-curriculum-mtp/train_gpt.py` — **passed**
- `SMOKE_TEST=1 python candidates/202604022121_forward-curriculum-mtp/train_gpt.py` — **blocked by environment**, because this runner does not have the repository's declared Python dependencies installed (`numpy`, `sentencepiece`, `torch`)
