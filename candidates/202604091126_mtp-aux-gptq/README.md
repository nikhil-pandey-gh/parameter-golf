# Candidate: MTP auxiliary head on the GPTQ-lite 11L XSA/EMA base

## Hypothesis

Enable a **single training-only multi-token prediction (MTP) head** on top of the strongest non-TTT stack in the repo. The extra head should improve sample efficiency inside the fixed 10-minute wallclock budget by forcing the shared trunk to plan one token further ahead, while **export size stays unchanged** because the auxiliary head is already stripped before serialization.

## Why this is promising for this repository

The current record history shows that this repo now wins by stacking small, artifact-safe improvements on top of the mature 11-layer GPTQ-lite/XSA recipe:

- sliding-window evaluation was a large early unlock,
- int6/GPTQ-lite quantization quality became a major bottleneck,
- 11L + MLP3x + XSA + partial RoPE + EMA became the strongest non-TTT backbone,
- recurrence looked attractive in papers but was already called out here as too wallclock-hungry for a 10-minute run.

MTP is a good fit because it targets **training efficiency**, not artifact bytes. That is exactly the constraint this repo is optimizing around.

## Repository evidence that informed this choice

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest non-TTT stack in the repo (`val_bpb: 1.1233`) with 11L, XSA4, partial RoPE, LN scale, VE128, SmearGate, BigramHash, EMA, GPTQ-lite int6 export.
- **Why not recurrence first:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - both flag looped depth / layer recurrence as compute-hungry under this challenge budget.
- **Why stay off the TTT path here:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - best absolute score, but much of the gain comes from legal score-first TTT and a more invasive optimizer rewrite, so it is a worse minimal base for a clean next candidate.

There were **no prior `candidates/` directories** in the repo when this candidate was created.

## External research that informed this choice

### Primary source for the chosen idea

- **Better & Faster Large Language Models via Multi-token Prediction** (Gloeckle et al., 2024)  
  https://arxiv.org/abs/2404.19737
  - trains the model to predict multiple future tokens with auxiliary heads on a shared trunk,
  - reports improved sample efficiency without changing the final backbone,
  - especially attractive here because auxiliary heads can be dropped at export time.

### Strong ideas considered but not chosen

- **Intra-Layer Recurrence** (Nguyen et al., 2025) — https://arxiv.org/abs/2505.01855  
- **Looped Transformers** (Yang et al., 2024) — https://arxiv.org/abs/2311.12424  
  - both are appealing byte-for-capacity ideas, but repo evidence already suggests recurrence loses too many steps in a 10-minute run.
- **QuaRot** (Croci et al., 2024) — https://arxiv.org/abs/2404.00456  
- **SpinQuant** (Liu et al., 2025) — https://arxiv.org/abs/2405.16406  
  - quantization-friendly rotations are interesting for this int6-heavy repo, but they require a broader functional reparameterization than this candidate.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. `MTP_NUM_HEADS` now defaults to **1** instead of `0`.
2. `MTP_LOSS_WEIGHT` stays at the existing conservative default of **0.2**.
3. `LATE_QAT_THRESHOLD` now defaults to **0.0** instead of `0.15`.
   - This candidate is intentionally centered on MTP, not on the copied base's late-QAT path.
   - If QAT is desired, prefer enabling it from startup with `QAT_ENABLED=1` rather than relying on a post-compile late toggle.

Everything else stays aligned with the proven GPTQ-lite 11L stack.

## How to run

From the candidate directory:

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This candidate resolves its default `DATA_PATH` and `TOKENIZER_PATH` from the repository root, so the command above works when launched from inside `candidates/202604091126_mtp-aux-gptq/`.

Relevant optional knobs:

```bash
MTP_NUM_HEADS=1
MTP_LOSS_WEIGHT=0.2
QAT_ENABLED=0
LATE_QAT_THRESHOLD=0.0
```

## How to evaluate

The script keeps the base recipe's flow:

1. train the 11L GPTQ-lite/XSA model with the auxiliary MTP head active,
2. apply EMA weights,
3. **drop `mtp_heads.*` from the exported state dict**,
4. quantize/export the normal backbone,
5. evaluate the exported model with the usual roundtrip and sliding-window metrics.

That means the auxiliary head only affects training; it does not consume submission bytes.

## Main expected risks and tradeoffs

- **Throughput risk:** even one extra vocab head adds compute, so step count may fall slightly.
- **Objective mismatch risk:** better sample efficiency is plausible, but the auxiliary loss could still trade off against the exact BPB metric.
- **No direct repo ablation yet:** this candidate relies on strong external evidence plus dormant support already present in the base code, but it has not been run to convergence here.
- **QAT caveat:** this candidate disables late-QAT by default so the experiment stays focused and does not depend on a compile-fragile activation path copied from the base.

## Validation

- From the **repository root**:
  - `python -m compileall train_gpt.py data candidates/202604091126_mtp-aux-gptq/train_gpt.py`
  - success
- From the **candidate directory**:
  - `python -m compileall train_gpt.py`
  - success
- Minimal CPU runtime smoke test:
  - not feasible in this environment; the copied base stack requires CUDA and the FlashAttention interface at runtime, so a CPU-only launch would fail before reaching training.
