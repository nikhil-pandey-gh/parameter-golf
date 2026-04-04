# MTP1 Auxiliary Head on the LeakyReLU² + Legal TTT Stack

## Hypothesis

A **single training-only multi-token prediction (MTP) head** should improve sample efficiency for this tiny 11-layer model without spending final artifact bytes, because the extra head is used only during training and stripped from export.

## Why this is promising here

The record progression in this repository is already concentrated around a strong 11-layer recipe:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

That line has already captured the biggest local wins: int6-aware training/export, EMA-style averaging, deep-layer XSA, partial RoPE, LeakyReLU(0.5)^2, and legal score-first TTT. The cleanest remaining lever is to make the model learn **more from the same 600-second training budget**.

This candidate uses the current best record implementation as the base because it already has:

- the best measured score in-tree (`val_bpb: 1.1194` with TTT),
- training-only `mtp_heads` support in the model,
- an export path that already excludes `mtp_heads` from the final artifact.

## Prior repository work that influenced this candidate

- **Baseline trend**: moving from 9L baseline to the 11L / MLP3x / int6-aware stack was the largest sustained improvement path.
- **Best current score**: `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` shows the strongest end-to-end stack in this repo.
- **Dormant prior-art in code**: several recent record scripts already exposed `MTP_NUM_HEADS` and `MTP_LOSS_WEIGHT`, but the submitted runs kept `MTP_NUM_HEADS=0`, so the idea was present but not actually exercised.
- **No prior candidates**: there was no existing `candidates/` directory when this candidate was created.

## External research that informed it

1. **Better & Faster Large Language Models via Multi-token Prediction** (Gloeckle et al., arXiv:2404.19737) argues that shared-trunk, multi-future-token heads improve sample efficiency and downstream capability, with especially strong gains on generative tasks.
2. **DeepSeek-V3** (DeepSeek-AI et al., arXiv:2412.19437) explicitly uses a multi-token prediction objective in a strong modern pretraining stack, which is a useful signal that MTP is not just an inference trick.
3. **On multi-token prediction for efficient LLM inference** (Mehra et al., arXiv:2502.09419) finds hidden states are strongly specialized for next-token prediction and that MTP works best when trained jointly with the backbone, which motivates a **small, conservative** setting here instead of many aggressive future-token heads.
4. **Multi-Token Prediction Needs Registers** (Gerontopoulos et al., arXiv:2505.10518) reinforces the theme that MTP is strongest when parameter overhead stays negligible and the base language-model objective remains intact.

## What changed vs the chosen base

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Candidate changes:

1. **Enable one auxiliary MTP head by default**
   - `MTP_NUM_HEADS`: `0 -> 1`
   - `MTP_LOSS_WEIGHT`: `0.2 -> 0.1`
2. **Fix the optimizer wiring**
   - the training-only `mtp_heads[*].weight` tensors are now added to the non-banked AdamW parameter group, so the auxiliary head actually trains.

Everything else stays intentionally unchanged: LeakyReLU(0.5)^2, parameter banking, Parallel Muon, EMA, deep-layer XSA, partial RoPE, late-QAT knob, GPTQ-lite export, and optional legal TTT are all inherited from the base record.

## Why it differs from the existing records

None of the existing records actually test a real MTP-trained run in the final leaderboard stack:

- the recent 11-layer scripts expose MTP knobs,
- but the published record commands leave them disabled,
- and the copied base implementation did not put `mtp_heads` into any optimizer parameter group.

So this candidate is a genuine new twist rather than a rename of an existing record.

## How to run or evaluate it

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To compare against the current best record style more directly, enable the same legal TTT path the 2026-03-23 record used:

```bash
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The export path already strips the auxiliary MTP heads before quantization/export, so the final artifact remains the base model only.

## Main expected risks and tradeoffs

- **Step-time risk**: even one extra vocab head adds training compute and could slightly reduce total steps in the 600-second cap.
- **Tiny-model risk**: this model is much smaller than the large-model MTP literature; the auxiliary loss may steal capacity from the main next-token objective if weighted too strongly.
- **Objective-mismatch risk**: Mehra et al. shows hidden states are specialized for NTP, so a larger MTP horizon is more likely to hurt than help here.
- **No export-size gain**: this is purely a training-signal bet, not a compression or architectural byte-saving change.

## Validation

Commands run in this repository:

```bash
python -m compileall candidates/202604042209_mtp1-aux-head/train_gpt.py
```

Outcome:

- `compileall`: passed
- CPU smoke test: attempted, but **not feasible in this runner** because both `python` and `python3` lacked an installed `torch` module, so a local forward-pass check could not be executed here without adding new infrastructure
