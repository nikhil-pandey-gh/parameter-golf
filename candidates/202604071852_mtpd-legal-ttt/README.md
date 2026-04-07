# Candidate: Self-Distilled MTP on the Legal-TTT Stack

## Hypothesis

The current best 10-minute stack already gets most of its gains from changes that improve training or evaluation **without increasing exported artifact size**: Partial RoPE, LN scale, EMA/GPTQ-lite export, LeakyReLU(0.5)^2, and legal score-first TTT. A **single self-distilled multi-token-prediction (MTP) head** should follow the same pattern: better training signal and token efficiency during the 600-second run, **zero exported-byte cost** after stripping the auxiliary head before quantization.

## Why this is promising here

1. The strongest published record is `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, which already has MTP scaffolding in code, already excludes `mtp_heads` from export, and therefore already points toward a "training-only auxiliary head" avenue.
2. None of the reviewed record READMEs actually used MTP. The only explicit mention is `MTP_NUM_HEADS=0` in the 2026-03-20 XSA README.
3. In the 2026-03-23 script, the MTP heads were **not added to any optimizer group**, so even if enabled they would not have trained. This candidate fixes that dormant path instead of inventing a brand-new stack.
4. Recent MTP papers argue that self-distillation is the low-overhead way to preserve main-head quality while still improving auxiliary future-token heads:
   - **Multi-Token Prediction via Self-Distillation** (arXiv:2602.06019)
   - **Self-Distillation for Multi-Token Prediction / MTP-D** (arXiv:2603.23911)

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best mean `val_bpb=1.1194`
  - LeakyReLU(0.5)^2, parameter banking + Parallel Muon, legal score-first TTT
- **Export discipline:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - reinforced that small export-side quantization wins matter late in the stack
- **Zero-byte architectural refinements:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Partial RoPE + LN scale showed that small/no-parameter changes can still move BPB materially
- **Earlier context:** the 2026-03-20/21 records established the now-standard 11L + BigramHash + SmearGate + XSA + EMA trajectory.

## External research that informed it

- **arXiv:2602.06019 — Multi-Token Prediction via Self-Distillation**
  - argues that online self-distillation can turn a standard autoregressive model into a multi-token predictor without needing a separate verifier model.
- **arXiv:2603.23911 — Self-Distillation for Multi-Token Prediction**
  - emphasizes that self-distillation improves auxiliary-head acceptance while preserving main-head quality better than naive MTP training.
- **Considered but not chosen:** recursive / shared-depth papers such as **Relaxed Recursive Transformers** (arXiv:2410.20672) and **Mixture-of-Recursions** (arXiv:2507.10524). They are promising for parameter efficiency, but they require a much larger architectural rewrite and a riskier compute tradeoff than this repository's current winner stack.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps the entire proven stack intact and makes only targeted changes:

| Change | Motivation |
|---|---|
| Default `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15` | enable a cheap one-head future-token auxiliary task |
| Add `MTP_DISTILL_WEIGHT` and `MTP_DISTILL_TEMPERATURE` | supervise the MTP head with detached future main-head logits, not just hard future-token labels |
| Add MTP head weights to the AdamW optimizer group | fixes the dormant wiring so enabled MTP heads actually train |
| Keep excluding `mtp_heads` from export | preserves the "no artifact-size tax" property |
| Add FlashAttention fallback + `SMOKE_TEST=1` path | allows lightweight CPU smoke validation in environments without the Hopper runtime stack |

## How to run

From this candidate directory:

```bash
SEED=1337 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
MTP_DISTILL_WEIGHT=0.5 MTP_DISTILL_TEMPERATURE=1.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Most of the strong 2026-03-23 defaults remain baked into the script: 11 layers, 2048 train/eval length, BigramHash, SmearGate, XSA on the last 4 layers, Partial RoPE, LN scale, EMA, Parallel Muon, int6+lzma export, and LeakyReLU(0.5)^2.

## How to evaluate

The script still performs the same end-to-end flow as the base record:

1. Train for the wallclock-capped 600-second budget.
2. Apply EMA.
3. Export with MTP heads removed.
4. Quantize to mixed int6 / int8 + lzma.
5. Re-load and measure roundtrip / sliding-window BPB.
6. If `TTT_ENABLED=1`, run the legal score-first TTT evaluation.

## Expected risks and tradeoffs

- **Training throughput risk:** even one extra MTP head adds loss computation and optimizer work.
- **Main-head regression risk:** naive MTP can hurt main next-token performance; that is why this candidate uses detached-teacher distillation instead of only hard future-token CE.
- **TTT interaction uncertainty:** if legal TTT already recovers most of the remaining error, the added training signal may have only a modest post-TTT effect.
- **CPU smoke is still limited:** it now checks forward/backward plus MTP-head optimizer inclusion/update, but it is still not a substitute for real FineWeb/H100 training.

## Validation run in this workflow

Because the base runner did not have the repo's Python deps installed, validation used a temporary venv at `/tmp/gh-aw/agent/pg-venv`.

### Commands

```bash
. /tmp/gh-aw/agent/pg-venv/bin/activate
python -m compileall candidates/202604071852_mtpd-legal-ttt/train_gpt.py
SMOKE_TEST=1 python candidates/202604071852_mtpd-legal-ttt/train_gpt.py
```

### Outcome

- `compileall`: passed
- `SMOKE_TEST=1`: passed, printing `smoke_test_loss:4.430256 mtp_weight_delta:9.999993e-03`

No full GPU training run was executed in this workflow environment; the real training path still expects the repository's CUDA/FineWeb setup.
