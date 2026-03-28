# Self-Distilled Training-Only MTP on the 1.1233 Base

## Hypothesis

A small amount of **training-only future-token supervision** can improve the hidden-state quality of the current best plain 11-layer stack without changing the exported model at all. In particular, predicting token `t+2` from the hidden state at `t`, while also distilling from the model's own main next-token head at position `t+1`, should improve sample efficiency and induction-style behavior in the same way recent multi-token prediction (MTP) papers report, but with essentially **zero artifact-byte cost** because the auxiliary head is stripped before export.

## Why this is promising for this repository

The strongest plain-training line in this repo improved mostly through tricks that changed **training dynamics or compression quality** without blowing up the 16MB artifact budget: XSA on late layers, EMA, partial RoPE, LN scaling, and GPTQ-lite clipping. A training-only MTP auxiliary objective fits that exact pattern.

This repo's recent 11-layer record family already contains dormant `mtp_heads` support and already excludes those heads from the exported state dict, but the checked-in logs do not show any prior run with `mtp_num_heads > 0`. That makes MTP a strong next step: low implementation risk, aligned with the current best base, and apparently still untested in practice here.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best non-TTT base in this checkout. This candidate forks that exact stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Confirms partial RoPE + LN scaling were real gains and that compile-sensitive toggles need care.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - Established late-layer XSA + EMA as a durable improvement.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Shows there is still headroom in this architecture family, but most of that win comes from eval-time TTT. This candidate instead targets the plain exported model.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - Argues that auxiliary future-token prediction improves sample efficiency and helps induction-like behavior.
- Guoliang Zhao et al., **"Self-Distillation for Multi-Token Prediction"** (`arXiv:2603.23911`)
  - Shows that self-distillation can preserve main-head quality while improving MTP behavior.
- John Kirchenbauer et al., **"Multi-Token Prediction via Self-Distillation"** (`arXiv:2602.06019`)
  - Reinforces that online self-distillation can add multi-token supervision without introducing a separate deployed model.

## What changed versus the chosen base implementation

This candidate starts from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` and makes four focused changes:

1. **Turns on one training-only MTP head by default**
   - `MTP_NUM_HEADS=1`
   - The auxiliary head predicts token `t+2` from the hidden state at token `t`.

2. **Adds self-distillation for the MTP head**
   - The auxiliary future-token logits are matched to the model's own main-head logits from the future position that predicts the same target.
   - New knobs:
     - `MTP_DISTILL_WEIGHT=0.5`
     - `MTP_DISTILL_TEMP=1.5`
   - The base next-token loss remains the primary objective.

3. **Keeps the auxiliary parameters out of the artifact**
   - `mtp_heads` are still excluded before quantization/export, so the deployed model shape is unchanged from the base stack.

4. **Adds a safe attention fallback for local smoke testing**
   - Uses FlashAttention 3 when available on CUDA.
   - Falls back to PyTorch SDPA otherwise, which makes tiny CPU import/forward smoke tests possible in a compatible environment.

## How to run / evaluate

Run from the candidate directory:

```bash
cd candidates/202603280507_selfdist-mtp
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.15 \
MTP_DISTILL_WEIGHT=0.5 \
MTP_DISTILL_TEMP=1.5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script defaults its dataset/tokenizer paths to the repository-root `data/` layout, so it can be launched from inside this candidate directory without editing paths.

## Expected risks / tradeoffs

- **Throughput risk:** even one auxiliary future-token head adds extra logits/loss work, so steps-in-600s may drop.
- **Main-head interference risk:** if the auxiliary loss is too strong, it may hurt the next-token objective instead of helping it.
- **Quantization mismatch risk:** the hidden representations may improve while the final int6 artifact does not move proportionally.
- **Research-transfer risk:** MTP papers are not written for this exact tiny-model + strict-byte-budget regime, so the effect size here may be smaller than in larger-model studies.

## Validation

Commands run locally for this candidate:

```bash
python -m compileall candidates/202603280507_selfdist-mtp/train_gpt.py
```

Outcome:

- `compileall` passed.

Additional smoke checks:

```bash
python -m venv /tmp/gh-aw/agent/venv
source /tmp/gh-aw/agent/venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet torch numpy sentencepiece

# eval-mode import + forward + export-filter check
python - <<'PY'
# imports candidate train_gpt.py, builds a tiny CPU model, runs one forward pass,
# and verifies `mtp_heads` are excluded from the export state dict
PY

# train-mode forward/backward check for the MTP branch
python - <<'PY'
# imports candidate train_gpt.py, runs a tiny CPU forward/backward pass in train mode,
# and verifies the MTP head receives gradients
PY
```

Outcomes:

- Temporary isolated virtualenv install succeeded.
- Eval-mode smoke test passed: module import worked, a tiny CPU forward pass returned a finite loss, and `mtp_heads` were correctly excluded from the export state dict.
- Train-mode smoke test passed: a tiny CPU forward/backward pass succeeded and the auxiliary MTP head received non-zero gradients.
