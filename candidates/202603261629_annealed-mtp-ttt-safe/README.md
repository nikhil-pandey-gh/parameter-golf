# Annealed 1-Head MTP, TTT-Safe

## Hypothesis

A **single future-token auxiliary head** can improve sample efficiency enough to help this repository's strongest 11-layer stack within the fixed 10-minute training budget, **as long as the auxiliary is treated conservatively**:

- enable only **1** MTP head,
- give it a modest loss weight,
- **turn it off during late warmdown** so the model finishes optimizing the true next-token objective,
- and **disable it during legal TTT** so post-training adaptation stays aligned with leaderboard evaluation.

The candidate is intentionally narrow: keep the current best training/eval stack intact and only add a lightweight training-only objective.

## Why this is promising here

The repository review found two useful facts:

1. The strongest current result is `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` at **1.1194 bpb** post-TTT.
2. That code already contains a dormant `mtp_*` path, but prior records left it disabled (`mtp_num_heads:0`).

That makes MTP attractive for this repo specifically:

- it is **materially different** from prior records without requiring a new infrastructure stack,
- it is **training-only**, so the exported artifact can stay on the proven quantized next-token model,
- and it targets the challenge's biggest constraint directly: **better learning per unit wallclock** rather than another large architectural rewrite.

## Prior records that influenced this candidate

### Primary base

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - best current score,
  - LeakyReLU(0.5)^2,
  - parameter banking + parallel Muon,
  - legal score-first TTT,
  - GPTQ-lite int6 export.

### Architectural lineage kept intact

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - strong clean pre-TTT stack,
  - GPTQ-lite rowwise clip search,
  - EMA + tight SWA,
  - partial RoPE + LN scale + XSA4 + VE.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - reinforced that partial RoPE and LN scaling help,
  - also documented a late-QAT compile pitfall, which is a reminder to prefer simple, explicit changes.

## External research that informed it

### 1. Multi-token prediction improves sample efficiency

- **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"**, arXiv:2404.19737
  - trains models to predict multiple future tokens from a shared trunk using independent heads,
  - reports improved sample efficiency and stronger generative behavior,
  - which is exactly the failure mode this challenge cares about under a strict wallclock cap.

### 2. Autoregressive LMs already carry usable future-token information

- **Mohammad Samragh et al., "Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential"**, arXiv:2507.11851
  - argues vanilla autoregressive LMs already encode useful information about future tokens,
  - supporting the idea that a small auxiliary future-token objective can help without changing the main architecture.

### 3. Exact MTP can be too aggressive if used naively

- **Zayd M. K. Zuhri et al., "Predicting the Order of Upcoming Tokens Improves Language Modeling"**, arXiv:2508.19228
  - reports that exact multi-token prediction is sometimes inconsistent on standard NLP benchmarks,
  - motivating a **conservative version** here: one extra head only, small loss weight, and warmdown-time disable.

### 4. Preserving main-head quality matters

- **Guoliang Zhao et al., "Self-Distillation for Multi-Token Prediction"**, arXiv:2603.23911
  - emphasizes that auxiliary MTP machinery can hurt the main head if handled poorly,
  - which informed the decision to keep this candidate strictly **training-only** and to **disable MTP during TTT**.

## What changed vs. the chosen base implementation

Base script: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate adds only four focused changes:

1. **Candidate-local path defaults**
   - default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root,
   - so `python train_gpt.py` works from inside the candidate directory without manual path fixes.

2. **Conservative MTP-on by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`

3. **Warmdown disable for the auxiliary loss**
   - new env: `MTP_DISABLE_SCALE=0.2`
   - when the LR warmdown scale falls below that threshold, the script zeroes the runtime MTP weight, sets `mtp_num_heads=0`, and switches the training loop to a freshly compiled no-MTP graph.
   - This keeps early dense supervision, then lets the model finish the tail on the true next-token objective **without continuing to pay the extra MTP forward cost**.

4. **TTT-safe behavior**
   - legal TTT explicitly saves and disables the MTP path before chunk adaptation,
   - then restores it afterward.
   - That keeps post-training adaptation aligned with the evaluation loss instead of mixing in an auxiliary objective.

## Export / artifact behavior

The exported artifact remains next-token-only:

- the code already excludes `mtp_heads` from the exported state dict before quantization,
- and evaluation after reload constructs the eval model with `mtp_num_heads=0`.

So this candidate is meant to spend extra parameters and compute **only during training**, not in the final submission artifact.

## How to run

From the repository root:

```bash
cd candidates/202603261629_annealed-mtp-ttt-safe
python train_gpt.py
```

For a record-style multi-GPU run, start from the current best record command and add the MTP flags:

```bash
cd candidates/202603261629_annealed-mtp-ttt-safe
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 MTP_DISABLE_SCALE=0.2 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

In practice, I would pair those MTP flags with the rest of the environment used by `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`.

## Validation run in this workflow

### Succeeded

```bash
python -m compileall candidates/202603261629_annealed-mtp-ttt-safe/train_gpt.py
```

Outcome: **passed**.

### Not feasible in this runner

I attempted a lightweight import smoke check, but this runner does not have the repository's Python dependencies installed.

Command attempted:

```bash
python - <<'PY'
import importlib.util
import pathlib
p = pathlib.Path('candidates/202603261629_annealed-mtp-ttt-safe/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcome: **failed before model startup** with `ModuleNotFoundError: No module named 'numpy'`.

I also checked the runner environment directly and confirmed that `numpy`, `torch`, and `sentencepiece` are not installed, so a real CPU smoke test is not possible here without the repository's runtime dependencies.

## Main expected risks / tradeoffs

1. **Step-time tax**
   - even one auxiliary head adds extra logits work during training,
   - so the gain must come from better learning per step, not just from adding supervision.

2. **Auxiliary-objective mismatch**
   - recent work suggests exact MTP can hurt next-token quality if it is too strong.
   - This candidate mitigates that with one head, a modest weight, and late disable, but the interaction is still unproven here.

3. **MTP + TTT interaction**
   - this candidate disables MTP during TTT on purpose.
   - That is the safest choice for leaderboard alignment, but it means any benefit must come from pretraining rather than from adaptation.

## Suggested follow-up experiments

If this candidate is promising, the next three sweeps I would run are:

1. `MTP_LOSS_WEIGHT` in `{0.08, 0.12, 0.15, 0.20}`
2. `MTP_DISABLE_SCALE` in `{0.10, 0.15, 0.20, 0.25}`
3. `MTP_NUM_HEADS=2` only if the 1-head version improves pre-TTT BPB without noticeably hurting step throughput
