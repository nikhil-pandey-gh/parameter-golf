# Annealed export-free MTP on the PR#414 stack

## Hypothesis

Enable the repo's dormant multi-token prediction (MTP) auxiliary heads on top of the current best in-repo training stack, then anneal the MTP loss to zero during warmdown so the model finishes aligned with the exact next-token objective and final quantized export path.

The core bet is that MTP improves early sample efficiency and induction-style pattern formation, while warmdown annealing avoids ending training with too much weight on an auxiliary objective that disappears at export time.

## Why this is promising for this repository

The record history suggests that most remaining gains come from stackable, low-artifact-cost improvements rather than large rewrites:

- sliding-window evaluation is already saturated and mandatory,
- 11-layer / 3x-MLP / compression-aware optimization is the dominant recipe,
- tiny architectural nudges like XSA, Partial RoPE, LN scale, and SmearGate have compounded well,
- the strongest current training script already includes MTP support and strips auxiliary heads from export.

That makes MTP unusually attractive here: it can improve training dynamics without increasing the final artifact bytes.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current in-repo score,
  - base stack for this candidate: parameter banking, parallel Muon, LeakyReLU(0.5)^2, XSA4, Partial RoPE, LN scale, VE, GPTQ-lite, legal score-first TTT.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - confirms the repo already had MTP scaffolding plus export-time removal of `mtp_heads`,
  - reinforces the importance of warmdown-aware finishing for quantized models.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - shows that compile-time constant folding can silently disable late-training switches if they are not represented as tensors.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - proposes multi-token prediction as an auxiliary training objective that improves sample efficiency and induction behavior.
- DeepSeek-AI, **"DeepSeek-V3 Technical Report"** (`arXiv:2412.19437`)
  - shows that MTP remained useful enough to retain in a modern frontier training recipe.
- Zhenzhong Lan et al., **"ALBERT"** (`arXiv:1909.11942`) and Mostafa Dehghani et al., **"Universal Transformer"** (`arXiv:1807.03819`)
  - were considered because parameter sharing and recurrent depth are natural fits for artifact limits,
  - but the repo history already hints that extra recurrent depth can cost too many training steps in this wallclock-constrained setting, so this candidate chooses the lower-risk training-only auxiliary objective instead.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. `MTP_NUM_HEADS` now defaults to `1` instead of `0`.
2. `MTP_LOSS_WEIGHT` now defaults to `0.15`.
3. New flag: `MTP_ANNEAL_WITH_LR=1` by default.
4. The live MTP weight is stored in a non-persistent tensor buffer (`mtp_weight`) so the compiled model can consume updated values during training without serializing that control value into the export artifact.
5. During training, the active MTP weight follows the same warmdown multiplier as the learning rate:
   - full weight before warmdown,
   - linearly decayed weight during warmdown,
   - near-zero auxiliary pressure by the time EMA / GPTQ-lite / export happen.
6. `mtp_heads` are optimized through the existing non-bank Adam path and participate in the replicated gradient sync.
7. Training logs now print the live `mtp_weight`.

The export path still removes `mtp_heads`, so the saved evaluation artifact remains the same inference graph as the non-MTP base.

## How to run

From this candidate directory:

```bash
RUN_ID=annealed_mtp \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Main MTP knobs:

```bash
MTP_NUM_HEADS=1
MTP_LOSS_WEIGHT=0.15
MTP_ANNEAL_WITH_LR=1
```

Clean ablation:

```bash
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- Even one MTP head adds extra logits and cross-entropy work, so throughput may drop enough to erase the optimization benefit.
- MTP gains may be smaller for a tiny 11-layer model than for the larger models highlighted in the literature.
- `torch.compile` can still behave unexpectedly around auxiliary paths, so runtime verification on the intended GPU stack is important.
- The interaction with score-first TTT is uncertain: MTP may improve the pre-TTT base model more than the final post-TTT score.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603291943_annealed-mtp/train_gpt.py
python -m py_compile candidates/202603291943_annealed-mtp/train_gpt.py
```

Outcomes:

- `compileall`: passed
- `py_compile`: passed

CPU/GPU smoke test status:

- A real runtime smoke test was not feasible in this environment because this script hard-requires CUDA plus the challenge's dataset / FlashAttention runtime stack, which are not available for execution in this workflow container.
