# Filtered + Anchored Score-First TTT

## Hypothesis

The current best repository result already gets a meaningful gain from legal score-first test-time training (TTT), but its adaptation loop still updates on every scored chunk sequence and lets weights drift continuously across the full validation stream. My hypothesis is that a more conservative TTT loop will work better for this challenge: update only on the lowest-entropy sequences in each TTT mini-batch, then apply a small pullback toward the pre-TTT checkpoint after every SGD step.

That should preserve most of the benefit of score-first adaptation while reducing two failure modes that matter for tiny models under this benchmark: over-reacting to noisy chunks and gradually forgetting the base model as adaptation proceeds through thousands of validation windows.

## Why this is promising for this repository

Repository evidence already says eval-time methods matter a lot here:

- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/` showed that evaluation protocol alone can move BPB by roughly `-0.03`.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the strongest reviewed run, and its legal score-first TTT contributes about `-0.0025 BPB` on top of the training architecture.
- Earlier `records/track_10min_16mb/2026-03-17_LoRA_TTT/` also showed that adaptation-aware evaluation helps, even though that early LoRA flavor was not the final answer.

Given that the latest frontier is already near the artifact limit and many recent architectural gains are incremental, tightening the TTT loop is a better fit than adding more training-time machinery or changing the model body again.

## Prior records that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Additional repository influence:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` for the 11-layer EMA/GPTQ-lite/partial-RoPE family this candidate keeps intact.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/` for the general lesson that evaluation-side improvements are first-class in this challenge.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`, whose README reports that layer recurrence was a poor direction here, helping rule out ALBERT/recurrent-style experiments for this candidate.

There were no prior `candidates/` directories in the repository at review time, so this candidate is not overlapping an earlier candidate iteration.

## External research that informed it

Two test-time adaptation papers are the main inspiration:

- **Tent: Fully Test-Time Adaptation by Entropy Minimization** (Wang et al., 2020), <https://arxiv.org/abs/2006.10726>
- **EATA: Efficient Test-Time Model Adaptation without Forgetting** (Niu et al., 2022), <https://arxiv.org/abs/2204.02610>

The important ideas for this repository are not the exact image-domain losses, but the higher-level lessons:

- conservative online adaptation can improve out-of-distribution behavior without retraining;
- high-entropy examples are often the noisiest signals for test-time updates;
- forgetting control matters when the same model is adapted repeatedly over a long stream.

This candidate ports those ideas into the repo's already-legal score-first LM TTT loop by filtering toward lower-entropy sequences and adding a cheap post-step anchor pullback instead of a more expensive Fisher-style regularizer.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in `train_gpt.py`:

- Added two new hyperparameters:
  - `TTT_KEEP_RATIO` (default `0.5`)
  - `TTT_ANCHOR_MIX` (default `0.002`)
- Added entropy-selection helpers so the kept subset is chosen globally across the distributed TTT mini-batch, not independently per rank.
- Added `apply_anchor_pullback_(...)` to lerp the adapted weights slightly back toward the pre-TTT checkpoint after each TTT optimizer step.
- Reworked the TTT training phase to:
  - compute logits with `forward_logits(...)`,
  - derive per-sequence NLL and predictive entropy,
  - train only on the selected low-entropy subset when `TTT_KEEP_RATIO < 1.0`,
  - apply the anchor pullback when `TTT_ANCHOR_MIX > 0.0`.

Everything else intentionally stays as close as possible to the current strongest repository stack.

## How to run / evaluate

From this candidate directory, start from the current best legal-TTT recipe and add the new TTT controls:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
TTT_KEEP_RATIO=0.5 TTT_ANCHOR_MIX=0.002 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Recommended first ablations:

- `TTT_KEEP_RATIO` in `{1.0, 0.75, 0.5}`
- `TTT_ANCHOR_MIX` in `{0.0, 0.001, 0.002, 0.005}`

A good sign would be matching or slightly improving the existing post-TTT gain while lowering late-chunk drift or TTT instability across seeds.

## Validation

I ran the lowest-cost checks available in this environment.

### Command

```bash
python -m compileall candidates/202603291110_filtered-anchored-ttt/train_gpt.py
```

### Outcome

- `compileall` passed successfully.

### CPU smoke-test status

I attempted an import-level smoke test next, but this runner does not have the repository's runtime dependency stack installed:

```text
ModuleNotFoundError: No module named 'torch'
```

Because this script imports PyTorch at module load time and its `main()` path also requires CUDA, a deeper CPU execution smoke test was not feasible in the current environment without installing new heavyweight dependencies.

## Main expected risks / tradeoffs

- Keeping only low-entropy sequences may under-adapt if the best TTT signal actually lives in moderate-entropy or hard examples.
- Anchor pullback could erase real TTT gains if `TTT_ANCHOR_MIX` is too large.
- The change only affects evaluation-time adaptation, so if the true remaining headroom is mostly in training-time compression or architecture, this candidate may plateau quickly.
- This candidate is intentionally a minimal intervention; it is betting that TTT quality, not model capacity, is the next highest-leverage knob.
