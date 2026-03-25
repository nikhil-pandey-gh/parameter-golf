# Forward MTP Curriculum on the 1.1194 Stack

## Hypothesis

The strongest current in-repo stack already wins on architecture, evaluation, and quantization, but it still trains with a pure next-token objective. A **small-model-friendly forward multi-token prediction (MTP) curriculum** should improve sample efficiency without materially changing artifact size:

- keep early training identical to the proven 1.1194 recipe,
- then gradually turn on a **single auxiliary future-token head** once the trunk has stabilized,
- exclude the auxiliary head from export so the candidate still spends its bytes on the main model.

The core idea is that the trunk learns better predictive features late in training, while the early next-token-only phase avoids the small-model instability that fixed MTP can introduce.

## Why this is promising for this repository

This repository's leaderboard has mostly improved by stacking:

1. better evaluation (`stride=64`, legal score-first TTT),
2. more quantization-friendly training/export,
3. low-byte architectural changes that preserve the 16 MB budget.

This candidate fits that pattern well:

- it starts from the current best in-repo recipe under `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`,
- it uses an **auxiliary training objective only**, so it does not need to survive export,
- the existing codebase already had dormant MTP support and already excludes `mtp_heads` from the exported state dict, making this a high-leverage, low-infrastructure extension.

## Prior records that influenced this candidate

### Chosen base

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current in-repo result (`val_bpb: 1.1194` mean),
  - already includes the strongest known stack here: LeakyReLU(0.5)^2, legal TTT, Parallel Muon, XSA, partial RoPE, LN scaling, VE, GPTQ-lite int6 export, and lzma compression.

### Supporting evidence from earlier runs

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - showed that post-training export quality is worth chasing aggressively.

- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - already carried MTP hooks in the training script, but kept `MTP_NUM_HEADS=0`.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - documented a `torch.compile` constant-folding failure for late-QAT. That informed this implementation: the curriculum scale is passed as a **runtime tensor input** so the auxiliary-loss schedule is not baked away.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - argues that predicting multiple future tokens can improve sample efficiency and downstream quality.

- Ansar Aynetdinov and Alan Akbik, **"Pre-Training Curriculum for Multi-Token Prediction in Language Models"** (`arXiv:2505.22757`)
  - specifically relevant here because it reports that **small language models struggle with fixed MTP**, while a **forward curriculum** helps them benefit from the objective.

The second paper is the key reason this candidate does **not** simply turn MTP on from step 1.

## What changed versus the chosen base implementation

This candidate copies the current best record script and makes a focused set of changes:

1. **Enable one auxiliary MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`

2. **Add a forward curriculum**
   - `MTP_CURRICULUM=1`
   - `MTP_START_FRAC=0.35`
   - `MTP_RAMP_FRAC=0.35`

   The MTP loss is off at the start, ramps in over the middle third of the timed run, and is fully active late in training.

3. **Compile both no-MTP and MTP forward paths during warmup**
   - this avoids paying the graph-compilation tax the first time the curriculum activates during the timed training window.

4. **Actually optimize the auxiliary heads**
   - the original dormant path counted MTP parameters for logging/export purposes, but did not add them to an optimizer group because the feature was always disabled.
   - this candidate adds `mtp_heads` to the AdamW-managed non-bank parameter set.

5. **Keep export artifact cost effectively unchanged**
   - `mtp_heads` are still stripped from `export_sd` before serialization, so the auxiliary objective spends compute, not bytes.

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202603252214_forward-mtp-curriculum

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 MTP_CURRICULUM=1 \
MTP_START_FRAC=0.35 MTP_RAMP_FRAC=0.35 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks / tradeoffs

- **Training-time overhead:** even one auxiliary head adds extra logits and CE work once the curriculum activates.
- **Schedule sensitivity:** if MTP turns on too early, the trunk may behave like the small-model failure mode described in `arXiv:2505.22757`; if it turns on too late, it may not matter.
- **Interaction risk with TTT and EMA:** the repo has strong evidence that TTT and export quality are fragile, so the curriculum may help the pre-export trunk but still be neutral after quantization and legal TTT.
- **Compile behavior:** this candidate avoids the earlier late-QAT constant-folding pitfall by using a runtime tensor input for the MTP scale, but that interaction still deserves an actual GPU run.

## Validation

### Commands run in this workflow

```bash
python -m compileall candidates/202603252214_forward-mtp-curriculum/train_gpt.py
```

### Outcomes

- `compileall` succeeded.
- A real runtime smoke test was **not feasible in this runner** because the environment did not have local `torch`, `sentencepiece`, or the cached FineWeb dataset/tokenizer artifacts installed. Rather than inventing a synthetic setup that diverges from the repository's actual runtime path, this candidate limits validation here to syntax-level checks and documents the intended GPU invocation above.
