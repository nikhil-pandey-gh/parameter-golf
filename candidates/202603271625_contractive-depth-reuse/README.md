# Contractive Depth Reuse

## Hypothesis

This candidate tests whether the repository's strong small-GPT stack can benefit from **shared-depth recurrence** without falling into the plain looping failure mode noted in earlier records. The key idea is to reuse a smaller set of transformer blocks across multiple recurrent passes, but stabilize the reuse with:

- a **contractive update** `x <- x + sigma(g_t) * (F_t(x) - x)`, inspired by SCORE,
- a learned **step embedding** for each unrolled depth position, inspired by Universal Transformer time-step conditioning,
- the existing **U-Net-style skip layout** from the root baseline, so the model still gets coarse encoder/decoder structure.

The intended payoff is more **effective depth per artifact byte**: if recurrence works, the model can trade parameters for additional iterative refinement instead of paying full cost for every layer independently.

## Why this is promising for this repository

The repo's winning trend is clear: better submissions keep finding ways to spend the 16MB budget on more useful capacity.

- Early wins came from evaluation and quantization hygiene.
- Later wins came from pushing to deeper 10L/11L stacks, larger MLPs, and sharper compression.
- A prior record explicitly called plain depth recurrence "promising" but said it likely needed more optimization than the raw 10-minute budget allowed.

This candidate tries to revisit that direction with a more modern recurrence recipe: instead of just looping the same blocks, it adds **controlled recurrent depth** and **iteration-specific conditioning** so reuse is less brittle and hopefully easier to optimize.

## Prior records and repo evidence that influenced this candidate

- Root baseline: `train_gpt.py`
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
  - explicitly notes that naive depth recurrence looked interesting but underperformed in the 10-minute setup.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md`
  - includes an unused `NUM_LOOPS` path, showing the repo already explored reusable depth mechanically, but not as an active record ingredient.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - together these show that deeper effective stacks, 2048-length training, stronger warmdown, and better evaluation all matter.

## External research that informed it

- **Universal Transformer** (Dehghani et al., 2019): recurrently applies a shared Transformer block with step-wise conditioning.
  - https://arxiv.org/abs/1807.03819
- **ALBERT** (Lan et al., 2020): cross-layer parameter sharing can preserve quality while reducing parameter count.
  - https://arxiv.org/abs/1909.11942
- **SCORE: Replacing Layer Stacking with Contractive Recurrent Depth** (Godin, 2026): proposes the contractive update `h_{t+1} = (1 - dt) * h_t + dt * F(h_t)` as a stable shared-depth alternative to ordinary layer stacking.
  - https://arxiv.org/abs/2603.10544
- **Mixture-of-Recursions** (Bae et al., 2025): shows that parameter sharing plus recursive depth can create a favorable quality/efficiency frontier.
  - https://arxiv.org/abs/2507.10524
- **AdaPonderLM** (Song et al., 2026): recent evidence that recurrent language models benefit from iteration-specific gating and adaptive depth ideas.
  - https://arxiv.org/abs/2603.01914

## What changed versus the chosen base implementation

This candidate starts from the repository root `train_gpt.py` rather than from one of the more complex record scripts, to keep the experiment self-contained and easy to reason about.

Compared with the root baseline, the candidate script changes the following:

- replaces independent depth with **`NUM_LAYERS` shared blocks** unrolled for **`RECURRENT_STEPS`** passes,
- adds learned **`step_embeddings`** and **`contractive_gate_logits`** control tensors,
- keeps the existing baseline block definition, tokenizer-aware BPB evaluation, Muon/Adam optimizer split, and int8+zlib export path,
- upgrades the default experiment settings toward the stronger repo regime:
  - `TRAIN_SEQ_LEN=2048`
  - `TRAIN_BATCH_TOKENS=786432`
  - `MLP_MULT=3`
  - `ITERATIONS=9000`
  - `WARMDOWN_ITERS=3000`
- adds **sliding-window final evaluation** support via `EVAL_STRIDE` / `EVAL_BATCH_SEQS`.

## How to run or evaluate it

From this candidate directory. The script now resolves its default dataset and tokenizer paths relative to its own file location, so the two path overrides below are optional if you are using the repository's standard layout:

```bash
RUN_ID=contractive_depth_reuse \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=6 RECURRENT_STEPS=2 \
MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
ITERATIONS=9000 WARMDOWN_ITERS=3000 MAX_WALLCLOCK_SECONDS=600 \
MATRIX_LR=0.03 SCALAR_LR=0.03 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
GRAD_CLIP_NORM=0.3 \
CONTRACTIVE_GATE_INIT=0.65 STEP_EMBED_INIT_STD=0.02 \
EVAL_STRIDE=64 EVAL_BATCH_SEQS=32 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you prefer to run it from the repository root, replace the final command with:

```bash
torchrun --standalone --nproc_per_node=8 candidates/202603271625_contractive-depth-reuse/train_gpt.py
```

## Main expected risks and tradeoffs

- **Historical risk**: the repo has already seen plain looping underperform. Contractive gating and step embeddings may still be insufficient.
- **Compute tradeoff**: shared-depth recurrence saves parameters, not FLOPs. A bad recurrence schedule can spend extra compute without converting it into lower BPB.
- **Artifact headroom**: this candidate still exports with the baseline int8+zlib path, so it is architecturally novel but not yet compression-maximal.
- **Tuning sensitivity**: the best gate initialization, number of shared layers, and recurrent steps are likely coupled to sequence length and warmdown.

## Validation run in this environment

Commands executed locally in this workflow runner:

```bash
python -m compileall candidates/202603271625_contractive-depth-reuse/train_gpt.py
```

Outcome:

- `PASS` — Python bytecode compilation completed successfully.

Attempted CPU smoke test:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch
# import candidate module and run a tiny forward pass
PY
```

Outcome:

- `BLOCKED` — this workflow runner does not have PyTorch installed (`ModuleNotFoundError: No module named 'torch'`), so a real import-and-forward smoke test was not feasible here.
- In the normal Parameter Golf training environment, rerun that tiny import/forward check or a 1-step local smoke launch before spending GPU time.
