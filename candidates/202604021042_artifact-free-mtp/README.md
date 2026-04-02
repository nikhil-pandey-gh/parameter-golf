# Artifact-Free MTP on the 11L EMA + GPTQ-lite Stack

## Hypothesis

Turn on **training-only multi-token prediction (MTP)** in the strongest no-TTT stack already in this repo, but keep the auxiliary heads **out of the exported artifact**. The bet is that a small amount of future-token supervision improves sample efficiency and next-token BPB under the same 10-minute budget, while costing essentially **zero submission bytes** because `mtp_heads.*` are still excluded from export.

## Why this is promising here

The record history has pushed hard on compression-aware training, quantization, and evaluation, but it has **not actually exercised the dormant MTP hooks** already present in the frontier training scripts:

- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md` explicitly runs with `MTP_NUM_HEADS=0`.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py` already contains MTP heads plus export-time exclusion for `mtp_heads.*`, making this idea unusually cheap to test.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` shows the frontier has kept finding small stacked gains, but mostly from eval or architectural refinements rather than from changing the training objective itself.

So this candidate explores a missing axis: **better supervision at train time without paying extra artifact bytes at eval time**.

## Chosen base implementation

This candidate is based on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

That base was the strongest clean no-TTT training stack in the repo review: 11 layers, 512d, 4-layer XSA tail, partial RoPE, VE128, EMA + tight SWA, GPTQ-lite int6 export, and late-QAT-aware tuning.

## Prior records that influenced the choice

- **2026-03-20 Efficient Partial XSA**: showed the late-layer XSA stack was strong and also surfaced the dormant `MTP_NUM_HEADS=0` knob.
- **2026-03-21 Partial RoPE + LN Scale**: validated that small late-stage architectural refinements keep stacking on the same family of models.
- **2026-03-22 EMA + GPTQ-lite + warmdown3500**: chosen base because it is the strongest no-TTT script and already has the exact MTP/export plumbing needed here.
- **2026-03-23 LeakyReLU² + Legal TTT + Parallel Muon**: confirms the frontier still moves through small additive improvements, but via a more complex eval-heavy stack than needed for this candidate.

There were **no prior experiments under `candidates/`** when this candidate was created.

## External research

The main external motivation is:

1. [Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737): predicts several future tokens from a shared trunk and reports better sample efficiency.
2. [Noci et al., *Thinking into the Future: Latent Lookahead Training for Transformers* (arXiv:2603.20219)](https://arxiv.org/abs/2603.20219): more evidence that future-aware supervision can help token prediction by forcing the model to plan ahead.
3. [Rybakov et al., *Methods of improving LLM training stability* (arXiv:2410.16682)](https://arxiv.org/abs/2410.16682): motivated keeping the auxiliary objective modest and horizon-decayed instead of pushing a large extra loss from the start.

## What changed vs. the chosen base

Only the candidate directory was added; the repository root and record folders were left untouched.

Relative to the 2026-03-22 base script, this candidate:

1. Sets **repo-root-relative default dataset/tokenizer paths** so the script can be launched from inside the candidate directory.
2. Enables **2 MTP heads by default** with `MTP_NUM_HEADS=2`.
3. Lowers the auxiliary strength to **`MTP_LOSS_WEIGHT=0.15`**.
4. Adds **`MTP_HORIZON_DECAY=0.5`** so the first future-token head is weighted most heavily and farther horizons matter less.
5. Keeps the existing **export-time exclusion of `mtp_heads.*`**, preserving the artifact-free nature of the auxiliary heads.

## How to run

From the candidate directory:

```bash
cd candidates/202604021042_artifact-free-mtp

RUN_ID=artifact_free_mtp \
SEED=1337 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
WARMDOWN_ITERS=3500 \
VAL_LOSS_EVERY=2000 \
EVAL_STRIDE=64 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_HORIZON_DECAY=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root, so this works from inside the candidate directory once the usual challenge data has been downloaded.

## Validation

Commands run while preparing this candidate:

```bash
python -m compileall candidates/202604021042_artifact-free-mtp/train_gpt.py
python - <<'PY'
from pathlib import Path
candidate = Path('/home/runner/work/parameter-golf/parameter-golf/candidates/202604021042_artifact-free-mtp/train_gpt.py').resolve()
repo_root = candidate.parents[2]
print(candidate)
print(repo_root)
print((repo_root / 'data').exists())
PY
```

Outcomes:

- `compileall` succeeded.
- The candidate's repo-root-relative path resolution behaves as intended.
- A minimal CPU runtime smoke test was **not feasible** in this container because this script is designed for the challenge CUDA environment and imports `flash_attn_interface` / requires CUDA execution paths.

## Main risks and tradeoffs

- Even export-free MTP heads still cost **training FLOPs and memory**, so they may lose too much throughput under the 10-minute wallclock cap.
- Future-token supervision can help sample efficiency, but it can also **steal optimization budget** from the main next-token objective if the weight is too large.
- The best follow-up if this helps at all is probably to stack it with the repo's later wins: **LeakyReLU²**, **Parallel Muon**, or even **legal TTT** at evaluation.
