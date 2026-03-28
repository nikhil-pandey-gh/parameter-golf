# Control-Tensor Legal TTT

## Hypothesis

The current frontier in this repo suggests that **legal score-first test-time training (TTT)** is the strongest remaining evaluation-time lever, but the best published recipe still adapts essentially the whole model during TTT. This candidate tests a more parameter-efficient variant: keep the strong `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` training/export stack, but restrict TTT updates to the model's existing **control tensors** (scales, gates, skip weights, mixes, and similar low-dimensional parameters).

The bet is that most of the useful adaptation signal lives in those existing control knobs, so a control-only TTT pass can recover much of the full-model TTT gain while being cleaner, stabler, and more compatible with the banked/quantization-aware structure of the current best script.

## Why this is promising for this repository

Three repo-specific observations point here:

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` shows that legal score-first TTT is already worth about **-0.0025 BPB** on top of an otherwise strong stack.
2. Earlier TTT exploration (`records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md`) tried **adapter-style TTT**, and the best current stack tried **full-block TTT**, but there is no prior run in this repo that targets only the already-existing scale/gate tensors.
3. The latest high-performing scripts already isolate many of those tensors and keep them in higher precision, making them natural low-byte adaptation handles.

So this candidate is intentionally a **nearby gap** rather than a broad rewrite: it preserves the strongest known structure and changes only *which parameters* move during legal TTT.

## Prior repository experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best overall stack in the repo
  - establishes LeakyReLU(0.5)^2 + legal score-first TTT + Parameter Banking / Parallel Muon as the current frontier
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - confirms that compression-aware post-training details like GPTQ-lite and EMA remain high-value
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - useful warning that ideas which do not slot cleanly into the compiled code path can silently become no-ops
- `records/track_10min_16mb/2026-03-17_LoRA_TTT/`
  - shows the repo has already explored adapter-like TTT, but not this specific control-only variant

## External research that informed the choice

- **Tent** (`arXiv:2006.10726`) updates channel-wise affine parameters at test time and shows that small normalization/affine updates can be enough for useful online adaptation.
- **BitFit** (`arXiv:2106.10199`) shows that surprisingly sparse parameter updates can stay competitive with full fine-tuning.
- **(IA)^3 / T-Few** (`arXiv:2205.05638`) argues that scaling a small number of inner activations is a strong parameter-efficient adaptation strategy.

Those papers do not solve this challenge directly, but they strongly support the repo-specific hypothesis that **the right small subset of existing scale/gate parameters may be enough** for useful legal TTT.

I also reviewed recent rotation-heavy quantization directions in the SpinQuant / QuaRot family. They look promising for general PTQ, but they are a worse fit here because they require broader graph-wide quantization machinery than this repository's current self-contained training script.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps that script almost entirely intact and changes only the TTT parameter-selection logic:

1. Added `TTT_PARAM_MODE`, with:
   - `control` (new candidate default)
   - `all` (fallback to the broader original behavior)
2. In `control` mode, legal TTT updates only:
   - low-dimensional tensors
   - named control tensors such as `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `skip_weights`, `smear.gate`, `bigram.scale`, and similar gating/scale parameters
3. Changed the candidate defaults to:
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`
   - `TTT_PARAM_MODE=control`
4. Added `bigram.scale` to the control-pattern list so it stays grouped with the rest of the explicit adaptation knobs.

What did **not** change:

- no root-file edits
- no new infrastructure
- no new adapter modules
- no changes to the banked training core, quantization path, or data pipeline

## Why this differs from existing records and candidates

This is **not** the same as the earlier LoRA-TTT idea, because it adds **no new adapter parameters** and instead reuses the model's existing control tensors.

It is also **not** the same as the current best full-model legal TTT recipe, because the banked matrix weights stay frozen during test-time adaptation. That makes this a cleaner parameter-efficient TTT experiment rather than just another sweep on the full-update recipe.

## How to run / evaluate

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_PARAM_MODE=control TTT_LR=0.002 TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To recover the broader original-style TTT behavior, set `TTT_PARAM_MODE=all`.

## Main expected risks / tradeoffs

- **Under-adaptation risk:** full-model legal TTT may be using bank-weight updates that control-only TTT cannot recover.
- **Retuning risk:** small-parameter adaptation often prefers different learning rates, chunk sizes, or epoch counts than full-model TTT.
- **Wallclock uncertainty:** control-only TTT should reduce optimization overhead, but backward still traverses the full model, so eval-time savings may be smaller than hoped.

## Validation run in this environment

Commands run:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data \
  candidates/202603282343_control-tensor-ttt/train_gpt.py
```

Outcome:

- Passed syntax compilation for the root baseline scripts, `data/`, and this candidate's `train_gpt.py`.

Additional import smoke check:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('candidates/202603282343_control-tensor-ttt/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
```

Outcome:

- Stopped immediately in this container with `ModuleNotFoundError: No module named 'numpy'`.

Why there is no CPU-only runtime smoke test here:

- this environment does not currently have the candidate's runtime Python stack available at import time
- the script is inherited from a CUDA/FlashAttention record path and `main()` explicitly requires CUDA

So the validation here is limited to syntax compilation plus documenting why a real start-up smoke test was not feasible in this container.
