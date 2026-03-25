# Candidate: Multi-Token Prediction on the 11L EMA + GPTQ-lite stack

## Hypothesis

Add a small **training-only multi-token prediction (MTP)** auxiliary loss to the strongest non-TTT 11-layer stack so the trunk learns more per token and per wall-clock minute, while leaving **evaluation and export unchanged**.

The key bet is that this repository has already squeezed a lot out of quantization, evaluation, and late-stack architectural tweaks. The next strong adjacent move is to improve **sample efficiency during training** without paying extra artifact bytes at export time.

## Why this is promising for this repository

Three pieces of repository evidence point in this direction:

1. The best recent non-TTT line already converged on a strong export recipe: 11 layers, 3x MLP, XSA on late layers, partial RoPE, EMA, GPTQ-lite, and sliding-window eval. That stack reached `1.1233 val_bpb` in `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`.
2. Longer or recurrent-style structural changes are risky under a fixed 10-minute budget. The 1x5090 non-record sweep explicitly found naive layer recurrence worse because extra depth reduced total optimizer steps in fixed wall-clock time.
3. The 2026-03-22 code already contains dormant **training-only MTP heads** and already excludes them from export, which makes MTP unusually cheap to try here.

This makes MTP a better next candidate than heavier changes like full depth sharing, BitNet-style retraining, or new low-bit kernels.

## Prior records and candidates that influenced this candidate

There were **no prior `candidates/` directories** in the repository when this candidate was created.

This candidate is based most directly on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest clean non-TTT base stack in this repo
  - already contains the MTP training path plus export-time stripping of MTP heads
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - useful reminder that some appealing QAT ideas can disappear under `torch.compile`, so a pure MTP experiment is a cleaner next step
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - shows that the frontier is now being pushed by training/eval refinements on top of an already-strong quantized stack
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative evidence against recurrence/depth reuse under strict wall-clock limits

## External research that informed the choice

Primary source:

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (2024), <https://arxiv.org/abs/2404.19737>
  - motivates using auxiliary future-token heads to improve learning efficiency from the same trunk states

Other researched but not chosen as the first next step:

- Zhenzhong Lan et al., **"ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"** (2019), <https://arxiv.org/abs/1909.11942>
  - cross-layer sharing is attractive under a byte budget, but repository evidence says naive recurrence can lose badly on a fixed-time budget
- Mostafa Dehghani et al., **"Universal Transformers"** (2018), <https://arxiv.org/abs/1807.03819>
  - similar tradeoff: depth reuse may help parameters but can hurt total step count
- Shuming Ma et al., **"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"** (2024), <https://arxiv.org/abs/2402.17764>
  - interesting long-term direction, but too much infrastructure change for the next repository-local candidate
- Bita Darvish Rouhani et al., **"Microscaling Data Formats for Deep Learning"** (2023), <https://arxiv.org/abs/2310.10537>
  - promising for sub-8-bit training, but again broader than the small, direct adaptation this repository favors

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **MTP is enabled by default**.
   - `MTP_NUM_HEADS` now defaults to `2`, so the model trains on the normal next-token loss plus auxiliary `+2` and `+3` token heads.
2. **Auxiliary loss weighting is horizon-aware**.
   - Added `MTP_LOSS_DECAY` (default `0.5`) so nearer-future targets matter more than farther-future ones.
   - The total MTP contribution is still controlled by `MTP_LOSS_WEIGHT` (default `0.15`).
3. **FlashAttention import now has a safe fallback**.
   - If `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA. This does not change the intended GPU path, but it makes import-time inspection and small CPU smoke tests possible in environments that have PyTorch but not FlashAttention.
4. **Added explicit logging for the active attention backend and MTP settings**.

Everything else stays close to the 2026-03-22 recipe so the experiment isolates the MTP hypothesis.

## How to run or evaluate it

From the repository root:

```bash
SEED=1337 \
RUN_ID=mtp_candidate_seed1337 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202603252127_multi-token-prediction/train_gpt.py
```

If you prefer to `cd` into the candidate directory first, pass the dataset and tokenizer paths explicitly:

```bash
cd candidates/202603252127_multi-token-prediction

SEED=1337 \
RUN_ID=mtp_candidate_seed1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
# Disable MTP entirely for an A/B check
MTP_NUM_HEADS=0

# Stronger or weaker auxiliary pressure
MTP_LOSS_WEIGHT=0.10
MTP_LOSS_WEIGHT=0.20

# Bias the auxiliary loss more toward the nearest future token
MTP_LOSS_DECAY=0.25
MTP_LOSS_DECAY=0.75
```

Evaluation/export behavior remains the same as the base stack:

- EMA-applied checkpoint is exported
- MTP heads are excluded from the final exported model state
- mixed quantization + GPTQ-lite export path remains intact
- sliding-window evaluation remains the main quality readout

## Main expected risks and tradeoffs

- **Training overhead**: MTP adds extra output-head work during training, so too many heads or too much weight can reduce total steps and erase the gain.
- **Auxiliary-task mismatch**: improvements in trunk learning may not translate cleanly to the single-step next-token metric used at evaluation.
- **Tuning sensitivity**: the best MTP horizon count and loss weight are likely narrow; `2` heads with modest loss weight is just the first safe setting.
- **No export-time benefit by itself**: unlike better quantization, MTP only helps if it improves the trunk representation enough during training.

## Validation

Commands run for this candidate in this environment:

```bash
python -m compileall candidates/202603252127_multi-token-prediction/train_gpt.py
```

Outcome:

- **Passed**.

Attempted smoke check:

```bash
python3 - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202603252127_multi-token-prediction/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcome:

- **Not feasible in this runner** because both `python` and `python3` are missing the `torch` package (`ModuleNotFoundError: No module named 'torch'`).
- Because of that environment limitation, I could only complete syntax validation here, not an import-time or forward-pass smoke test.
