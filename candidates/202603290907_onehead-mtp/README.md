# One-Head MTP on the LeakyReLU2 + Legal TTT Stack

## Hypothesis

Add a **single training-only multi-token prediction (MTP) head** to the current strongest 11-layer stack so the trunk learns a slightly richer future-aware representation during training, while keeping the exported artifact essentially unchanged because the auxiliary head is dropped before serialization.

In concrete terms, this candidate predicts one extra future token beyond the standard next-token target (`MTP_NUM_HEADS=1`) with a modest auxiliary weight (`MTP_LOSS_WEIGHT=0.1`). The main bet is that this improves hidden-state quality enough to help both the int6 roundtrip model and the legal score-first TTT evaluation path.

## Why this is promising for this repository

Recent record progress in this repository has come from **high-leverage improvements that cost little or nothing in artifact bytes**: sliding-window evaluation, EMA, GPTQ-lite clip search, Partial RoPE/LN scaling, and legal TTT. A training-only auxiliary loss follows the same pattern: spend extra training signal to improve the final compressed model without bloating the exported checkpoint.

This repository's strongest scripts already contain latent MTP support, but the record READMEs do not show a submission that actually turns it on. That makes MTP a strong next candidate because it is:

- already compatible with the winning code path,
- cheap to implement surgically,
- naturally excluded from the export artifact, and
- grounded in recent research showing that future-token supervision improves the trunk's representations.

## Prior repository runs that influenced this candidate

### Primary base

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best overall current record in this checkout,
  - already includes parameter banking, GPTQ-lite int6 + lzma export, LeakyReLU(0.5)^2, legal score-first TTT, and dormant MTP support.

### Supporting record lineage

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best non-TTT training/export stack and the clearest precursor for the modern 11-layer recipe.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - establishes Partial RoPE + LN scaling as durable zero-byte wins.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - earlier 11-layer XSA/EMA stack where the dormant MTP path is also present.

### Prior candidates

- No prior `candidates/` directory existed in this checkout before this addition.

## External research that informed the idea

- **Better & Faster Large Language Models via Multi-token Prediction** (`arXiv:2404.19737`)
  - argues that adding auxiliary future-token heads improves sample efficiency and downstream capability without changing the shared trunk at inference.
- **Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential** (`arXiv:2507.11851`)
  - provides more evidence that standard autoregressive LMs already encode future-token information that can be unlocked with lightweight auxiliary machinery.
- **Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing** (`arXiv:2603.17942`)
  - shows even plain next-token LMs contain latent MTP structure, which increases confidence that a small explicit MTP auxiliary head can help this tiny model regime.
- **Thinking into the Future: Latent Lookahead Training for Transformers** (`arXiv:2603.20219`)
  - reinforces the broader idea that future-looking supervision can improve sequence modeling by encouraging the model to form better internal predictions before committing.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- enabled **one** auxiliary MTP head by default:
  - `MTP_NUM_HEADS=1`
  - `MTP_LOSS_WEIGHT=0.1`
- registered the MTP head weights in the replicated AdamW path so the auxiliary loss actually updates the head and can start shaping the trunk during training
- set the script defaults closer to the intended candidate run:
  - `ITERATIONS=9000`
  - `BIGRAM_VOCAB_SIZE=1536`
  - `TTT_ENABLED=1`
  - `TTT_FREEZE_BLOCKS=0`
- made default data/tokenizer paths resolve from the script location, so the script can be run directly from this candidate directory without manually pointing back to the repo root.

Everything else intentionally stays on the strongest known stack:

- 11 layers / 512 width / 8 heads / 4 KV heads,
- 3x MLP with LeakyReLU(0.5)^2,
- Partial RoPE (16 dims), LN scale, XSA on the last 4 layers,
- shared value embeddings on layers 9-10,
- EMA + tight SWA,
- parameter banking + parallel Muon,
- GPTQ-lite int6 export with lzma,
- legal score-first TTT.

## Why the artifact size should stay competitive

The script already excludes `mtp_heads` from the exported state dict before serialization and re-instantiates the evaluation model with `mtp_num_heads=0`. That means the extra MTP head is **training-only** and should not materially increase the compressed checkpoint size.

The only byte increase here is code/documentation, not model weights. The main tradeoff is therefore **training compute**, not artifact size.

## How to run

From the repository root:

```bash
cd candidates/202603290907_onehead-mtp
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Because this candidate resolves its default `DATA_PATH` and `TOKENIZER_PATH` relative to `train_gpt.py`, it can be launched from this directory directly.

Useful overrides:

```bash
cd candidates/202603290907_onehead-mtp
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a shorter GPU smoke run in the official Parameter Golf environment:

```bash
cd candidates/202603290907_onehead-mtp
RUN_ID=mtp_smoke \
ITERATIONS=4 \
MAX_WALLCLOCK_SECONDS=5 \
VAL_LOSS_EVERY=0 \
TTT_ENABLED=0 \
TRAIN_BATCH_TOKENS=8192 \
TRAIN_SEQ_LEN=256 \
EVAL_SEQ_LEN=256 \
VAL_BATCH_SIZE=8192 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Evaluation notes

By default this candidate keeps the legal score-first TTT path enabled, matching the intended record-style evaluation recipe. If you want to inspect only the pre-TTT / standard quantized path, set `TTT_ENABLED=0`.

## Main expected risks and tradeoffs

- **Training-time overhead:** even one auxiliary head adds extra projection + cross-entropy work each step, so total 600-second step count may drop.
- **Objective interference:** the extra future-token target might improve representations, but it could also slightly hurt the exact next-token objective if the weight is too high.
- **TTT interaction is unknown:** better trunk features may help TTT adaptation, but the auxiliary loss could also change calibration in ways that make the final TTT gain smaller.
- **Tiny-model regime mismatch:** most MTP literature is strongest at larger scales, so the gain here is plausible but unproven.

## Validation

### Completed here

- Syntax check:

```bash
python -m compileall candidates/202603290907_onehead-mtp/train_gpt.py
```

Expected outcome: successful bytecode compilation.

- Environment feasibility check for a runtime smoke test:

```bash
python - <<'PY'
from datetime import datetime, timezone
import importlib.util
print('UTC_NOW=', datetime.now(timezone.utc).strftime('%Y%m%d%H%M'))
for name in ['torch', 'numpy', 'sentencepiece', 'flash_attn_interface']:
    print(name, importlib.util.find_spec(name) is not None)
PY
```

Observed in this workflow environment:

- `torch`: missing
- `numpy`: missing
- `sentencepiece`: missing
- `flash_attn_interface`: missing

### Why no runtime smoke test was executed here

A safe runtime smoke test was **not feasible in this workflow environment** because the Python ML dependencies required by the candidate script are not installed here. The official challenge environment is expected to provide those dependencies, but this container does not.
