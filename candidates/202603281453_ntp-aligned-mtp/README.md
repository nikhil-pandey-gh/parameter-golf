# NTP-Aligned Scheduled MTP

## Hypothesis

The strongest recent record scripts already carry dormant multi-token prediction (MTP) support, but every logged strong run keeps it disabled (`mtp_num_heads:0`). A carefully aligned, training-only MTP auxiliary loss could improve sample efficiency inside the 10-minute budget without increasing exported artifact size, as long as it stays close to the next-token objective and fades out before the lowest-LR endgame.

## Why this is promising for this repository

This repository's best progression came from cheap, composable gains on top of an already strong 11-layer stack: XSA, EMA, partial RoPE, GPTQ-lite, LeakyReLU(0.5)^2, and legal TTT. The remaining gap looks more like a data-efficiency problem than a basic architecture problem, which makes MTP attractive because it is a train-time objective change with zero required inference-time parameters in the final export.

The key repo-specific detail is that recent record code already excludes `mtp_heads` from export. That means we can spend extra training-only parameters and optimizer state on a stronger objective, then strip them out before quantization and evaluation. This candidate keeps that property and makes the MTP path first-class instead of dormant.

## Influential records and prior candidates

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - strongest current backbone: parameter banking, Parallel Muon, LeakyReLU(0.5)^2, partial RoPE, LN scale, EMA/SWA, and legal TTT.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - clean pre-TTT record showing the immediate prior stack and explicit export-time MTP stripping.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/train.log`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train.log`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train.log`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_seed1337.log`
  - all show `mtp_num_heads:0`, so the code path exists but is not actually part of the winning runs.

There were no prior experiments under `candidates/` when this candidate was created.

## External research that informed this candidate

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - motivates MTP as an auxiliary objective that improves sample efficiency with independent future-token heads.
- Nikhil Bhendawade et al., **"Speculative Streaming: Fast LLM Inference without Auxiliary Models"** (`arXiv:2402.11131`)
  - supports the broader idea that future n-gram style objectives can be parameter-efficient.
- Anastasios Gerontopoulos et al., **"Multi-Token Prediction Needs Registers"** (`arXiv:2505.10518`)
  - emphasizes keeping MTP closely aligned with the next-token objective while adding very little extra machinery.
- Somesh Mehra et al., **"On multi-token prediction for efficient LLM inference"** (`arXiv:2502.09419`)
  - highlights a key caveat: NTP-trained hidden states are specialized, so naive MTP heads are not guaranteed to help. That motivated the aligned initialization and late-stage fadeout in this candidate.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in `train_gpt.py`:

- enable a **single MTP head by default** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15`
- add **NTP-aligned initialization** with `MTP_ALIGN_INIT=1`, copying the tied token projection into the auxiliary head instead of starting from zeros
- add a **schedule**:
  - `MTP_WARMUP_STEPS=400` ramps the auxiliary loss in gradually
  - `MTP_DISABLE_LR_SCALE=0.35` linearly fades it out during the late low-LR phase
- keep MTP **training-only** by continuing to exclude `mtp_heads` from export and by instantiating eval models with `mtp_num_heads=0`
- resolve default `DATA_PATH` and `TOKENIZER_PATH` relative to the candidate script so it can be run from the candidate directory itself
- add a **FlashAttention fallback** to PyTorch SDPA so the module can still be imported and smoke-tested in environments without `flash_attn_interface`

The March 23 stack's legal TTT path is still present, but this candidate is aimed first at improving the train-time backbone. TTT remains off by default.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603281453_ntp-aligned-mtp
RUN_ID=ntp_aligned_mtp \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Because this candidate resolves defaults relative to `train_gpt.py`, it should find:

- `../../data/datasets/fineweb10B_sp1024/`
- `../../data/tokenizers/fineweb_1024_bpe.model`

when launched from the candidate directory inside a normal repository checkout.

Useful knobs:

```bash
# Disable the auxiliary loss entirely
MTP_NUM_HEADS=0

# Keep MTP active longer or shorter
MTP_DISABLE_LR_SCALE=0.50
MTP_DISABLE_LR_SCALE=0.20

# Compare aligned vs zero-init auxiliary heads
MTP_ALIGN_INIT=0

# Layer legal TTT back on after measuring the backbone change
TTT_ENABLED=1
```

## Validation

Commands run in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603281453_ntp-aligned-mtp/train_gpt.py
```

Outcome:

- passed

Attempted additional smoke validation:

```bash
python - <<'PY'
import importlib.util
print(importlib.util.find_spec("torch"))
PY
```

Outcome:

- `None`
- this runner does not have `torch` installed, so a real import/forward smoke test was not feasible here even though the candidate now includes a non-FlashAttention fallback path

## Main risks and tradeoffs

- Even one future-token head adds training FLOPs and optimizer state, so the objective must pay for itself in better sample efficiency.
- The tiny-model / 10-minute regime is exactly where MTP could fail to transfer cleanly; `arXiv:2502.09419` is an explicit warning sign.
- Aligned initialization only applies in the default tied-embedding setup; it may help optimization, but it could also bias the auxiliary head too strongly toward NTP behavior.
- Interactions with legal TTT are untested in this candidate. The safest first experiment is to measure the training-only delta before composing it with score-first TTT.
