# Export-Free MTP Aux Heads on the Parallel-Muon Stack

## Hypothesis

Multi-token prediction (MTP) can improve sample efficiency for this challenge's tiny models without increasing submission size, because the auxiliary heads are only used during training and are explicitly dropped from the exported artifact.

## Why this is promising for this repository

The repo history suggests most easy wins are already taken: sliding-window eval, mixed low-bit export, deeper 11-layer stacks, EMA, XSA, partial RoPE, and LeakyReLU^2 are all already present in the current frontier. That makes a training-time-only signal attractive, especially one that does not consume extra artifact bytes.

The key external motivation is **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737), which reports improved sample efficiency by asking the shared trunk to predict multiple future tokens at once.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - strongest current stack in-repo; used as the direct implementation base.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - strongest pre-TTT stack and evidence that export quality still matters a lot.
- `records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md`
  - useful reminder that eval-only tricks can dominate apparent gains, so this candidate isolates a training-time change instead.

## External research

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** — <https://arxiv.org/abs/2404.19737>

## What changed versus the chosen base implementation

Base implementation:
`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate changes only a small number of things:

- enables `MTP_NUM_HEADS=2` by default
- sets `MTP_LOSS_WEIGHT=0.15` by default
- aligns `BIGRAM_VOCAB_SIZE=1536` with the published top-stack run command
- preserves the existing export path that strips `mtp_heads` before serialization, keeping the auxiliary loss artifact-free
- adds a CPU / non-FlashAttention fallback path so the script can be compile-checked and smoke-tested without the exact leaderboard runtime

## How to run or evaluate

From the candidate directory:

```bash
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The main hypothesis knobs are:

```bash
MTP_NUM_HEADS=2
MTP_LOSS_WEIGHT=0.15
TTT_ENABLED=0
```

That last setting keeps legal TTT off while measuring the pure training-time MTP effect. Once the effect is understood, MTP can be stacked with TTT later.

For local debugging without CUDA / FlashAttention:

```bash
USE_TORCH_COMPILE=0 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
python train_gpt.py
```

## Main expected risks or tradeoffs

- MTP may help larger models more than this compact 11-layer stack, so gains may be small.
- Even though the heads are excluded from export, they still add training-time projection work and optimizer state.
- The best auxiliary weight is likely narrow; too much future-token loss can hurt next-token quality.
- This candidate intentionally avoids mixing in a second new hypothesis so that any signal is easier to interpret.

## Validation

Validation commands run for this candidate:

- `python -m compileall candidates/202603282023_mtp-aux-heads/train_gpt.py`
  - **passed**
- minimal CPU smoke test with a temporary SentencePiece tokenizer and tiny synthetic train/val shards under `/tmp/gh-aw/agent/mtp_smoke`, using `USE_TORCH_COMPILE=0`
  - **passed** on a reduced config (`NUM_LAYERS=2`, `MODEL_DIM=32`, `TRAIN_SEQ_LEN=16`, `ITERATIONS=1`)
  - confirmed the script reaches training, validation, EMA application, export, and int6 roundtrip evaluation on CPU
  - confirmed the export path logs `export_excluding_mtp_params:4096`, showing the auxiliary MTP heads were excluded from the saved artifact as intended
