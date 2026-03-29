# Candidate: Warmdown MTP

## Hypothesis

The current best local stack already gets excellent parameter-efficiency from LeakyReLU^2, Parameter Banking + Parallel Muon, legal score-first TTT, and aggressive int6 export. The most promising *unused* lever in that stack is a training-only multi-token prediction (MTP) auxiliary head: it can add future-token supervision during training, then be dropped from export so artifact size stays unchanged.

This candidate enables a single MTP head by default (`MTP_NUM_HEADS=1`, `MTP_LOSS_WEIGHT=0.15`) on top of the current best record code. The bet is that one-step lookahead supervision improves sample efficiency under the 600s training cap without paying a byte-budget penalty at submission time.

## Why this is promising for this repository

- The repo history shows that cheap evaluation wins were harvested early, then the strongest gains came from improving *training efficiency* while preserving the 16 MB artifact budget.
- The best local record at `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` already contains dormant MTP code paths, but no prior record README in this repo reports an MTP run.
- This makes MTP unusually attractive here: it is a meaningful new idea for this codebase, but it can be tested with a very small diff against the strongest known implementation.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best local result in this repo snapshot (`val_bpb: 1.1194` mean).
  - Established the base stack kept here: LeakyReLU^2 MLP, Parameter Banking + Parallel Muon, legal TTT, GPTQ-lite int6 export.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Reinforced that post-training export quality still matters, but also showed the remaining quantization wins were becoming incremental.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Confirmed that subtle training/runtime interactions matter here; notably, a nominal Late QAT path was compiled away, so new ideas should avoid similar dead-code traps.

## External research that informed the choice

- **Self-Distillation for Multi-Token Prediction** (`arXiv:2603.23911`)
  - Argues that MTP is practical when the auxiliary heads are trained carefully, and reports better multi-token head usefulness with minimal extra training cost.
- **Thinking into the Future: Latent Lookahead Training for Transformers** (`arXiv:2603.20219`)
  - Motivates the broader idea that future-directed supervision can help autoregressive models by spending more training signal on hard next-token decisions.
- **Activation-aware Weight Quantization (AWQ)** (`arXiv:2306.00978`) and **SmoothQuant** (`arXiv:2211.10438`)
  - These were the main quantization alternatives considered during research. They are compelling, but this repo has already spent many iterations squeezing export quality with protected embeddings, GPTQ-lite, and int6/int8 mixes, so a training-signal idea looked like the better next bet.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Enable a **single training-only MTP head by default**.
   - `MTP_NUM_HEADS` default: `0 -> 1`
   - `MTP_LOSS_WEIGHT` default: `0.2 -> 0.15`
   - Export still strips `mtp_heads.*`, so artifact size should remain effectively unchanged.
2. Make the script runnable from the candidate directory.
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root instead of the current working directory.
3. Add a **CPU/SDPA fallback path**.
   - If FlashAttention 3 or CUDA is unavailable, the script falls back to PyTorch SDPA and unfused optimizers.
   - This does not change the intended fast-path on the original Hopper setup, but it makes lightweight local startup checks possible once dependencies are installed.

## How to run

From the repository root:

```bash
cd candidates/202603291646_warmdown-mtp
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate inherits the strongest local defaults from the 2026-03-23 record, but now turns on one MTP head automatically. To ablate the idea cleanly:

```bash
cd candidates/202603291646_warmdown-mtp
MTP_NUM_HEADS=0 MTP_LOSS_WEIGHT=0 SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run for this candidate:

- `python -m compileall candidates/202603291646_warmdown-mtp/train_gpt.py`
  - **Passed**
- Synthetic CPU smoke attempt using a temporary tokenizer + shard pair under `/tmp/gh-aw/agent/`
  - **Blocked by environment**: neither `python` nor `python3` in this workflow image had the required runtime deps (`numpy`, `sentencepiece`, `torch`) installed, so the script could not be executed end-to-end here.

Because of that dependency gap, this candidate was validated statically rather than with a live training step in this environment.

## Main risks and tradeoffs

- Even one MTP head adds compute, so the extra supervision must pay for itself in better early learning or it could reduce the number of training steps completed in 600s.
- If `MTP_LOSS_WEIGHT=0.15` is too strong, the auxiliary target could steal capacity from the main next-token objective and slightly hurt pre-TTT quality.
- The repo’s current best score already benefits from legal TTT at evaluation time, so small pre-TTT gains may be hard to measure without a careful multi-seed comparison.
- The CPU-safe path was added for robustness and smoke testing, but it was not benchmarked for throughput in this dependency-limited environment.
