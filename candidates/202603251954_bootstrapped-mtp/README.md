# Bootstrapped MTP Heads on the 11L GPTQ-lite Stack

## Hypothesis

A lightweight multi-token-prediction (MTP) auxiliary objective should improve the shared hidden state geometry of the current best non-TTT stack without increasing exported artifact size, because the extra heads are dropped before quantization/export.

This candidate uses the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack as the base, then turns on MTP by default with two changes meant to make the objective gentler for a next-token-trained backbone: initialize each MTP head from the main output head instead of zero, and downweight farther-future heads with a geometric decay.

## Why this is promising for this repository

Recent records show that this repo already squeezed a lot out of architecture, quantization, and evaluation: XSA on late layers, Partial RoPE, LN scaling, EMA, and GPTQ-lite each helped, but the frontier gains are now incremental. The repo survey also found that `mtp_num_heads` is already implemented in the strongest non-TTT scripts, yet no existing record appears to claim a positive MTP run.

That makes MTP appealing here: it is a real gap, it does not spend any export bytes when `mtp_heads` are excluded from the artifact, and it keeps the proven 11-layer dense stack intact.

## Prior records and repo patterns that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best pre-TTT training stack in the repo survey: XSA4, Partial RoPE, LN scaling, VE, SmearGate, BigramHash, EMA, GPTQ-lite.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Shows that the leaderboard frontier has shifted toward eval-time tricks; this candidate instead targets a cheaper training-only improvement.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Documents that naive layer recurrence was a bad wall-clock trade here, which pushed this candidate toward a zero-export-cost auxiliary objective instead of more depth reuse.

There were no prior `candidates/` directories in the repository when this candidate was created.

## External research that informed it

- **Medusa** (`arXiv:2401.10774`) shows that extra decoding heads can improve future-token prediction without needing a separate draft model.
- **MuToR** (`arXiv:2505.10518`) argues that multi-token prediction can be useful with negligible parameter overhead and minimal architectural disruption.
- **On multi-token prediction for efficient LLM inference** (`arXiv:2502.09419`) finds that frozen next-token backbones are specialized for NTP and that joint training helps, which motivated the two stabilizers in this candidate: bootstrapping the auxiliary heads from the main output head and emphasizing nearer horizons more than farther ones.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

- `MTP_NUM_HEADS` now defaults to `2`.
- `MTP_LOSS_WEIGHT` now defaults to `0.15`.
- Added `MTP_HEAD_DECAY` (default `0.7`) so the auxiliary loss weights nearer-future heads more heavily than farther-future heads.
- Added `MTP_INIT_FROM_MAIN_HEAD` (default `1`) so each MTP head starts from the main output projection instead of a zero-init classifier.
- Added a tiny `SMOKE_TEST=1` CPU path plus a PyTorch SDPA fallback when `flash_attn_interface` is unavailable; CUDA SDPA backends are enabled automatically in that fallback case.

Everything else stays as close as possible to the proven 2026-03-22 stack, including export behavior that excludes `mtp_heads` from the final artifact.

## How to run or evaluate it

From this candidate directory on an 8xH100 box with the cached FineWeb data already downloaded at the repository root (`../../data/...`), the script now resolves the default dataset and tokenizer paths automatically:

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If your data lives elsewhere, override `DATA_PATH` and `TOKENIZER_PATH` explicitly.

The main candidate-specific knobs are:

```bash
MTP_NUM_HEADS=2
MTP_LOSS_WEIGHT=0.15
MTP_HEAD_DECAY=0.7
MTP_INIT_FROM_MAIN_HEAD=1
```

`MTP_INIT_FROM_MAIN_HEAD=1` is defined for the default tied-embedding configuration (`TIE_EMBEDDINGS=1`). If you untie embeddings, set `MTP_INIT_FROM_MAIN_HEAD=0`.

If the auxiliary loss is too aggressive, the first ablations to try are lowering `MTP_LOSS_WEIGHT` to `0.1` or `0.08`, or setting `MTP_NUM_HEADS=1`.

## Main expected risks and tradeoffs

- MTP may still hurt the base next-token objective if the auxiliary task pulls hidden states away from what the artifact ultimately needs at evaluation time.
- Even with bootstrapped init and horizon decay, the extra heads add training-time optimizer work and may reduce step throughput slightly.
- The repo survey noted that MTP support already existed but was not part of a winning record; that means this is a real gap, but also a sign that it may need hyperparameter tuning rather than working immediately with defaults.

## Validation

Local validation was run from the repository root inside an isolated virtualenv created under `/tmp/gh-aw/agent/pgolf-venv` because the system Python is PEP 668-managed.

- `python -m compileall candidates/202603251954_bootstrapped-mtp/train_gpt.py`
  - Passed.
- `SMOKE_TEST=1 python candidates/202603251954_bootstrapped-mtp/train_gpt.py`
  - Passed with `smoke_test:ok loss:4.7736 logits_shape:(2, 16, 64) exported_tensors:23`.

These checks cover syntax, CPU forward/backward, and the export/dequantize roundtrip used by this candidate's artifact path.
