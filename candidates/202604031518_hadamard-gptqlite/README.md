# Hadamard GPTQ-lite on the LeakyReLU² + Legal TTT stack

## Hypothesis

The current best record already trains well enough that a meaningful part of the remaining gap is in the **post-training int6 roundtrip**, not in raw training loss alone. A lightweight, output-preserving **Hadamard-basis search inside GPTQ-lite** should reduce quantization error for the largest attention/MLP matrices without changing the training stack or adding learned parameters.

In short: **keep the strongest in-repo model, make the export path smarter**.

## Why this is promising for this repository

Repository evidence points in the same direction:

- The leaderboard improved repeatedly through **better packing/compression-aware export**, not just bigger training runs:
  - baseline int8 roundtrip: `1.2244` in `records/track_10min_16mb/2026-03-17_NaiveBaseline/README.md`
  - XSA + EMA: `1.1271` in `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`
  - Partial RoPE + LN scale: `1.1248` in `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - GPTQ-lite clip search + EMA + warmdown3500: `1.1233` in `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - LeakyReLU² + legal TTT + Parallel Muon: `1.1194` in `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- The non-record 4-hour run still suffered a quantization gap despite much longer training, which suggests export quality is still worth attacking: `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md`
- The current best pure export-only idea in-repo is still the **per-row GPTQ-lite clip percentile search** from `2026-03-22`, so there is room for a stronger but still local quantization improvement.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-17_NaiveBaseline/` for the root architecture and artifact target.
2. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` for the 11-layer XSA + EMA backbone.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for Partial RoPE + LN scale.
4. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` for GPTQ-lite clip search as the direct base export idea.
5. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` for the strongest full stack to fork.

There were **no prior `candidates/` directories** when this candidate was created.

## External research that informed it

- **SmoothQuant** (`arXiv:2211.10438`) argues that offline, mathematically equivalent transforms can move quantization difficulty into a friendlier basis without changing the model function.
- **QuaRot** (`arXiv:2404.00456`) shows that **Hadamard-style rotations** can remove hidden-state outliers and make low-bit LLM quantization much easier, including nearly lossless 6- and 8-bit cases.
- **SpinQuant** (`arXiv:2405.16406`) shows that the *choice of rotation matters*, and that better rotations measurably improve low-bit accuracy over naive random choices.

This candidate takes the smallest repo-compatible slice of those ideas: **search between identity and a fixed block-Hadamard basis per large matrix, using reconstruction MSE to decide whether the rotated basis is better before storing int6 weights**.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in `train_gpt.py`:

1. **Hadamard-aware GPTQ-lite**
   - During int6 export, each 2D attention/MLP matrix now searches over:
     - the original basis
     - a **normalized block-Hadamard transform** on the input dimension (`INT6_HADAMARD_BLOCK_SIZE=512` by default)
   - The script keeps whichever basis yields lower reconstruction MSE after dequantization back into the original basis.
   - Rotation metadata is stored per tensor and inverted during load/eval.
2. **Runnable from the candidate directory**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the candidate file back to the repository root.
3. **CPU-safe smoke mode**
   - `SMOKE_TEST=1` runs a dummy forward pass and a small quantize/dequantize roundtrip on CPU.
4. **Attention fallback for local smoke**
   - If `flash_attn_interface` is unavailable or the run is not on CUDA, attention falls back to `torch.nn.functional.scaled_dot_product_attention`.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202604031518_hadamard-gptqlite
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Relevant knobs:

```bash
INT6_HADAMARD_ROT=1
INT6_HADAMARD_BLOCK_SIZE=512
```

CPU-only smoke path:

```bash
cd candidates/202604031518_hadamard-gptqlite
SMOKE_TEST=1 SMOKE_SEQ_LEN=16 python train_gpt.py
```

## Main expected risks and tradeoffs

- The candidate optimizes **quantization error**, not training dynamics, so gains may be smaller than a new training trick if the current bottleneck is elsewhere.
- Fixed Hadamard blocks are much simpler than the learned rotations in SpinQuant, so the effect may be modest or uneven across layers.
- Lower reconstruction MSE does not guarantee lower **val_bpb** after full roundtrip + sliding eval + TTT.
- The rotated basis could interact differently with `lzma` compression, so artifact size should still be checked carefully.
- Export time increases because each large matrix now evaluates an extra basis candidate.

## Validation

Commands run during candidate creation:

```bash
python -m compileall candidates/202604031518_hadamard-gptqlite/train_gpt.py
```

Outcome:

- **Passed**.

Attempted smoke validation:

```bash
cd candidates/202604031518_hadamard-gptqlite
SMOKE_TEST=1 SMOKE_SEQ_LEN=16 python train_gpt.py
```

Outcome:

- The candidate now defers `numpy` and `sentencepiece` imports outside the smoke path, so `SMOKE_TEST=1` only needs PyTorch.
- This workflow runner still did not have `torch`, and a follow-up attempt to create an isolated venv and install a CPU `torch` wheel was blocked by the runner's proxy/network policy, so a real CPU smoke run was **not feasible in this environment**.
