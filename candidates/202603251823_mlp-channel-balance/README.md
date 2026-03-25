# Homogeneous MLP Channel Balance

## Hypothesis

The strongest current Parameter Golf stacks still leave meaningful performance on the table in the export path: recent records kept improving by shrinking the quantization gap with fp16-safe embeddings, int6/int5 exports, GPTQ-lite clip search, and longer warmdown. This candidate targets that same bottleneck, but does it with a mathematically exact transform that is specific to this repository's current MLP.

The key observation is that the current best stack uses `LeakyReLU(0.5)^2` in the MLP. For any positive per-channel scale vector `s`, that activation is exactly 2-homogeneous:

```python
phi(s * z) = (leaky_relu(s * z, 0.5)) ** 2 = s**2 * phi(z)
```

That means we can rescale each MLP hidden channel before export without changing the float model:

```python
up' = diag(s) @ up
down' = down @ diag(s^-2)
```

The candidate searches a tiny exponent grid per MLP layer, picks the scale vector that minimizes a cheap roundtrip proxy MSE after int6 quantization, and then quantizes the balanced weights. The float network stays functionally equivalent; only the post-quantization error changes.

## Why this is promising for this repository

Repository history suggests that quantization-aware export is still a live frontier here:

- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` improved the strong 11-layer family mostly via architecture, but its README explicitly notes the intended late-QAT path was effectively inactive.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` still gained another `-0.0013` BPB mostly from better export/averaging choices, especially GPTQ-lite clip search.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` became the overall best result with `LeakyReLU(0.5)^2`, legal TTT, and Parallel Muon, but it still uses the same basic post-training int6 export structure.

So the pitch here is simple: keep the current record stack, but make the MLP weights easier to quantize at zero training-time cost.

## Prior records and candidates that influenced this

There were no pre-existing `candidates/` directories in the repo at review time.

The main repository influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Chosen as the direct implementation base because it is the current strongest end-to-end submission.
  - Provides the exact `LeakyReLU(0.5)^2` activation that makes the channel-balance transform mathematically exact.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Strong evidence that export-side quantization improvements are still worth chasing on this repo.
  - Motivated keeping the idea focused on PTQ quality instead of adding more training complexity.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Confirms the best pure architectural path remains the 11-layer XSA/Partial-RoPE/LN-scale family.

## External research that informed it

This candidate borrows the general principle of *output-preserving pre-quantization transforms*, but adapts it to this repo's very specific MLP nonlinearity.

- **SmoothQuant** (`arXiv:2211.10438`) showed that quantization difficulty can be migrated offline with mathematically equivalent transformations.
- **AWQ** (`arXiv:2306.00978`) showed that scaling salient channels can materially reduce weight-only quantization error.
- **QuaRot** (`arXiv:2404.00456`) and **SpinQuant** (`arXiv:2405.16406`) showed that output-preserving transforms/rotations can strongly improve low-bit PTQ.
- **FlatQuant** (`arXiv:2410.09426`) reinforced that flattening weight/activation distributions before quantization is still a strong direction even after earlier PTQ work.

The twist here is deliberately smaller and cheaper than those methods: instead of global rotations or activation calibration, it uses a data-free diagonal transform on each MLP pair that is exact because this model's activation is squared and positively homogeneous.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added **homogeneous MLP channel balancing** before int6 export.
   - Unbanks the MLP weights.
   - For each layer, computes row RMS for `mlp.fc.weight` and column RMS for `mlp.proj.weight`.
   - Searches `MLP_BALANCE_EXPONENTS=0.0,0.1666667,0.25,0.3333333,0.5`.
   - Applies the scale that minimizes a local de-transformed roundtrip MSE proxy after int6 quantization.
   - Logs the chosen exponent histogram and scale range.

2. Kept the rest of the high-performing stack intact.
   - LeakyReLU squared activation.
   - Legal score-first TTT.
   - Parameter banking + Parallel Muon.
   - XSA / Partial RoPE / VE / BigramHash / EMA/SWA / lzma export.

3. Added a tiny **`SMOKE_TEST=1`** path.
   - Instantiates a small CPU model that is large enough to trigger real int6 MLP quantization.
   - Verifies that the balance transform preserves float logits and stays close in the bf16-style inference path.
   - Runs an actual quantize/dequantize roundtrip on the balanced weights.
   - This is only for local sanity checks; it does not try to emulate the real 8xH100 path.

4. Added an optional **FlashAttention fallback** to PyTorch SDPA.
   - Normal GPU runs should still use `flash_attn_interface` when available.
   - The fallback exists mainly so the smoke path can run on CPU or non-FlashAttention environments.

## How to run or evaluate it

### Full training/eval run

Use the same environment assumptions as the recent strong records (PyTorch + SentencePiece + `flash_attn_interface`, 8 GPUs):

```bash
cd candidates/202603251823_mlp-channel-balance

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MLP_BALANCE_ENABLED=1 MLP_BALANCE_MAX_SCALE=4.0 \
MLP_BALANCE_EXPONENTS=0.0,0.1666667,0.25,0.3333333,0.5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Local smoke check

```bash
cd candidates/202603251823_mlp-channel-balance
SMOKE_TEST=1 python train_gpt.py
```

This only needs `torch`; the script now defers `numpy` and `sentencepiece` requirements until the real training/evaluation path.

## Validation commands and outcomes

### 1. Python syntax compile

Command:

```bash
python -m compileall candidates/202603251823_mlp-channel-balance/train_gpt.py
```

Outcome:

- Passed in this workflow run.

### 2. Minimal CPU smoke test

Command attempted:

```bash
cd candidates/202603251823_mlp-channel-balance
SMOKE_TEST=1 python train_gpt.py
```

Outcome in this workflow environment:

- Could **not** be completed because the runner's `python` environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).
- The candidate still includes the smoke path, so this command should work in a normal repo environment where the standard training dependencies are installed.

## Main expected risks or tradeoffs

- The balance search uses a **local proxy MSE**, not end-to-end validation BPB, so it may choose transforms that look good for weight roundtrip error but do not translate into better final BPB.
- The transform only targets **MLP weights**. If attention or embedding quantization dominates on this stack, the gain may be small.
- Because the balancing is export-time only, it does not help optimization during training; it only helps the quantized artifact.
- The SDPA fallback is for portability/smoke checks, not performance. Real leaderboard runs should still rely on the FlashAttention path.
