# Cubic MLP Equalization on the LeakyReLU² GPTQ-lite Stack

## Hypothesis

The current top Parameter Golf stack looks more compression-limited than training-limited: the 4-hour non-record run still lands at `1.2074` after quantization, while the stronger March 22-23 records keep winning by improving export behavior rather than changing the core model. This candidate targets that bottleneck directly.

The key observation is that the March 23 leader uses a `leaky_relu(x, 0.5).square()` MLP. For any positive per-channel scale `s`, this activation is degree-2 homogeneous:

```python
phi(s * x) = s**2 * phi(x)
```

So each MLP pair can be rescaled exactly, without changing the unquantized function:

```python
W_up'[j, :] = s_j * W_up[j, :]
W_down'[:, j] = W_down[:, j] / (s_j ** 2)
```

This candidate chooses `s_j` with a cubic-root balance between the `W_up` row magnitude and `W_down` column magnitude, then quantizes the transformed weights with the existing GPTQ-lite/export path. The goal is to reduce int6 roundtrip loss at **zero training-time cost** and with **no extra model parameters**.

## Why it is promising for this repository

1. `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` shows that even very long training still leaves a large post-quant gap.
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` shows that a pure post-training quantization improvement (GPTQ-lite clip search) was worth `-0.0006 BPB` by itself.
3. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` shows the best current stack already uses leaky-ReLU-squared MLPs, which makes an exact degree-2 rescaling possible.
4. At review time there were **no prior experiments under `candidates/`**, so this does not repeat an earlier candidate branch.

## Prior experiments that influenced this candidate

- **March 23 leader** (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`): strongest overall base stack; contributed the leaky-ReLU² MLP, parameter banks, legal TTT, and GPTQ-lite export path.
- **March 22 GPTQ-lite record** (`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`): strongest evidence that small export-only quantization improvements are still worthwhile.
- **March 21 partial-RoPE/LN-scale record** (`records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`): reminder to keep the strong late-stack architectural refinements intact instead of swapping the model.
- **4-hour non-record baseline** (`records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`): evidence that quantization remains a first-order bottleneck even when training improves.

## External research that informed it

- **SmoothQuant** (Xiao et al., 2022, arXiv:2211.10438): motivates equivalent offline transformations that shift quantization difficulty without changing model behavior.
- **AWQ** (Lin et al., 2023/2024, arXiv:2306.00978): motivates channel-wise scaling as a weight-only quantization aid, especially when only a subset of channels dominate error.
- **AffineQuant** (Ma et al., 2024, arXiv:2403.12544): supports the broader idea that PTQ can benefit from optimizing an equivalent transformed parameterization, not just raw clipping.
- **SASQ** (Mao et al., 2025, arXiv:2512.14481): reinforces the value of lightweight static quantization factors, even when the main weights themselves are not retrained.

This candidate deliberately keeps the transformation **calibration-free** and **weight-only**, because the repo’s current exporter is simple, deterministic, and already near the challenge sweet spot.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Export-time degree-2 MLP equalization**
   - Added `MLP_EQUALIZE`, `MLP_EQUALIZE_STRENGTH` (default `1.0` for exact cubic-root balancing), and `MLP_EQUALIZE_MAX_SCALE` (default `2.5`).
   - Before `mixed_quantize_int6`, the script unbanks the MLP weights, applies the cubic-root hidden-channel rescaling, and then quantizes the transformed weights.
2. **Repo-root-aware defaults**
   - `DATA_PATH` and `TOKENIZER_PATH` now default to the repository root even when the script is run from inside the candidate directory.
3. **CPU-safe smoke path**
   - Added `SMOKE_TEST=1` to run a tiny CPU forward/export/dequantize/forward roundtrip without datasets or CUDA.
4. **Attention fallback for local/non-FlashAttention runs**
   - If `flash_attn_interface` is unavailable or the model is on CPU, attention falls back to PyTorch SDPA. This keeps local validation working, but the intended fast path for leaderboard-style Hopper runs is still FlashAttention 3 when that interface is available in the runtime.

Everything else is intentionally kept as close as possible to the March 23 record so the candidate isolates the export idea.

## How to run or evaluate it

### CPU smoke validation

From the repository root:

```bash
python -m venv /tmp/gh-aw/agent/pg-venv
/tmp/gh-aw/agent/pg-venv/bin/pip install -r requirements.txt
SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python candidates/202604091814_cubic-mlp-eq/train_gpt.py
```

### Full 8xH100-style training run

From `candidates/202604091814_cubic-mlp-eq/`:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MLP_EQUALIZE=1 MLP_EQUALIZE_STRENGTH=1.0 MLP_EQUALIZE_MAX_SCALE=2.5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The new behavior happens only in the export path, so training cost should remain effectively unchanged. On the intended challenge runtime, FlashAttention 3 should be available; if it is missing locally, the script falls back to SDPA and will likely run slower.

## Main expected risks and tradeoffs

- The equalizer is **heuristic and calibration-free**. It may be weaker than activation-stat-driven AWQ/SmoothQuant variants.
- The current exporter quantizes `mlp_down` per row, while the compensating transform touches its columns, so the cubic-root balance is only an approximation to the true quantization objective.
- Over-aggressive scaling can hurt compressibility or move error from `W_up` into `W_down`; `MLP_EQUALIZE_STRENGTH` and `MLP_EQUALIZE_MAX_SCALE` will likely need tuning.
- The idea improves only the **quantized export**. If the March 23 stack is already dominated by legal TTT gains rather than quantization loss, the net BPB win may be modest.

## Validation commands and outcomes

- `python -m compileall candidates/202604091814_cubic-mlp-eq/train_gpt.py`  
  - **Passed**
- `python -m venv /tmp/gh-aw/agent/pg-venv && /tmp/gh-aw/agent/pg-venv/bin/pip install --quiet -r requirements.txt`  
  - **Passed** in a session-local validation environment
- `SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python candidates/202604091814_cubic-mlp-eq/train_gpt.py`  
  - **Passed** with `smoke_test:ok loss:4.8841 roundtrip_loss:4.8841 quant_bytes:595372 eq_layers:2`
- `cd candidates/202604091814_cubic-mlp-eq && SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python train_gpt.py`  
  - **Passed** with the same roundtrip result when executed from the candidate directory
- `cd candidates/202604091814_cubic-mlp-eq && /tmp/gh-aw/agent/pg-venv/bin/python - <<'PY' ...`  
  - **Passed**; default paths resolved to the repository-root dataset/tokenizer locations:
    - `/home/runner/work/parameter-golf/parameter-golf/data/datasets/fineweb10B_sp1024`
    - `/home/runner/work/parameter-golf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`
