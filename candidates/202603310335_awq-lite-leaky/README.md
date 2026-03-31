# AWQ-lite GPTQ + LeakyReLU^2

## Hypothesis

The strongest pre-TTT record stack in this repository is already highly optimized on architecture and training, but the quantization/export path still matters a lot. The hypothesis for this candidate is that **activation-aware channel scaling before the mixed int6 export** can protect salient input channels in the same spirit as AWQ/SmoothQuant, narrowing the post-training quantization gap with only a small metadata cost.

I also keep the recent **LeakyReLU(0.5)^2** MLP activation because the current best record shows it is a strong low-cost improvement.

## Why this looks promising for this repository

Repository evidence points to quantization as a major bottleneck and opportunity:

- The current baseline already exports with post-training int8 compression, so export quality is part of the core design, not an afterthought.
- The strongest non-TTT record before the latest evaluation-heavy stack is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which improved specifically by making the int6 export smarter.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` is especially informative: it reaches a much better pre-quant loss than the baseline, but still degrades badly after compression, which strongly suggests that better export can unlock real gains.
- The repo sweep also showed that AWQ/SmoothQuant/QuaRot/SpinQuant-style activation- or outlier-aware quantization ideas do **not** appear in prior records or candidates.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - 11-layer stack
  - BigramHash + SmearGate
  - XSA on the last 4 layers
  - Partial RoPE + LN scale
  - EMA on top of the quantization-aware export path
  - GPTQ-lite percentile search for int6 export

Additional ingredients borrowed conceptually:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - LeakyReLU(0.5)^2 MLP activation
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Partial RoPE and LN scale remain part of the chosen base
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - XSA + EMA direction
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - the enduring SmearGate + BigramHash pattern

## External research that informed it

This candidate is primarily motivated by adapting activation-aware PTQ ideas to the repository's existing mixed int6 export path:

- **AWQ** — Activation-aware Weight Quantization for LLM Compression and Acceleration, arXiv:2306.00978
  - Key idea used here: activation statistics identify salient channels better than weights alone; scaling those channels before quantization can reduce error.
- **SmoothQuant** — SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models, arXiv:2211.10438
  - Key idea used here: migrate outlier burden into weights via an equivalent transformation before quantization.
- **QuaRot** — arXiv:2404.00456
  - Important evidence that outlier suppression via equivalent transforms can materially improve low-bit quantization.
- **SpinQuant** — arXiv:2405.16406
  - Reinforces the same theme with more recent results, though its learned rotations are more invasive than I wanted for this repo iteration.

I chose an **AWQ-lite** path rather than a rotation-based one because it fits the repository's current code structure much better: it can be layered directly onto the existing GPTQ-lite export without adding broad new infrastructure.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate adds:

1. **LeakyReLU(0.5)^2** in the MLP by default.
2. **AWQ-lite calibration** after training and EMA application:
   - sample a few training batches,
   - collect mean absolute input activation statistics for large `CastedLinear` modules in attention/MLP paths and aggregate them across ranks,
   - try a small grid of scaling exponents,
   - quantize the scaled weights with the existing GPTQ-lite row-percentile search,
   - keep the scale that minimizes reconstructed weight MSE.
3. **Per-matrix input-channel scales** stored only when they actually help, then folded back during round-trip dequantization.
4. **Repo-root-relative defaults** for dataset and tokenizer paths so the script can be run directly from this candidate directory.
5. **Optional `flash_attn` fallback** to PyTorch SDPA so import/runtime behavior is less brittle when the FA3 binding is absent.
6. **`SMOKE_TEST=1` mode** for local lightweight sanity checks when PyTorch is available.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603310335_awq-lite-leaky
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Representative knobs to ablate quickly:

```bash
AWQ_ENABLED=0
AWQ_CALIBRATION_STEPS=8
AWQ_CALIBRATION_TOKENS=131072
AWQ_ALPHA_CANDIDATES=0.0,0.25,0.5,0.75,1.0
AWQ_MAX_SCALE_FACTOR=4.0
MLP_NEGATIVE_SLOPE=0.5
```

The script defaults `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so running from the candidate directory should work without extra path edits as long as the standard dataset/tokenizer files are present.

For a lightweight local check on a machine that has PyTorch installed:

```bash
cd candidates/202603310335_awq-lite-leaky
SMOKE_TEST=1 python train_gpt.py
```

## Main expected risks or tradeoffs

- **Artifact size risk**: AWQ-lite adds extra per-input-channel scale vectors for some int6 matrices. The metadata cost should be small, but it is not free under the 16MB budget.
- **Neutral or negative gain risk**: the current GPTQ-lite export is already strong; activation-aware scaling may be redundant on some matrices.
- **Calibration sensitivity**: the method uses a small number of training batches for saliency estimation. That is cheap and simple, but could be noisy.
- **Throughput risk**: the training path remains close to the strong pre-TTT stack, but fallback attention paths are slower than FlashAttention 3 if FA3 is unavailable.
- **No full GPU result yet**: this candidate is an implementation-ready hypothesis, not a validated leaderboard improvement.

## Validation run for this candidate

Commands attempted on this runner:

```bash
python -m compileall candidates/202603310335_awq-lite-leaky/train_gpt.py
SMOKE_TEST=1 python candidates/202603310335_awq-lite-leaky/train_gpt.py
```

Outcomes:

- `python -m compileall ...` **succeeded**.
- `SMOKE_TEST=1 python ...` could not be completed on this GitHub Actions runner because the environment does not have the repository's runtime dependency `torch` installed. The smoke mode itself is implemented in the script, but it could not be exercised here without that dependency.
