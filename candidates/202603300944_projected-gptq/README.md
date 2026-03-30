# Projected GPTQ Warmdown on the 11L EMA stack

## Hypothesis

A **compile-safe, deployment-matched late quantization warmdown** should improve the strong 11-layer GPTQ-lite/EMA stack more reliably than the older `CastedLinear._qat_enabled` fake-quant branch.

Instead of relying on a training-time branch that can be brittle under `torch.compile`, this candidate **projects the large attention/MLP matrices back onto the same GPTQ-lite int6 manifold used at export** every few late-training steps. I also fold in the later **LeakyReLU(0.5)^2** MLP activation win, since it was the clearest low-cost training improvement after the 2026-03-22 record.

## Why this looks promising for this repo

The repo history points to a clear pattern:

- Compression-aware improvements have driven most of the leaderboard movement after the early sliding-window-eval gains.
- `../../records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` is the strongest **pre-TTT** stack and already proved that better clip selection and EMA are worth real BPB.
- `../../records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly notes that its Late QAT path was effectively dead under `torch.compile` constant folding.
- `../../records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` showed that **LeakyReLU(0.5)^2** is a cheap, real gain on the later stack.

So the underexplored gap is not “invent a new architecture from scratch,” but rather **make the late quantization pressure real on a strong existing stack without dragging in the full parameter-banking + TTT complexity of the current record**.

## Prior records that informed this candidate

Primary base implementation:

- `../../records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Key influences:

- `../../records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - partial RoPE + LN scale are real wins
  - the original late-QAT branch was not actually helping under compile
- `../../records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - LeakyReLU(0.5)^2 is a proven low-cost activation improvement
- `../../records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`
  - EMA on the 11-layer int6 stack was a meaningful gain

## External research that informed it

The implementation is a small, repo-compatible slice of the broader “quantization-aware / deployment-matched fine-tuning” literature:

- **OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models** (2023)
  - <https://arxiv.org/abs/2308.13137>
  - Takeaway: directly optimize around the eventual quantized deployment, rather than only the fp model.
- **Learned Step Size Quantization** (2019)
  - <https://arxiv.org/abs/1902.08153>
  - Takeaway: late-stage quantization-aware training can reduce the export gap if it matches the real quantizer.
- **PACT: Parameterized Clipping Activation for Quantized Neural Networks** (2018)
  - <https://arxiv.org/abs/1805.06085>
  - Takeaway: clipping-aware quantization pressure is useful during training, not just after it.
- **SmoothQuant** (2022)
  - <https://arxiv.org/abs/2211.10438>
  - Takeaway: deployment-aware rescaling/quantization is often a better use of engineering time than large architectural changes.
- **QuaRot** (2024) and **SpinQuant** (2024)
  - <https://arxiv.org/abs/2404.00456>
  - <https://arxiv.org/abs/2405.16406>
  - Takeaway: quantization quality is often bottlenecked by outlier structure, so making the train/export path more aligned is a promising direction for tiny-model work too.

This candidate does **not** implement full LET/rotation machinery. Instead, it takes the smallest practical step that matches this repo: **late in-place projection with the exact GPTQ-lite row-clip search already used by export**.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four focused changes:

1. **LeakyReLU(0.5)^2 MLP**
   - Swaps `relu(x)^2` for `leaky_relu(x, 0.5)^2`.

2. **Compile-safe projected late QAT**
   - Replaces the old training-time fake-quant branch with a post-optimizer projection step.
   - Once warmdown scale drops below `LATE_QAT_THRESHOLD`, every `QAT_PROJECT_EVERY` steps the script quantizes/dequantizes the large `mlp` and `attn` matrices in-place using the same per-row GPTQ-lite clip search as final export.

3. **Shared projection/export quantizer**
   - `quantize_int6_per_row()` now accepts configurable clip percentile lists.
   - The late projection step and final artifact export both use the same percentile schedule from `QAT_PROJECT_PERCENTILES`, reducing train/export mismatch.

4. **CPU/FlashAttention fallback for local smoke**
   - Adds a small `causal_attention()` helper that falls back to PyTorch SDPA when FlashAttention 3 is unavailable.
   - Adds `SMOKE_TEST=1` path for tiny local model construction / quantization roundtrip in a proper PyTorch environment.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603300944_projected-gptq

DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
SWA_ENABLED=1 SWA_EVERY=50 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
LATE_QAT_THRESHOLD=0.15 QAT_PROJECT_EVERY=4 \
QAT_PROJECT_PERCENTILES=0.9990,0.9995,0.9999,0.99999,1.0 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional tiny smoke path in an environment that already has PyTorch installed:

```bash
cd candidates/202603300944_projected-gptq
SMOKE_TEST=1 python train_gpt.py
```

## Validation

Commands attempted in this workflow:

```bash
python -m compileall candidates/202603300944_projected-gptq/train_gpt.py
```

Outcome:

- **Passed**.

Attempted smoke check:

```bash
SMOKE_TEST=1 python candidates/202603300944_projected-gptq/train_gpt.py
```

Outcome:

- **Not feasible in this runner** because the local environment does not have `torch` installed, so the script cannot be executed here even in smoke mode.
- The candidate still includes the `SMOKE_TEST=1` path for a repo-like environment that has the normal training dependencies.

## Main expected risks and tradeoffs

- **Projection overhead:** in-place GPTQ-lite projection every few late steps adds extra work during warmdown, so the best interval may be larger than `4`.
- **Potential overprojection:** if projection is too frequent, the model may lose useful fp32 freedom before EMA has finished smoothing.
- **Confounded gain source:** LeakyReLU(0.5)^2 and projected warmdown are both active, so any future ablation should isolate them.
- **Still behind the full record stack:** this intentionally avoids parameter banking + legal TTT complexity, so it is meant as a strong next candidate for the simpler pre-TTT family rather than a full replacement for the 2026-03-23 record.
