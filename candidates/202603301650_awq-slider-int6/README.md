# AWQ-Slider Int6 on the clean 11L EMA+GPTQ-lite stack

## Hypothesis

The current leaderboard suggests the next high-signal gains are more likely to come from **better export-time quantization** than from adding another large training/eval subsystem. This candidate keeps the strong `2026-03-22` 11-layer EMA + GPTQ-lite + Partial-RoPE + XSA4 stack, adds the repo-proven **LeakyReLU(0.5)^2** MLP activation, and then replaces plain per-row GPTQ-lite export with a **layer-sensitive activation-aware int6 export**.

The core idea is:

1. shallow and deep transformer layers are usually more quantization-sensitive than the middle layers,
2. activation statistics are a better guide than raw weight magnitude for deciding which channels need finer effective quantization,
3. the repository already wins by squeezing more quality out of the same artifact budget.

So instead of spending extra metadata everywhere, this candidate only calibrates and stores per-input-channel scales for the **first and last two layers**. The quantizer searches over a small set of activation exponents and GPTQ-lite clip percentiles, quantizes `W * s`, and reconstructs `W_hat = dequant(W * s) / s` at load time. That preserves the original linear form while giving sensitive columns a finer effective grid.

## Why this looks promising for this repository

Repository history points in a clear direction:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the strongest clean non-TTT base and leaves real artifact headroom.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` shows **LeakyReLU(0.5)^2** is a real gain on top of a strong stack.
- Earlier records consistently improved by reducing the quantization gap, preserving embeddings, and using compression to fund a better model rather than a more complex optimizer story.

This candidate follows that pattern: keep the good 11L architecture, spend only a little extra metadata where it should matter most, and avoid adding a heavy eval-time system like TTT.

## Prior records and candidates that informed this

There were **no prior `candidates/` directories** in this repository when this candidate was created.

The main repository influences were:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - chosen as the base because it is the strongest low-complexity, non-TTT reference point,
  - already includes EMA, GPTQ-lite clip search, Partial RoPE, LN scaling, XSA, BigramHash, and the 11L 512d layout.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - contributed the **LeakyReLU(0.5)^2** activation change,
  - also reinforced that small, targeted improvements can still matter on top of the 11L family.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - reinforced that the best-performing family is still the 11L Partial-RoPE/XSA line rather than a radically different architecture.

## External research that informed this

This candidate is mainly motivated by four quantization papers:

- **AWQ** — *Activation-aware Weight Quantization for LLM Compression and Acceleration* (`arXiv:2306.00978`)
  - argues that activation statistics identify salient channels better than weight magnitude,
  - protects important channels through equivalent scaling transforms without retraining.
- **SmoothQuant** — *Accurate and Efficient Post-Training Quantization for Large Language Models* (`arXiv:2211.10438`)
  - shows that equivalent channel-wise rescaling can move quantization difficulty across a linear layer.
- **GPTQ** — *Accurate Post-Training Compression for Generative Pretrained Transformers* (`arXiv:2210.17323`)
  - motivates one-shot post-training weight quantization with stronger reconstruction objectives than naive clipping.
- **SliderQuant** — *Accurate Post-Training Quantization for LLMs* (`arXiv:2603.25284`)
  - argues that quantization sensitivity is not uniform across depth and that shallow/deep layers deserve different treatment.

The implementation here is intentionally much simpler than those papers: it adapts the ideas to a single-file Parameter Golf script with the smallest practical amount of new state.

## What changed versus the chosen base implementation

Base chosen: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

- Switched the MLP nonlinearity from `ReLU^2` to **`LeakyReLU(0.5)^2`**.
- Added a short **post-training calibration pass** over training batches to collect per-channel RMS activation statistics.
- Added **layer-window targeting** so only the first and last two layers are eligible for activation-aware scaling.
- Added an **activation-aware int6 search**:
  - search `alpha` in `AQ_ALPHA_CANDIDATES`,
  - search GPTQ-lite clip percentile in `{0.9990, 0.9995, 0.9999, 0.99999, 1.0}`,
  - score each candidate by activation-weighted reconstruction error.
- Stored per-input-channel scales only when the selected transform is non-identity.
- Set `LATE_QAT_THRESHOLD=0` by default so the experiment isolates the export hypothesis instead of depending on an additional late-training interaction.
- Changed default `DATA_PATH` and `TOKENIZER_PATH` to resolve from the repository root so the script can be run directly from this candidate directory.
- Added a fallback SDPA attention path if `flash_attn_interface` is unavailable, although the training script still targets CUDA execution.

## How to run

From the repository root:

```bash
cd candidates/202603301650_awq-slider-int6
RUN_ID=awq_slider_int6 \
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_WD=0.04 ADAM_WD=0.04 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 LATE_QAT_THRESHOLD=0 \
AQ_ENABLED=1 AQ_LAYER_WINDOW=2 AQ_CALIBRATION_STEPS=4 \
AQ_ALPHA_CANDIDATES=0.0,0.35,0.5,0.75 AQ_MAX_SCALE=4.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script defaults `DATA_PATH` and `TOKENIZER_PATH` to the repository-level `data/` directory, so no extra path overrides are needed if you run it from this folder inside a standard repo checkout.

## How to evaluate the idea

The main things to watch in logs are:

- `DIAGNOSTIC post_ema ...` for the pre-export quality,
- `aq_slider:calibration ...` to confirm activation stats were collected,
- `aq_slider:quantized_candidates ... scaled_tensors:...` to see how often non-identity channel scaling was selected,
- final `int6` roundtrip and sliding-window BPB,
- total submission size after compression.

A good outcome would be a smaller quantization gap than the plain GPTQ-lite base while keeping the artifact comfortably below the 16 MB cap.

## Main risks and tradeoffs

- **Metadata overhead**: if too many layers or too many matrices need non-identity scales, the added bytes may eat the gain.
- **Calibration bias**: the calibration batches come from the training stream and are intentionally tiny; they may be too noisy or too narrow.
- **Heuristic search space**: `AQ_LAYER_WINDOW=2` and the alpha grid are reasonable defaults, not fully tuned optima.
- **Interaction uncertainty**: LeakyReLU^2 and activation-aware export are individually plausible, but the combined effect still needs real 8xH100 runs.

## Validation run for this candidate

Validated locally in this repository with:

```bash
python -m compileall candidates/202603301650_awq-slider-int6/train_gpt.py
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603301650_awq-slider-int6/train_gpt.py
```

Outcome:

- both compile-only validation commands completed successfully.
- a CPU-only smoke run was **not** executed in this environment because the trainer still requires CUDA in `main()` and this repository does not provide an existing CPU execution path for the full training loop.

## Suggested next experiments if this lands cleanly

1. Sweep `AQ_LAYER_WINDOW` in `{1, 2, 3}` to trade bytes for quantization accuracy.
2. Try a slightly denser alpha grid around `0.35-0.6`.
3. Compare this export path on the latest TTT-free 11L stack **with and without** LeakyReLU^2 to separate the training and export gains.
4. If the artifact margin stays large, extend activation-aware scaling to only the most sensitive projection types rather than every q/k/v/out/up/down matrix in the selected layers.
