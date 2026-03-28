# SpQR-lite Residual Rescue + LeakyReLU^2

## Hypothesis

The strongest near-term improvement for this repository is to keep the proven 11-layer EMA/XSA/Partial-RoPE stack, add the already-winning `LeakyReLU(0.5)^2` MLP activation, and then spend a tiny amount of artifact budget on **sparse residual corrections** for the worst post-quantization outliers.

The export path already gets strong gains from GPTQ-lite clip search, but it still forces every large matrix through a dense low-bit format. A SpQR-style escape hatch for a very small number of high-error entries could reduce the int6 round-trip gap more efficiently than broad architectural churn.

## Why this is promising for this repository

This challenge is jointly constrained by training time, artifact size, and tokenizer-agnostic evaluation. The recent records show that:

- the best non-TTT stack already lives around the `11L + XSA + EMA + Partial RoPE + GPTQ-lite` recipe,
- small activation or export improvements are still moving the score by meaningful margins,
- training-free or late-stage compression tricks are especially attractive because they preserve the fast training path.

This candidate therefore targets the part of the pipeline where the repo has repeatedly found wins: **export-aware compression**. The residual patch budget is intentionally conservative so it can plausibly fit inside the remaining artifact headroom of the 16 MB limit.

## Prior repository influences

### Main base implementation

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`

This candidate starts from that record because it is the strongest non-TTT implementation in the repo and already contains the stable 11-layer recipe, EMA, GPTQ-lite export, Partial RoPE, XSA, VE, SmearGate, and warmdown tuning.

### Additional repo evidence that informed this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - motivated carrying over `LeakyReLU(0.5)^2`, which was reported as a meaningful standalone gain.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - reinforced keeping Partial RoPE + layerwise LN scaling.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`
  - reinforced the basic 11-layer XSA/EMA direction.

At the time this candidate was created, the repository had **no existing `candidates/` directory**, so there were no prior candidates to inherit from or avoid duplicating.

## External research that informed it

- **GPTQ** (`arXiv:2210.17323`) motivated the repo's existing GPTQ-lite direction: use smarter post-training quantization rather than only denser training.
- **AWQ** (`arXiv:2306.00978`) highlighted that a small set of salient channels can dominate quantization error.
- **SpQR** (`arXiv:2306.03078`) showed that isolating outlier weights in higher precision can preserve quality at nearly the same compression ratio.

This candidate is intentionally a **minimal adaptation** of those ideas to the current codebase: instead of a full new compression framework, it adds sparse fp16 residual patches for a very small number of high-error entries after int6 quantization.

## What changed versus the chosen base

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, this candidate makes four targeted changes:

1. **LeakyReLU(0.5)^2 MLP**
   - swaps the base relu-squared MLP for the activation that the latest record reported as helpful.

2. **SpQR-lite residual rescue during int6 export**
   - after GPTQ-lite per-row int6 quantization, the exporter identifies a tiny top-k set of highest-energy residual entries and stores only those exact corrections in fp16.
   - dequantization reconstructs the dense int6 tensor and then scatters the residual patches back in.

3. **CPU/portable attention fallback**
   - if FlashAttention 3 is unavailable (for example in the smoke test), the script falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - this is only for portability and validation; the intended GPU path remains unchanged.

4. **Candidate-local defaults that work from this directory**
   - default dataset and tokenizer paths resolve relative to the repository root, so the script can be run from the candidate directory directly.

## How to run or evaluate

From the repository root:

```bash
cd candidates/202603281946_spqr-lite-leaky
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
EVAL_STRIDE=64 INT6_RESIDUAL_FRACTION=0.0005 INT6_RESIDUAL_MAX=384 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The script defaults to the repo's standard tokenizer and FineWeb cache paths, so it can be launched directly from this candidate folder.
- `INT6_RESIDUAL_FRACTION` and `INT6_RESIDUAL_MAX` are the new knobs controlling the sparse residual budget.

### Cheap local smoke command

```bash
cd candidates/202603281946_spqr-lite-leaky
SMOKE_TEST=1 python train_gpt.py
```

This smoke path uses synthetic CPU inputs and validates model construction, forward/backward, quantization, dequantization, and the sparse residual rescue branch without requiring GPUs or dataset shards.
It only needs PyTorch plus the standard library because the dataset/tokenizer dependencies are deferred until the full training path.

## Validation

Validation run for this candidate:

- `cd candidates/202603281946_spqr-lite-leaky && python -m compileall train_gpt.py`
  - passed.
- `cd candidates/202603281946_spqr-lite-leaky && SMOKE_TEST=1 python train_gpt.py`
  - passed with:
  - `smoke_test:ok loss:5.5004 roundtrip_loss:5.5004 residual_patches:117`

I did **not** run a full training job in this environment because the workflow runner does not provide the multi-GPU challenge setup or dataset preparation needed for a meaningful end-to-end training/eval run. The smoke path is therefore the safest validation that still exercises the new candidate logic.

## Main expected risks and tradeoffs

- The sparse residual budget may help round-trip loss but still lose on final compressed bytes if the corrections compress worse than expected.
- The `LeakyReLU(0.5)^2` gain was shown on the newer parameter-banked/TTT stack; it may transfer imperfectly to this simpler base.
- The best patch budget is likely sensitive. Too few patches will not move the quant gap; too many patches may erase the artifact-size win.
- This is still a post-training approximation of SpQR/AWQ-style ideas, not a full activation-aware or learned quantization system.

## Suggested next experiments

1. Sweep `INT6_RESIDUAL_FRACTION` and `INT6_RESIDUAL_MAX` against actual compressed artifact size.
2. Compare residual patches on all int6 tensors vs. only MLP projections.
3. Combine this export path with the `BIGRAM_VOCAB_SIZE=3072` setting hinted by the latest record ablations.
4. If the export gain is real, try a slightly more activation-aware patch selection rule instead of pure residual energy.
