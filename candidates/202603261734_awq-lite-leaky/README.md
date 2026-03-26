# AWQ-lite export + LeakyReLU² on the 11L GPTQ-lite stack

## Hypothesis

The strongest unexplored direction adjacent to the current non-TTT frontier is to make the **export path more activation-aware**, not just more weight-aware.

The best training-only record in this repo already uses an 11-layer stack with XSA, partial RoPE, LN scaling, VE, EMA, and GPTQ-lite percentile search, and the overall SOTA shows that a one-line swap to **LeakyReLU(0.5)²** is worth carrying forward. My hypothesis is that **tail-batch activation statistics collected during the normal timed training loop**, plus **per-column saliency scaling** before int6 export, will reduce the roundtrip quantization error more effectively than plain weight-MSE clip search alone while staying inside the challenge's 10-minute training / 16MB artifact spirit.

## Why this is promising for this repository

Several repo patterns all point in the same direction:

- The 4-hour non-record baseline made it obvious that post-training compression was a major bottleneck even after much longer optimization.
- The 2026-03-22 record showed GPTQ-lite percentile search was worth about `-0.0006 BPB` at effectively zero training cost.
- The 2026-03-23 record showed **LeakyReLU(0.5)²** can still unlock another small but meaningful gain on a closely related stack.
- Earlier records repeatedly found that sensitive tensors and sensitive directions matter disproportionately under low-bit export.

This candidate keeps the strong 11-layer recipe, adds the proven activation tweak, and then replaces purely weight-only int6 export scoring with a lightweight **AWQ-lite** variant that uses training activations to decide which columns deserve more effective quantization resolution.

## Prior records and experiments that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Direct influences:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for the clean XSA + partial-RoPE + LN-scale stack.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` for the `LeakyReLU(0.5)²` activation.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` and `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/` for the broader lesson that export-aware mixed precision and quantization choices keep funding model quality.
- The full `records/` tree was reviewed before choosing this direction; there were no prior `candidates/` to compare against.

## External research that informed it

The export change is a deliberately small adaptation of ideas from low-bit LLM quantization papers:

- **AWQ** — *Activation-aware Weight Quantization for LLM Compression and Acceleration* (`arXiv:2306.00978`): salient channels should be identified from activation statistics, not weight magnitudes alone.
- **SmoothQuant** — *Accurate and Efficient Post-Training Quantization for Large Language Models* (`arXiv:2211.10438`): offline channel scaling can move quantization difficulty into weight space in a hardware-friendly way.
- **EfficientQAT** (`arXiv:2407.11062`) and **Scaling Law for QAT** (`arXiv:2505.14302`): quantization quality remains heavily shaped by how error is distributed across layers and parameters, especially in deeper language models.
- **SASQ** — *Static Activation Scaling for Quantization-Aware Training in Large Language Models* (`arXiv:2512.14481`): static activation-derived scaling can improve quantized performance without requiring a heavy retraining recipe.

I did **not** try to reproduce those papers wholesale. Instead, I implemented the smallest repo-compatible version of the shared idea: use a short calibration pass over training data to derive per-input-channel saliency, then use that saliency to precondition int6 export.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes two targeted changes:

1. **LeakyReLU(0.5)² MLP**
   - Replaces `relu(x).square()` with `leaky_relu(x, 0.5).square()` in the MLP.
   - This ports the strongest simple activation change from the current overall SOTA back into the strongest non-TTT stack.

2. **AWQ-lite tail-batch int6 export**
   - During the late stage of the **timed training loop**, the script reuses already-seen training batches and runs a few small eager forwards to collect per-input-channel RMS activations for the quantized `attn` and `mlp` linear layers.
   - For each exported int6 matrix, a per-column scale vector is built from those statistics:
     - normalize activation RMS,
     - raise it to `AWQ_ALPHA`,
     - clamp it to `1 / AWQ_SCALE_CLAMP .. AWQ_SCALE_CLAMP`,
     - re-center it to geometric mean 1.
   - The int6 percentile search is then run on `W * col_scale` rather than plain `W`, and the scale vector is stored in the export metadata so the dequantized weight is reconstructed as `dequant(W_scaled) / col_scale`.

In short: the export remains row-wise int6, but the quantizer gets to spend more of its limited resolution on columns that matter more under the observed training activations, **without opening a fresh post-training pass over `train_files`**.

## Files added

- `candidates/202603261734_awq-lite-leaky/README.md`
- `candidates/202603261734_awq-lite-leaky/train_gpt.py`

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603261734_awq-lite-leaky
SEED=1337 \
RUN_ID=awq_lite_leaky_seed1337 \
AWQ_ENABLED=1 \
AWQ_CALIBRATION_BATCHES=8 \
AWQ_CALIBRATION_TOKENS=131072 \
AWQ_ALPHA=0.5 \
AWQ_SCALE_CLAMP=2.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults already keep the 11-layer XSA / partial-RoPE / LN-scale / VE / EMA / GPTQ-lite family from the base record.

Useful knobs for fast ablation:

- `AWQ_ENABLED=0` disables the new export path and falls back to plain GPTQ-lite percentile search.
- `AWQ_ALPHA` controls how strongly activation saliency reshapes the exported weights.
- `AWQ_SCALE_CLAMP` limits how aggressively any one column can be amplified or shrunk.
- `AWQ_CALIBRATION_BATCHES` controls how many already-seen late-training batches are sampled for activation statistics.
- `AWQ_CALIBRATION_TOKENS` is the approximate **per-rank** token slice taken from each sampled late-training batch.

## Expected risks and tradeoffs

- **Artifact-size risk**: the candidate stores an extra fp16 column-scale vector for each activation-aware int6 matrix. I expect this to be modest, but it does consume some of the `~450KB` headroom that the 2026-03-22 record had under 16MB.
- **Calibration quality risk**: if the sampled late-training batches are noisy, the extra scaling metadata could fail to pay for itself.
- **Training/export mismatch risk**: this is intentionally lighter than full AWQ or QAT; it may improve roundtrip quality only marginally.
- **Training-time overhead**: the AWQ statistics are collected inside the timed training loop, so the candidate spends a small amount of its 600-second budget on calibration-quality export metadata.

## Validation

Commands run in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603261734_awq-lite-leaky/train_gpt.py
```

Outcome:

- Passed for the root scripts, `data/`, and the candidate script.

Attempted environment probe:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

Outcome:

- Could not run because this runner does not have `torch` installed.
- Because the candidate script also requires CUDA, FlashAttention, the cached FineWeb shards, and the SentencePiece tokenizer, a meaningful CPU-only smoke test was **not feasible in this environment without adding new infrastructure**.
