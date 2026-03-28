# Mirror-Shared U-Net with Low-Rank Layer Deltas

## Hypothesis

The repository has already mined a lot of gain from better quantization, evaluation, and small architecture knobs, but it has barely explored **structured cross-layer sharing**. My hypothesis is that this codebase's existing U-Net-style encoder/decoder stack is a good fit for **mirror sharing**: reuse the same full-rank block weights across mirrored depths, then recover layer-specific capacity with tiny low-rank deltas.

That should preserve most of the compute profile of a normal deep stack while reducing artifact pressure enough to support a deeper/wider default recipe than the root baseline.

## Why this looks promising here

Two repo observations point in the same direction:

1. The strongest record family moved steadily toward **deeper 10-11 layer models**, bigger MLPs, and better parameter allocation/compression instead of radical attention replacements.
2. A prior negative result said **naive depth recurrence** was promising in principle but lost under the 10-minute budget because it increased step cost too much.

This candidate tries to split the difference:

- keep a normal unrolled decoder-only/U-Net forward pass,
- **share storage**, not steps,
- keep only small per-layer low-rank residuals unique,
- carry forward a couple of cheap knobs that repeatedly helped in records.

## Prior repository runs that influenced this candidate

- Root baseline `train_gpt.py`: clean self-contained starting point with U-Net skips, Muon, tokenizer-agnostic BPB evaluation, and int8+zlib export.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600`: showed that tied embeddings are quantization-sensitive and explicitly called out naive depth recurrence as too slow under the wallclock budget.
- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow`: strong evidence that deeper stacks plus larger MLPs are worth funding when artifact size allows it.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: partial RoPE and layerwise norm scaling were cheap, repeatable wins.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: LeakyReLU(0.5)^2 was a materially positive one-line activation change.

## External research that informed the design

- **ALBERT** ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942)): cross-layer sharing can reduce memory/parameter pressure without collapsing model quality.
- **Universal Transformer** ([arXiv:1807.03819](https://arxiv.org/abs/1807.03819)): recurrent depth can improve expressivity, but in this repo the compute budget means we need a sharing variant that does not add extra unrolled steps.
- **Subformer** ([arXiv:2101.00234](https://arxiv.org/abs/2101.00234)): sandwich-style sharing is better than naive all-layer sharing for generative transformers.
- **ResidualTransformer** ([arXiv:2310.02489](https://arxiv.org/abs/2310.02489)): shared full-rank weights plus small low-rank per-layer residuals are an effective compromise.
- **Basis Sharing** ([arXiv:2410.03765](https://arxiv.org/abs/2410.03765)): cross-layer shared bases become more attractive as compression pressure increases.

## What changed versus the chosen base implementation

Chosen base: the repository root `train_gpt.py`.

This candidate keeps the baseline's training loop, BPB evaluation, Muon optimizer split, U-Net skip structure, and int8+zlib artifact format, but changes the model and validation ergonomics in a targeted way:

- **Mirror-shared block cores**: mirrored encoder/decoder depths reuse the same full-rank attention and MLP weights.
- **Per-layer low-rank deltas**: each logical layer gets its own small adapter matrices so sharing is not purely hard-tied.
- **Default depth/width bias**: defaults move to **11 layers** and **3x MLP** because sharing reduces storage pressure.
- **Partial RoPE**: only the first `ROPE_DIMS` head dimensions receive rotary embeddings.
- **Layerwise norm scaling**: deeper layers damp their normalized activations by `1/sqrt(layer_idx+1)`.
- **LeakyReLU(0.5)^2 MLP** instead of ReLU^2.
- **FP16 tied embedding passthrough** during export by default.
- **CPU smoke mode** (`SMOKE_TEST=1`) with synthetic tokens so the candidate can be sanity-checked without GPUs or FineWeb shards.
- **Alias-aware quantized export** so shared weights are not serialized repeatedly just because they appear under multiple module paths.

## How to run or evaluate it

### Main candidate run

```bash
torchrun --standalone --nproc_per_node=8 candidates/202603280029_mirror-shared-unet/train_gpt.py
```

Useful ablations / knobs:

```bash
NUM_LAYERS=11 \
MLP_MULT=3 \
MIRROR_SHARE=1 \
ADAPTER_RANK=16 \
ADAPTER_ALPHA=16 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LEAKY_RELU_SLOPE=0.5 \
EMBED_KEEP_FP16=1 \
torchrun --standalone --nproc_per_node=8 candidates/202603280029_mirror-shared-unet/train_gpt.py
```

### Local CPU smoke check

```bash
RUN_ID=smoke_candidate \
SMOKE_TEST=1 ENABLE_COMPILE=0 \
ITERATIONS=2 WARMUP_STEPS=0 VAL_LOSS_EVERY=1 TRAIN_LOG_EVERY=1 \
TRAIN_BATCH_TOKENS=256 VAL_BATCH_SIZE=256 TRAIN_SEQ_LEN=32 \
VOCAB_SIZE=128 NUM_LAYERS=4 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2 ADAPTER_RANK=4 ADAPTER_ALPHA=4 ROPE_DIMS=8 MAX_WALLCLOCK_SECONDS=0 \
python candidates/202603280029_mirror-shared-unet/train_gpt.py
```

## Validation run for this candidate

I ran the following lightweight checks in this repo checkout:

1. Syntax check:

```bash
python -m compileall candidates/202603280029_mirror-shared-unet/train_gpt.py
```

Outcome: **passed**.

2. Minimal CPU smoke test (using a temporary virtualenv because the container lacked repo deps):

```bash
python3 -m venv /tmp/gh-aw/agent/pg-venv
/tmp/gh-aw/agent/pg-venv/bin/pip install numpy sentencepiece torch --quiet
RUN_ID=smoke_candidate \
SMOKE_TEST=1 ENABLE_COMPILE=0 \
ITERATIONS=2 WARMUP_STEPS=0 VAL_LOSS_EVERY=1 TRAIN_LOG_EVERY=1 \
TRAIN_BATCH_TOKENS=256 VAL_BATCH_SIZE=256 TRAIN_SEQ_LEN=32 \
VOCAB_SIZE=128 NUM_LAYERS=4 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2 ADAPTER_RANK=4 ADAPTER_ALPHA=4 ROPE_DIMS=8 MAX_WALLCLOCK_SECONDS=0 \
/tmp/gh-aw/agent/pg-venv/bin/python candidates/202603280029_mirror-shared-unet/train_gpt.py
```

Outcome: **passed**. The run completed training, serialized `final_model.int8.ptz`, reloaded it, and finished the round-trip evaluation with:

- `final_int8_zlib_roundtrip val_loss: 4.85279417`
- `final_int8_zlib_roundtrip val_bpb: 7.00110208`

These smoke metrics are synthetic and only confirm correctness/startup, not challenge quality.

## Main expected risks / tradeoffs

- **Sharing may underfit** if `ADAPTER_RANK` is too small; the idea depends on the low-rank deltas restoring enough layer specialization.
- **Compute is not reduced**, only storage. This candidate still needs real H100 step-time measurement and likely batch-size tuning before it can compete with the best 10-minute records.
- **Export correctness matters** more than usual because shared tensors appear under multiple logical paths; this candidate adds alias-aware export specifically to avoid bloating the artifact.
- **Best follow-up experiments** are likely adapter-rank sweeps, mirror-sharing ablations, and checking whether the saved parameter budget is better spent on batch size, sequence length, or additional width.
