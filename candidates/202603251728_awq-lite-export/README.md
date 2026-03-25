# Candidate: AWQ-lite activation-aware int6 export

## Hypothesis

The record lineage in this repository has already extracted most of the obvious architecture wins, and the remaining gains are increasingly coming from **better post-training export under the 16MB cap** rather than from larger structural changes. This candidate tests whether an **AWQ-lite** export step can improve the existing GPTQ-lite int6 pipeline by using a few post-training **train-set calibration batches** to identify salient input channels and rescale them before int6 quantization.

The core bet is that this repository's current strong 11-layer stack is now **quantization-limited**: the easiest remaining win is to preserve more of the trained fp/bf16 model during export, without paying meaningful extra training cost.

## Why this is promising for this repository

Repository history points strongly at export quality as the remaining frontier:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` showed that preserving sensitive tensors at higher precision was immediately valuable.
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md` and `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/README.md` showed that quantization-aware training and mixed-precision export could buy back enough artifact budget for larger, better models.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that even a zero-training-cost export refinement (GPTQ-lite clip-percentile search) was still worth about `-0.0013` BPB on an already strong stack.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` showed a large **pre-quant / post-quant gap**, reinforcing that compression quality is still a bottleneck.

By contrast, several more invasive architectural ideas already look risky under this challenge's fixed wallclock budget. For example, layer recurrence was explicitly a negative result in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`.

## Which records influenced this candidate

This candidate is primarily based on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`

That base already includes the repository's strongest pre-TTT non-banked stack:

- 11 layers
- 3x MLP
- XSA on the last 4 layers
- Partial RoPE (16/64)
- LN scale
- VE128 on layers 9-10
- EMA + tight SWA
- GPTQ-lite int6 export

I also explicitly pulled in the **LeakyReLU(0.5)^2** MLP activation from the current best record:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

This implementation is motivated by lightweight, activation-aware post-training quantization methods:

- **AWQ** — *Activation-aware Weight Quantization for LLM Compression and Acceleration* (`arXiv:2306.00978`): activation statistics identify salient channels; scaling those channels before weight-only quantization preserves accuracy without backprop or heavy reconstruction.
- **SmoothQuant** — *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* (`arXiv:2211.10438`): equivalent transformations can migrate quantization difficulty across channels and make low-precision export easier.
- **QuaRot** — *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs* (`arXiv:2404.00456`): further evidence that reducing outlier concentration before quantization is a productive path, even when the exact mechanism differs.

This candidate deliberately implements the **smallest repo-native version** of that idea: no new infrastructure, no reconstruction solver, no external kernels, and no change to the training objective.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four focused changes:

1. **Repo-root-relative defaults**

   The candidate script defaults `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root so it can be run directly from the candidate directory without editing paths.

2. **LeakyReLU(0.5)^2 MLP activation**

   The base record used `relu^2`. This candidate switches to the latest record's `leaky_relu(0.5)^2`, which is orthogonal to the export change and already had strong repo evidence.

3. **Activation-stat calibration pass on train data**

   After EMA is applied, the script runs a short inference-only pass over a few training batches and records mean absolute input activation statistics for each `CastedLinear` weight.

4. **AWQ-lite int6 export search**

   For each int6-quantized 2D weight, the exporter now searches over:

   - the existing GPTQ-lite clip-percentile grid,
   - plus an input-channel scaling grid derived from activation statistics.

   The quantizer stores an additional per-weight `input_scale` vector only when the search actually chooses a non-trivial activation-aware rescaling. Dequantization divides that scale back out, so the runtime model stays functionally equivalent apart from the quantization error that the search is trying to minimize.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603251728_awq-lite-export
RUN_ID=awq_lite_export \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
AWQ_ENABLED=1 AWQ_ALPHA_GRID=0.0,0.25,0.5,0.75,1.0 \
AWQ_SCALE_CLAMP=4.0 AWQ_CALIBRATION_BATCHES=8 \
AWQ_CALIBRATION_BATCH_TOKENS=131072 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you run from the candidate directory exactly as above, the script defaults to the repository's `data/` paths automatically. If your dataset or tokenizer live elsewhere, override `DATA_PATH` and `TOKENIZER_PATH` explicitly.

## Main expected risks or tradeoffs

- **Artifact-size risk:** the extra per-weight `input_scale` metadata improves export fidelity, but it also costs bytes. The search has to earn that overhead back.
- **Calibration mismatch risk:** AWQ-style scaling uses train-activation statistics; if those do not line up with the most important evaluation-time activations, gains may be small or negative.
- **Extra evaluation/export time:** the calibration pass is inference-only and short, but it still adds some wallclock versus the base exporter.
- **Interaction risk with LeakyReLU^2:** this candidate composes two ideas that individually look promising, but their combination is not yet validated on challenge hardware.

## Validation

The following lightweight validation was run in this workflow environment:

```bash
cd /home/runner/work/parameter-golf/parameter-golf
python -m compileall candidates/202603251728_awq-lite-export/train_gpt.py
```

Outcome:

- `compileall` succeeded.

I also checked whether a runtime smoke test was feasible on this runner:

```bash
python - <<'PY'
import importlib.util
for name in ('torch', 'flash_attn_interface', 'sentencepiece', 'numpy'):
    print(f"{name}={bool(importlib.util.find_spec(name))}")
PY
```

Observed result on this workflow runner:

- `torch=False`
- `flash_attn_interface=False`
- `sentencepiece=False`
- `numpy=False`

So a meaningful CPU-only or CUDA smoke test was **not feasible here** without adding infrastructure that does not already exist in this repository environment. The candidate script also requires CUDA at runtime, consistent with the record implementations it is based on.
