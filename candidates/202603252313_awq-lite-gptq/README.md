# Activation-Aware GPTQ-lite (AWQ-lite) Candidate

## Hypothesis

The current record family has already squeezed a lot out of weight-only export with int6 QAT, GPTQ-lite clip search, EMA, and careful mixed precision, but it still treats quantization mostly as a **weight-only** problem. This candidate tests whether collecting a small amount of **training-set activation statistics** and using them to apply an **equivalent per-channel rescaling** before GPTQ-lite export can reduce the remaining int6 error without changing the root baseline or introducing new infrastructure.

In short: if the repo's strongest pre-TTT stack is already near the limit of weight-only tuning, an **activation-aware weight transform** may be the next cheap lever.

## Why this is promising for this repository

Repository evidence points in the same direction:

- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/README.md` framed the challenge as largely a **post-training quantization bottleneck**.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` showed some tensors are unusually **quantization-sensitive**.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` pushed the repo to a stronger **GPTQ-lite** export, but still with weight-centric heuristics.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` demonstrated that the strongest recent gains are now small, additive, and usually come from better late-stage behavior.

What I did **not** find in the records or prior candidates was a compact, self-contained **activation-aware export path** in the style of AWQ / SmoothQuant / static activation scaling.

## Prior records that influenced this candidate

This candidate is most directly based on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

The 2026-03-22 record supplied the cleanest strong base implementation: 11 layers, XSA on the last 4 blocks, partial RoPE, LN scaling, shared value embeddings, EMA, and GPTQ-lite export. I kept that structure and changed only the pieces needed for the new export idea.

The 2026-03-23 record contributed one proven training-side tweak that was easy to carry over: `LeakyReLU(0.5)^2` in the MLP.

## External research that informed it

This candidate is grounded in a lightweight adaptation of activation-aware / smoothing quantization work:

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (Lin et al., MLSys 2024, arXiv:2306.00978)
  - Key idea used here: use activation statistics to identify and protect salient channels via an equivalent rescaling, while keeping inference hardware-friendly.
- **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** (Xiao et al., ICML 2023, arXiv:2211.10438)
  - Key idea used here: migrate quantization difficulty across an equivalent transform rather than changing the function of the model.
- **SASQ: Static Activation Scaling for Quantization-Aware Training in Large Language Models** (Mao et al., arXiv:2512.14481)
  - Key idea used here: small static scale factors can be an efficient deployment-friendly knob even when you do not want a large new quantization stack.
- **Post Training Quantization of Large Language Models with Microscaling Formats** (Sharify et al., arXiv:2405.07135)
  - Useful motivation for combining multiple PTQ techniques rather than treating GPTQ/AWQ/SmoothQuant as mutually exclusive.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate adds four focused changes:

1. **Activation-aware calibration pass on training data**
   - After training and EMA application, the script runs a short calibration sweep over a few training batches.
   - Forward pre-hooks record per-input-channel absolute maxima for each `CastedLinear`.
   - Calibration defaults are intentionally small: `AWQ_CALIBRATION_STEPS=12`, `AWQ_CALIBRATION_TOKENS=262144`.

2. **Equivalent per-channel runtime scaling for quantized linears**
   - Each `CastedLinear` owns a small `awq_runtime_scale` buffer.
   - During export, quantized attention/MLP weights are rescaled column-wise and the inverse runtime scale is stored in the module, preserving the full-precision function while making GPTQ-lite see a smoother matrix.
   - The plain `final_model.pt` checkpoint stays as the untransformed post-EMA model; the AWQ-style transform is applied only to the quantized export path.

3. **LeakyReLU(0.5)^2 MLPs by default**
   - The strong activation tweak from the 2026-03-23 record is carried in with `MLP_NEGATIVE_SLOPE=0.5`.

4. **CPU-safe attention fallback for smoke validation**
   - If `flash_attn_interface` is unavailable, attention falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - This is not meant as the leaderboard path, but it makes static/synthetic smoke validation more practical.

## How to run or evaluate it

Run from this candidate directory:

```bash
cd candidates/202603252313_awq-lite-gptq

MLP_NEGATIVE_SLOPE=0.5 \
AWQ_LITE_ENABLED=1 AWQ_ALPHA=0.5 AWQ_SCALE_CLAMP=8.0 \
AWQ_CALIBRATION_STEPS=12 AWQ_CALIBRATION_TOKENS=262144 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The default dataset and tokenizer paths are resolved relative to the repository root, so the script can be launched from inside this candidate directory without patching `DATA_PATH` or `TOKENIZER_PATH`.

## Validation

Validation run in this workflow:

- from the repository root:
  - `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603252313_awq-lite-gptq/train_gpt.py`
  - **Passed**
- from inside `candidates/202603252313_awq-lite-gptq`:
  - `python -m compileall train_gpt.py`
  - **Passed**
- attempted CPU smoke import / synthetic forward
  - **Not feasible in this container** because the workflow image does not have `torch`, `sentencepiece`, or `numpy` installed for `python`/`python3`, and there is no repo-local dependency install path available in this run.

## Main expected risks and tradeoffs

- The activation-aware scales are another small set of fp16 buffers, so artifact bytes go up slightly.
- A short calibration sweep may be noisy or over/under-estimate salient channels.
- The transform is intentionally lightweight and static; it is **not** a full AWQ or SmoothQuant implementation.
- Runtime now includes a per-channel divide before quantized linears during evaluation, so eval may get slightly slower.
- The best alpha / clamp settings may differ for attention versus MLP, but this candidate uses one global setting to stay simple.
