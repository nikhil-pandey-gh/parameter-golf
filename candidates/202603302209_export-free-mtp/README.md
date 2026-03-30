# Export-free MTP on the 11L LeakyReLU² + TTT stack

## Hypothesis

The current best script already contains multi-token prediction (MTP) heads and already excludes them from the exported artifact, but the heads are never added to any optimizer. That means the repo has a nearly free training-only knob that is present in code yet effectively inactive.

This candidate fixes that path end-to-end and makes it the main experiment: train with a small auxiliary MTP objective to improve sample efficiency, then drop the extra heads before quantization/export so the final artifact size is unchanged.

## Why this is promising for this repository

- The repo history says the strongest gains come from techniques that improve training or evaluation **without spending extra artifact bytes**, especially sliding eval, legal TTT, and better quantization/export handling.
- Recent MTP work reports better sample efficiency and stronger generative performance from predicting multiple future tokens with auxiliary heads on a shared trunk.
- Unlike recurrence or weight sharing, MTP does **not** reduce the number of optimizer steps available under a fixed 10-minute wallclock. That matters here because a prior recurrence experiment regressed badly once step throughput fell.

## Prior records and experiments that influenced this candidate

- Root baseline: `train_gpt.py`
- Current best record: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- Quantization-focused records:
  - `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- Negative recurrence signal:
  - `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`

The chosen base is the `2026-03-23` record because it is the strongest local stack and already has the machinery to exclude MTP heads from export. I did **not** choose recurrent/shared-depth reuse even though it looks attractive in outside literature, because the repo’s own recurrence sweep was clearly negative under a fixed wallclock budget.

## External research that informed it

- Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (2024): https://arxiv.org/abs/2404.19737
  - Motivates MTP as a sample-efficiency improvement using multiple future-token heads on a shared trunk.
- Zhenzhong Lan et al., *ALBERT* (2019): https://arxiv.org/abs/1909.11942
- Mostafa Dehghani et al., *Universal Transformers* (2018): https://arxiv.org/abs/1807.03819
  - These made recurrent/shared-depth reuse worth considering, but repo-local negative results pushed this candidate away from that path.
- Maximilian Croci et al., *QuaRot* (2024): https://arxiv.org/abs/2404.00456
- Zechun Liu et al., *SpinQuant* (2024/2025): https://arxiv.org/abs/2405.16406
  - Both reinforce that low-bit export remains central in this challenge, which is why this candidate keeps the final exported model unchanged and artifact-neutral.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **MTP is now actually trainable**
   - The base script instantiated `mtp_heads` and logged their parameter count, but never attached them to an optimizer.
   - This candidate adds a dedicated AdamW optimizer group for `mtp_heads`.

2. **MTP defaults are turned on**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

3. **Tiny-model-friendly MTP shaping**
   - New `MTP_LOSS_DECAY` weights nearer future-token heads more heavily than farther ones.
   - New `MTP_INIT=tie` initializes each MTP head from the main token projection / tied embedding instead of starting from an inert zero head.

4. **Validation portability**
   - Added a safe FlashAttention fallback using PyTorch SDPA when `flash_attn_interface` is unavailable or when tensors are not on CUDA.
   - Added `SMOKE_TEST=1` mode for a tiny synthetic CPU forward/backward check that exercises the MTP path without dataset or CUDA requirements.

## How to run or evaluate it

Recommended full run from the candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_LOSS_DECAY=0.7 MTP_INIT=tie \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Minimal synthetic smoke check:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run for this candidate

Commands run while preparing this candidate:

```bash
python -m compileall candidates/202603302209_export-free-mtp/train_gpt.py
SMOKE_TEST=1 python candidates/202603302209_export-free-mtp/train_gpt.py
```

Outcome:

- `compileall`: passed for `candidates/202603302209_export-free-mtp/train_gpt.py`. I also successfully ran `python -m compileall train_gpt.py train_gpt_mlx.py data` against the unchanged baseline files.
- `SMOKE_TEST=1`: not runnable in this workflow runner as-is, because the environment does not currently have the repo's core Python ML dependencies installed (`torch`, `numpy`, `sentencepiece`). The candidate script includes a CPU smoke path, but the runtime stack itself is missing here.

## Main expected risks or tradeoffs

- MTP helps sample efficiency in the literature, but most published gains are on much larger models; the effect size on a 512-wide model may be modest.
- The tied-initialized MTP heads could bias the auxiliary task too strongly toward next-token behavior if `MTP_LOSS_WEIGHT` is too high.
- Training-time overhead still rises because the auxiliary heads do extra projection and loss work, even though the exported artifact stays unchanged.
- The best local stack already relies on legal TTT for part of its gain, so it may be hard to separate pure pre-TTT improvements from end-to-end post-TTT improvements without a controlled sweep.
