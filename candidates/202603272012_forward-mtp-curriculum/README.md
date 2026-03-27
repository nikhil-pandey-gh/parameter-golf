# Forward-Curriculum MTP on the LeakyReLU^2 + Legal TTT Stack

## Hypothesis

A small 11-layer Parameter Golf model can benefit from **multi-token prediction (MTP)**, but only if the auxiliary objective is introduced gradually.

Recent MTP work argues that predicting multiple future tokens can improve sample efficiency, while newer follow-up work shows that **small language models struggle with always-on MTP** and instead benefit from a **forward curriculum** that ramps from plain next-token prediction into multi-token prediction over training.

This candidate tests that idea on top of the strongest current training stack in this repository: keep the March 23 LeakyReLU^2 + Parallel Muon + legal TTT architecture intact, but turn the dormant MTP path into a real training objective with a small-model-friendly curriculum.

## Why this looks promising for this repository

The records show a few strong patterns:

- Better training efficiency and cheap auxiliary inductive bias keep paying off.
- Post-training and evaluation tricks are already highly optimized, so the remaining room is increasingly in **better pre-TTT representations**.
- Multiple record scripts already carry dormant MTP code paths, but the published runs keep `MTP_NUM_HEADS=0`, so the idea has been scaffolded without being seriously exercised.
- The top record still spends the entire 10-minute budget on training, so a **training-only auxiliary loss** that does not affect exported weights or inference structure is especially attractive.

The candidate therefore targets an underexplored gap that is compatible with the artifact budget and requires only incremental code changes.

## Prior repository work that influenced this candidate

### Main base implementation

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This is the current best record and provides the actual architecture copied here: 11 layers, LeakyReLU(0.5)^2 MLPs, bigram hash, XSA on the last 4 layers, partial RoPE, EMA, legal score-first TTT, and Parallel Muon parameter banking.

### Supporting record lineage

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`

These runs established that the heavy 11-layer stack, EMA, XSA, partial RoPE, and quantization-aware warmdown choices were robust wins.

### Why I did **not** choose recurrence/depth reuse instead

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md`

Those files explicitly note that plain depth recurrence was promising in theory but poor in the 10-minute regime, so I treated recurrence as a weaker next bet than a training-only auxiliary objective.

### Prior candidates

There was no `candidates/` directory in the repository before this submission.

## External research that informed the idea

### 1. Better & Faster Large Language Models via Multi-token Prediction

- Fabian Gloeckle et al., 2024
- arXiv: `2404.19737`
- https://arxiv.org/abs/2404.19737

Key takeaway: predicting multiple future tokens with separate auxiliary heads can improve sample efficiency and downstream quality while sharing the same main trunk.

### 2. Pre-Training Curriculum for Multi-Token Prediction in Language Models

- Ansar Aynetdinov and Alan Akbik, 2025
- arXiv: `2505.22757`
- https://arxiv.org/abs/2505.22757

Key takeaway: **small language models** do not reliably benefit from raw MTP from step 0, but a **forward curriculum** that gradually increases MTP complexity improves next-token performance and output quality.

## What changed versus the chosen base implementation

Starting from the March 23 record script, this candidate makes four tightly scoped changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` now defaults to `2` instead of `0`.
   - The existing MTP auxiliary heads are now intended to be active rather than dormant.

2. **Add a forward curriculum for MTP**
   - New knobs:
     - `MTP_CURRICULUM=1`
     - `MTP_START_FRAC=0.25`
     - `MTP_RAMP_FRAC=0.50`
   - The first auxiliary head activates first, and later heads ramp in only after the earlier ones have warmed up.
   - Inactive heads are skipped entirely, so early training also avoids paying their projection/loss compute.
   - This follows the small-model guidance from the 2025 curriculum paper.

3. **Weight MTP losses by per-head curriculum weights**
   - The auxiliary loss is no longer all-or-nothing.
   - The model stores non-persistent per-head curriculum weights and scales each auxiliary prediction head accordingly during training.

4. **Actually optimize the MTP heads in the Parallel Muon stack**
   - In the March 23 script, the MTP path existed but `MTP_NUM_HEADS=0` kept it dormant.
   - This candidate wires the MTP head weights into the AdamW side of the optimizer split so the heads train correctly when enabled.

Importantly, the exported artifact still excludes the MTP heads, exactly like the earlier dormant MTP code path intended. That means the candidate keeps the inference/export footprint unchanged aside from training-time code bytes.

## How to run or evaluate it

### Recommended training/eval command

```bash
RUN_ID=forward_mtp_curriculum \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 \
MTP_CURRICULUM=1 MTP_START_FRAC=0.25 MTP_RAMP_FRAC=0.50 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

EMA is always applied in this script, so there is no separate `EMA_ENABLED` toggle.

### Minimal syntax validation

```bash
python -m compileall candidates/202603272012_forward-mtp-curriculum/train_gpt.py
```

## Validation run for this candidate

### Commands run

```bash
python -m compileall candidates/202603272012_forward-mtp-curriculum/train_gpt.py
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603272012_forward-mtp-curriculum/train_gpt.py
python - <<'PY'
import importlib.util
mods = ['torch', 'sentencepiece', 'flash_attn_interface']
for name in mods:
    print(f"{name}:{bool(importlib.util.find_spec(name))}")
PY
```

### Outcomes

- `compileall` passed for the candidate script.
- The broader repository syntax sweep also passed.
- A real runtime smoke test was **not feasible in this environment**:
  - `torch` is not installed,
  - `sentencepiece` is not installed,
  - `flash_attn_interface` is not installed,
  - and this script explicitly requires CUDA at runtime.

Because of those hard environment limits, I did not claim a fake CPU smoke run.

## Main expected risks and tradeoffs

1. **Extra training compute**
   - Even though the MTP heads are small and excluded from export, they still add training-time matmuls and losses.
   - If the extra per-step cost is too high, the model may lose enough steps over 600 seconds to erase the sample-efficiency gain.

2. **Small-model sensitivity**
   - The 2025 paper exists precisely because small models can regress under naïve MTP.
   - The curriculum should help, but the exact start/ramp schedule may matter a lot.

3. **Optimizer mismatch for auxiliary heads**
   - I intentionally kept the implementation minimal by routing the auxiliary MTP heads through the AdamW side of the existing optimizer split instead of inventing a separate optimizer regime.
   - That is simple and robust, but may not be the optimal LR/weight-decay setup.

4. **TTT interaction is uncertain**
   - This candidate is primarily trying to improve the pretrained representation before legal TTT.
   - It is possible MTP helps the pre-TTT model but changes how much additional gain TTT can still extract.

## Suggested next experiments if this is promising

- Sweep `MTP_NUM_HEADS` in `{1, 2, 3}`.
- Sweep `MTP_START_FRAC` and `MTP_RAMP_FRAC` to shift the curriculum later or earlier.
- Try the same curriculum on the March 22 pre-TTT stack to isolate whether the gain comes from better pretraining or from interaction with the March 23 stack.
- Give MTP heads their own LR if the current shared AdamW group looks undertrained.
