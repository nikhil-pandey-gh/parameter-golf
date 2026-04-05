# 202604050140_mtp-curriculum

## Hypothesis

A tiny 11-layer model in this repo is more likely to benefit from **late-onset multi-token prediction (MTP)** than from always-on MTP. The trunk can spend the first part of the 10-minute budget learning a strong next-token model, then add a small amount of future-token supervision once the shared representation is already useful.

## Why this is promising here

The local record trend is already concentrated around one strong recipe: 11 layers, 3x MLP, late-layer XSA, partial RoPE, EMA/tight averaging, bigram/value-embedding add-ons, aggressive post-training quantization, and in the latest record, LeakyReLU^2 plus legal TTT. What is *not* used in any winning record is the dormant MTP path that appears in several recent `train_gpt.py` variants but is always left at `MTP_NUM_HEADS=0`.

That makes MTP unusually attractive for this repository:

- it is **export-free** here because the auxiliary heads are already excluded from the final artifact,
- it can improve **sample efficiency** instead of artifact compression,
- it adapts naturally to the existing shared-trunk design,
- and it only needs a surgical extension to the current best fast training stack.

## Repository review: winning patterns and dead ends

### Winning patterns from `records/`

- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`: 11L + XSA + EMA established the modern strong base.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: partial RoPE and layer scaling improved the same stack.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: stronger post-training quantization and warmdown tuning pushed the best pre-TTT score.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: LeakyReLU^2 and legal score-first TTT currently define the best overall recipe, while parameter banking keeps training fast enough to make another training-side idea plausible.

### Dead ends or weaker directions

- Early low-level tuning alone (`LowerLR`, `TrainingOptSeq4096`, `LongContextSeq2048`) helped, but not enough to stay competitive once the 11-layer/XSA/EMA stack appeared.
- SwiGLU looked good in the single-GPU non-record exploration but was explicitly noted as too slow on the main multi-GPU track.
- The repo has carried MTP scaffolding since the March 20 record family, but every tracked run still used `MTP_NUM_HEADS=0`, so the idea is present but effectively untested here.

## External research that informed this candidate

1. **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737, 2024).**  
   This paper argues that adding independent future-token heads on top of a shared trunk improves sample efficiency and downstream quality, with the auxiliary heads treated as a training task rather than a permanent artifact requirement.

2. **Ansar Aynetdinov and Alan Akbik, "Pre-Training Curriculum for Multi-Token Prediction in Language Models" (arXiv:2505.22757, 2025).**  
   This is the key reason not to use raw always-on MTP here: the paper specifically reports that **smaller language models struggle with the plain MTP objective**, and that a **forward curriculum** helps them benefit from MTP without paying the same quality penalty.

## Base implementation

This candidate is derived from:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

That base was chosen because it is the strongest current stack in the repo and already has the fast parameter-banking optimizer path needed to offset some MTP overhead.

## What changed versus the base

1. **MTP is enabled by default** with `MTP_NUM_HEADS=2`.
2. **Forward curriculum switch** via `MTP_ENABLE_FRAC` (default `0.2`): the model trains as plain next-token prediction for the first 20% of progress, then enables MTP for the rest.
3. **Head bootstrap at switch time**: when MTP turns on, the auxiliary heads are initialized from the current main output weights instead of learning from zero late in training.
4. **Optimizer wiring fix for the fast parameter-banking stack**: the latest fast script had MTP heads in the forward pass and export exclusion path, but not in the AdamW parameter set. This candidate wires them into optimization and replicated gradient reduction so enabling MTP actually trains them.

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For the full current-best evaluation stack, keep the candidate defaults and enable legal TTT:

```bash
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 \
TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Plain next-token baseline inside this candidate
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Always-on MTP, to compare against the curriculum switch
MTP_ENABLE_FRAC=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Risks and tradeoffs

- MTP still adds training-time compute, so if the extra supervision is too weak the reduced step count could erase the gain.
- The optimal switch point is unknown; `0.2` is a research-grounded default, not a tuned optimum for this exact benchmark.
- Bootstrapping the auxiliary heads from the main head is a pragmatic adaptation for this short training window, but it is not yet locally ablated.
- The biggest remaining uncertainty is whether this helps more on the **pre-TTT** model, the **post-TTT** score, or both.

## Validation

Executed:

```bash
python -m compileall \
  /home/runner/work/parameter-golf/parameter-golf/train_gpt.py \
  /home/runner/work/parameter-golf/parameter-golf/train_gpt_mlx.py \
  /home/runner/work/parameter-golf/parameter-golf/data \
  /home/runner/work/parameter-golf/parameter-golf/candidates/202604050140_mtp-curriculum/train_gpt.py
```

Outcome:

- success; the root training scripts, existing `data/` Python files, and the candidate `train_gpt.py` all compiled cleanly

Additional smoke-check attempt:

- a real import/startup smoke test was **not feasible in this runner** because the required runtime packages for this training stack (`torch`, `sentencepiece`, and `flash_attn_interface`) were not installed here
