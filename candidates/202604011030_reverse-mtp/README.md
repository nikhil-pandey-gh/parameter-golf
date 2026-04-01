# Candidate: Reverse-Curriculum MTP on the 1.1233 Core Stack

## Hypothesis

The strongest next idea for this repository is to keep the current best non-TTT training stack intact and make it **more token-efficient** with **multi-token prediction (MTP)**, but in a way that is explicitly adapted for small models and next-token evaluation.

Concretely, this candidate starts from the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` record and enables **two auxiliary MTP heads during training only**, then applies a **reverse curriculum** that fades the farther-ahead targets out over training so the last part of optimization is pure next-token prediction again.

## Why this is promising for this repository

The repository history already shows that the main frontier is not “train longer” but rather:

- a strong 11-layer compression-aware trunk,
- sliding-window evaluation,
- careful quantization/export,
- and small additive improvements on top of that base.

That makes MTP attractive here because it can improve **training signal per forward pass** without forcing a broader architecture rewrite or changing the exported artifact.

This idea also fits the repo constraints unusually well:

- the candidate inherits the proven `1.1233` GPTQ-lite/EMA/XSA/Partial-RoPE stack,
- the extra MTP heads are **excluded from export**, so the 16 MB artifact budget still applies to the same inference model,
- and the challenge metric is still next-token BPB, which makes a **reverse curriculum** especially appealing because it lets the model end training on the exact evaluation objective.

## Prior records and candidate history that informed it

There were **no prior `candidates/` directories** in the repository when this candidate was created.

The main record history that informed this design:

- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/` showed that sliding-window scoring was a massive “free” gain and remains table stakes.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` established the modern 11-layer compression-aware core.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` showed that Partial RoPE + LN scale were meaningful zero- or near-zero-cost improvements.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the best non-TTT base and the direct code parent of this candidate.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is stronger overall, but a large part of its gain comes from legal score-first TTT and a more complex eval-time path, so it is a riskier base for a non-TTT candidate.

I also explicitly rejected parameter-sharing / recurrence ideas here because the non-record exploration in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` found **layer recurrence x2** to be a bad wallclock trade under this benchmark.

## External research that informed it

The candidate is mainly grounded in these primary sources:

- Fabian Gloeckle et al., [*Better & Faster Large Language Models via Multi-token Prediction*](https://arxiv.org/abs/2404.19737), 2024.  
  Main takeaway used here: MTP can improve sample efficiency and downstream quality with little extra training-time overhead relative to trunk compute.

- Ansar Aynetdinov and Alan Akbik, [*Pre-Training Curriculum for Multi-Token Prediction in Language Models*](https://arxiv.org/abs/2505.22757), 2025.  
  Main takeaway used here: **small language models need curriculum**, and the **reverse curriculum** variant can improve next-token quality the most even if it gives up self-speculative decoding benefits.

I also considered newer parameter-sharing research such as [*Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA*](https://arxiv.org/abs/2410.20672), but did not choose that direction because this repo’s own history already suggests recurrence can lose too many optimizer steps under the 10-minute wallclock cap.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Enabled training-only MTP by default**
   - `MTP_NUM_HEADS` now defaults to `2` instead of `0`.
   - The auxiliary MTP heads are still excluded from export, preserving the inference-time artifact definition.

2. **Added a reverse curriculum for MTP**
   - New knobs:
     - `MTP_CURRICULUM_END_FRAC` default `0.75`
     - `MTP_HEAD_DECAY` default `0.5`
   - The farther-ahead auxiliary heads fade out first, and all MTP supervision decays to zero by the configured fraction of training progress.

3. **Made the MTP schedule compile-safe**
   - The training loop computes per-head weights each step and passes them into the compiled model as a tensor, so the curriculum is data-driven rather than relying on Python-side toggles inside the compiled graph.

4. **Made the script runnable from the candidate directory**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root via `__file__`, so `cd candidates/202604011030_reverse-mtp && torchrun ... train_gpt.py` works without rewriting paths.

5. **Added a CPU-safe attention fallback and smoke-test path**
   - If FlashAttention 3 is unavailable, the script falls back to PyTorch SDPA.
   - `SMOKE_TEST=1 python train_gpt.py` runs a tiny CPU-only forward/backward + export-roundtrip sanity check that does not require the FineWeb shards.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202604011030_reverse-mtp
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.2 \
MTP_CURRICULUM_END_FRAC=0.75 \
MTP_HEAD_DECAY=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults already point at the repository tokenizer and dataset, so this should work directly when launched from this directory in the normal repo environment.

For a cheap startup sanity check:

```bash
cd candidates/202604011030_reverse-mtp
SMOKE_TEST=1 python train_gpt.py
```

## Main expected risks and tradeoffs

- **Training-time overhead:** even though the MTP heads are small relative to the trunk, they still add extra projection/loss work and may reduce step count slightly under the 600s wallclock cap.
- **Small-model sensitivity:** the curriculum paper argues small models need careful scheduling; poor settings for `MTP_CURRICULUM_END_FRAC` or `MTP_LOSS_WEIGHT` could negate the benefit.
- **Next-token alignment vs. auxiliary signal:** the reverse curriculum is designed to protect the final next-token objective, but if the decay is too aggressive it may leave performance on the table; if it is too weak it may hurt final BPB.
- **Inherited stack complexity:** this candidate intentionally keeps the best non-TTT stack, which is strong but already fairly optimized and therefore less forgiving of extra instability.

## Validation

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604011030_reverse-mtp/train_gpt.py`
  - **Result:** passed.

- `python3 -m venv /tmp/gh-aw/agent/pg-venv && . /tmp/gh-aw/agent/pg-venv/bin/activate && python -m pip install numpy sentencepiece torch`
  - **Result:** passed. This was needed because the runner did not have the candidate runtime dependencies preinstalled.

- `cd candidates/202604011030_reverse-mtp && . /tmp/gh-aw/agent/pg-venv/bin/activate && SMOKE_TEST=1 python train_gpt.py`
  - **Result:** passed with `smoke_test:ok train_loss:6.3769 eval_loss:4.9213 export_params:923164`.
