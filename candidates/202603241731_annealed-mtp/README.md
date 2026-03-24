# 202603241731_annealed-mtp

## Hypothesis

The strongest current 11-layer recipe already carries dormant multi-token prediction (MTP) support, but every shipped run keeps `MTP_NUM_HEADS=0`. My hypothesis is that a **small, warmdown-annealed MTP auxiliary loss** can improve early sample efficiency without hurting the late-stage compression-aware tuning that dominates this repo's best scores.

Concretely: keep the current SOTA backbone almost unchanged, turn on a single future-token head, and **linearly fade the auxiliary loss to zero with the same warmdown scale used for the learning rate**. Once that live weight becomes negligible, the candidate also **turns the MTP branch off entirely** so the last stretch of wallclock goes back to the base next-token objective instead of paying for near-zero-weight auxiliary compute.

## Why this looks promising for this repository

This repo's 10-minute track has converged on a very stable winning stack:

- 11 layers at width 512 with 3x MLPs
- XSA on the deepest layers
- partial RoPE and LN scaling
- EMA plus tight late SWA
- compression-aware int6/int8 export, culminating in GPTQ-lite clip search

The local evidence also shows that:

- late training details matter a lot more than naive architectural churn,
- quantization-aware endgame behavior matters,
- and the best branches already kept MTP plumbing around but never actually exercised it.

That makes MTP attractive here specifically because it is:

- already close to production in the best code path,
- free at export time because the extra heads are excluded from the artifact,
- and naturally compatible with the repo's heavy emphasis on sample efficiency under fixed wallclock.

The annealing twist is important because this repository's best results come from a compression-aware warmdown, not just from raw pre-quant next-token loss.

## Prior repository work that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Most relevant prior experiments:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - current best stack, adds GPTQ-lite clip search, EMA, warmdown 3500, and shared value embeddings.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - shows the 11-layer XSA/EMA lineage and confirms late-stage details matter.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md`
  - establishes EMA-over-SWA and the strong 11-layer seq2048 recipe.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/train_gpt.py`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train_gpt.py`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
  - all three retain MTP code paths but log `mtp_num_heads:0`, so the idea exists in code but was never part of a submitted run.

There were no prior iterations under `candidates/` when this candidate was created.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - motivates MTP as an auxiliary objective that improves sample efficiency by predicting several future tokens from a shared trunk.
- Ofir Press and Lior Wolf, **"Using the Output Embedding to Improve Language Models"** (`arXiv:1608.05859`)
  - not a new change in this candidate, but relevant context because this repo already relies on tied embeddings and the candidate preserves that compact-output design.

## What changed versus the chosen base implementation

Base file copied from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Candidate-specific changes:

1. **MTP defaults are enabled, but conservatively**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`
   - new knobs: `MTP_FINAL_FRAC=0.0`, `MTP_DISABLE_BELOW=0.02`

2. **The MTP contribution is now dynamic instead of fixed**
   - `GPT.forward(...)` now accepts an explicit `mtp_scale` argument plus an `mtp_active` flag.
   - training computes `mtp_scale = MTP_LOSS_WEIGHT * (MTP_FINAL_FRAC + (1 - MTP_FINAL_FRAC) * lr_scale)`.
   - with the default `MTP_FINAL_FRAC=0.0`, the auxiliary loss fades to zero across warmdown.

3. **Late-stage MTP compute is explicitly reclaimed**
   - new knob: `MTP_DISABLE_BELOW=0.02`.
   - once the live MTP weight falls below that threshold, the training loop passes `mtp_active=False`, so the auxiliary heads stop running instead of just being multiplied by a tiny scalar.

4. **Validation explicitly disables the auxiliary path**
   - validation calls the model with `mtp_scale=0`, so reported metrics stay aligned with the real export/eval objective.

5. **Warmup still compiles the MTP branch**
   - the warmup loop uses a nonzero MTP scale so the auxiliary path is exercised during graph warmup, then model/optimizer state is restored exactly as before.

6. **Logging now records the live auxiliary strength**
   - training logs include `mtp_scale:...` so it is easy to verify the annealing behavior.

7. **Export behavior is unchanged**
   - `mtp_heads` are still excluded from the final state dict before quantization/export, so the artifact budget is unchanged in spirit.

## How to run or evaluate it

From the repository root:

```bash
RUN_ID=annealed_mtp \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.15 \
MTP_FINAL_FRAC=0.0 \
MTP_DISABLE_BELOW=0.02 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202603241731_annealed-mtp/train_gpt.py
```

Two obvious follow-up sweeps if this starts promisingly:

- `MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.10`
- `MTP_FINAL_FRAC=0.05` to keep a tiny auxiliary signal alive deeper into warmdown

## Main expected risks and tradeoffs

- **Step-count risk:** even one extra future-token head adds training-time compute. The candidate now drops that branch once its live weight is negligible, but the early and mid-training overhead is still real.
- **Objective-mismatch risk:** MTP is only a training-time auxiliary objective. If its weight is too high, the trunk may drift away from what helps the exported next-token model.
- **Compile-path risk:** these branches already existed in code but were never used in submitted runs, and this candidate now introduces a two-phase `mtp_active` toggle under `torch.compile`, so there could still be edge cases under DDP or graph specialization.
- **Quantization uncertainty:** MTP should help the trunk, but it does not directly improve GPTQ-lite export the way better clipping or more fp16 keep-outs do.

## Validation

Ran from the repository root:

```bash
python -m compileall candidates/202603241731_annealed-mtp/train_gpt.py
```

Outcome:

- `PASS` - Python bytecode compilation succeeded.

Attempted a minimal CPU smoke test by importing the candidate module with a mocked `flash_attn_interface` fallback and running a tiny forward pass.

Outcome:

- `NOT FEASIBLE IN THIS CONTAINER` - the environment does not have `torch` installed for the interpreter used here (`ModuleNotFoundError: No module named 'torch'`).
- A real end-to-end smoke test would also require the CUDA training runtime plus local dataset shards.

