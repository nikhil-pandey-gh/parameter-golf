# Forward-Curriculum MTP on the 11L EMA + GPTQ-lite Stack

## Hypothesis

The repo's strongest recent gains are already concentrated in quantization, evaluation, and small architectural tweaks. A stronger next step is to improve **sample efficiency during training** without increasing final artifact size.

This candidate enables **multi-token prediction (MTP)** as a training-only auxiliary objective on top of the strongest pre-TTT stack in `records/`, then applies a **forward curriculum** so the model starts with standard next-token prediction and only gradually ramps into longer-horizon prediction. The goal is to make MTP work for this repository's tiny-model regime without paying permanent artifact bytes.

## Why this is promising for this repository

Three repo trends pushed this direction:

- The strongest pre-TTT base is already `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, which has a stable 11L/XSA/Partial-RoPE/LN-scale/EMA/GPTQ-lite recipe and even contains dormant MTP plumbing that was never actually turned on.
- The current top record (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) wins mostly by stacking TTT and systems work on top of an already-strong pre-TTT model, which suggests the less-crowded space is **training objective design**, not another tiny quantization or eval tweak.
- The late-record review showed the current stack is crowded around XSA, partial RoPE, EMA/SWA, warmdown tuning, and GPTQ-lite, while the external research review ranked **forward-curriculum MTP** as the best non-duplicate next bet.

Crucially, this challenge only counts the final exported model and code. Auxiliary MTP heads are therefore attractive because they can improve optimization while being **dropped before export**.

## Which records influenced this candidate

This implementation is based most directly on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - chosen as the base because it is the strongest clean pre-TTT stack and already includes the needed MTP export exclusions.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - for the evidence that Partial RoPE + layerwise LN scaling are robust gains.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - for the durable XSA + EMA base recipe.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
  - for the broader lesson that bytes are precious and training-time-only tricks are unusually valuable in this challenge.

Relevant negative evidence also mattered:

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
  - found that naive layer recurrence was strongly negative in a fixed wallclock budget, which made me prefer a **training-objective** change over a depth-reuse change.

There were **no prior `candidates/` directories** in this checkout.

## External research that informed it

The main sources were:

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737)
  - argues that predicting multiple future tokens can improve sample efficiency and downstream quality while keeping the main trunk shared.
- Ansar Aynetdinov and Alan Akbik, **"Pre-Training Curriculum for Multi-Token Prediction in Language Models"** (arXiv:2505.22757)
  - especially relevant here because it focuses on **small language models** and reports that a **forward curriculum** helps SLMs benefit from MTP more reliably.

Those papers suggest the exact twist this repo needs: **use MTP, but do not switch it on abruptly for a small model**.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes six focused changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` now defaults to `2` instead of `0`.
   - `MTP_LOSS_WEIGHT` now defaults to `0.15`.

2. **Add a forward curriculum for MTP**
   - New knobs: `MTP_CURRICULUM_START` and `MTP_CURRICULUM_END`.
   - A new `mtp_head_weights` buffer ramps heads in sequentially over training progress instead of turning all auxiliary heads on at once.
   - Default schedule: start at 25% of training progress, finish by 75%.

3. **Keep MTP training-only and artifact-free**
   - The auxiliary heads still participate in optimization.
   - They are still explicitly excluded from the exported state dict before quantization, so the final artifact path remains aligned with the challenge rules.

4. **Make the script runnable from the candidate directory**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root instead of assuming the current working directory is the repo root.

5. **Fix Late-QAT wiring inherited from the record stack**
   - The candidate switches QAT enablement from a mutable class attribute to per-module flags, so the late-QAT transition is less likely to be constant-folded away under `torch.compile`.

6. **Add optional CPU-safe smoke support**
   - FlashAttention import is now optional.
   - Non-CUDA fallback uses `torch.nn.functional.scaled_dot_product_attention`.
   - `ALLOW_CPU=1` plus `ENABLE_COMPILE=0` allows a tiny local smoke run when dataset shards and tokenizer files are available.
   - `SMOKE_VAL_TOKENS` can cap validation tokens for that smoke path.

The core model stack otherwise stays intentionally close to the strong 2026-03-22 recipe: 11 layers, 512 dim, XSA on the last 4 layers, Partial RoPE, LN scaling, VE128, EMA, tight SWA, GPTQ-lite int6 export, and the same optimizer family.

## How to run or evaluate it

From this candidate directory:

```bash
RUN_ID=fcmtp_seed1337 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides for ablations:

```bash
# Disable the curriculum and turn on both auxiliary heads immediately
MTP_CURRICULUM_START=0.0 MTP_CURRICULUM_END=0.0 \
RUN_ID=fcmtp_no_curriculum torchrun --standalone --nproc_per_node=8 train_gpt.py

# Reduce to one auxiliary head
MTP_NUM_HEADS=1 RUN_ID=fcmtp_h1 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Disable MTP entirely to recover the chosen base recipe more closely
MTP_NUM_HEADS=0 RUN_ID=fcmtp_off torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional tiny CPU smoke path for a local machine that already has the challenge data and tokenizer available:

```bash
ALLOW_CPU=1 ENABLE_COMPILE=0 \
ITERATIONS=1 VAL_LOSS_EVERY=1 EVAL_STRIDE=0 \
TRAIN_SEQ_LEN=8 EVAL_SEQ_LEN=8 TRAIN_BATCH_TOKENS=64 VAL_BATCH_SIZE=64 \
SMOKE_VAL_TOKENS=64 \
python train_gpt.py
```

## Main expected risks or tradeoffs

- **Extra training compute:** even training-only heads cost some wallclock, so MTP must improve sample efficiency enough to pay for itself.
- **Small-model fragility:** the 2025 curriculum paper is encouraging precisely because small models can struggle with abrupt MTP; if the schedule is still too aggressive, the gain may wash out.
- **Compile interaction risk:** the curriculum uses a tensor buffer so it should be safer than the dead class-attribute Late-QAT pattern noted in prior records, but it still needs real GPU validation.
- **No artifact win by itself:** unlike a better quantizer, MTP does not directly reduce bytes; it must win through lower NTP loss at export time.

## Validation

Validation run in this checkout:

```bash
python -m compileall candidates/202603311343_forward-curriculum-mtp/train_gpt.py
```

Outcome:

- **Passed**: Python bytecode compilation succeeded.

A CPU smoke run was **not executed in this environment** because the repository checkout does not currently include:

- `data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
- `data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- `data/tokenizers/fineweb_1024_bpe.model`

So syntax validation was possible here, but data-dependent runtime validation was not.
