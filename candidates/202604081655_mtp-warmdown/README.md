# Candidate: Training-Only MTP Warmdown

## Hypothesis

A small amount of **training-only multi-token prediction (MTP)** should improve sample efficiency for the strong 11-layer compressed stack from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, while a **late warmdown of the auxiliary loss** should let the model finish aligned to the leaderboard's next-token objective.

The key bet is that this challenge is wallclock-limited and artifact-limited, so an auxiliary objective that only exists during training can be attractive if its extra heads are excluded from the exported model.

## Why this looks promising in this repository

After reviewing the baseline and prior records, three trends stood out:

1. The strongest gains now come from **evaluation, compression, and small cumulative training tricks**, not from simply training longer.
2. **Depth recurrence / looped reuse** looked weak in short-wallclock runs, so a low-churn training objective is safer than another architecture rewrite.
3. Recent strong scripts already contain **export-safe MTP plumbing**, but the logged record runs still kept `MTP_NUM_HEADS=0`, so this is still largely unexplored in the published run history.

There were **no prior `candidates/` directories** in this checkout, so this is the first candidate iteration in that namespace.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Strong pre-TTT stack with EMA, GPTQ-lite int6 export, XSA4, partial RoPE, LN scale, BigramHash, and value embeddings.
- **Architecture/training stack lineage:**  
  - `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Current leaderboard context / what not to repeat:**  
  - `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Evidence that recurrence is a poor fit here:**  
  - `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
- **Dormant MTP evidence in recent stacks:**  
  - logs under the 2026-03-21, 2026-03-22, and 2026-03-23 record folders still show `mtp_num_heads:0`.

## External research that informed it

- **Better & Faster Large Language Models via Multi-token Prediction** — arXiv:2404.19737  
  Direct evidence that MTP can improve **sample efficiency** by predicting multiple future tokens with a shared trunk plus extra output heads.
- **DeepSeek-V3 Technical Report** — arXiv:2412.19437  
  Practical evidence that a production LLM can use an **MTP training objective** and discard the extra MTP module at inference.
- **Pre-Training Curriculum for Multi-Token Prediction in Language Models** — arXiv:2505.22757  
  Especially relevant here because it explicitly studies **smaller language models** and finds that changing MTP emphasis during training can improve final next-token quality.

## What changed versus the chosen base

This candidate starts from the `2026-03-22` stack and makes four surgical changes:

1. **Turn MTP on by default**
   - `MTP_NUM_HEADS` now defaults to `1` instead of `0`.
   - `MTP_LOSS_WEIGHT` now defaults to `0.15`.
2. **Warm MTP down with the existing wallclock-aware LR schedule**
   - New `MTP_WARMDOWN_THRESHOLD` env var (default `0.2`).
   - New `MTP_DISABLE_FRACTION` env var (default `0.05`).
   - While LR scale is above that threshold, MTP runs at full weight.
   - Below the threshold, the MTP weight decays linearly toward zero.
   - Once the weight falls below 5% of its starting value, the training loop skips the MTP branch entirely to claw back late-stage compute.
3. **Make the script runnable from the candidate directory**
   - Default dataset/tokenizer paths resolve relative to the repository root derived from `__file__`.
4. **Add a non-FlashAttention fallback**
   - If `flash_attn_interface` is unavailable **or** the GPU is not Hopper-class, attention falls back to PyTorch SDPA so the module remains usable off the FA3 path.

The existing export behavior from the base stack is kept: **`mtp_heads` are excluded from the serialized artifact**, so the auxiliary objective is training-only.

## How to run

From this candidate directory:

```bash
RUN_ID=mtp_warmdown_seed1337 \
SEED=1337 \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.15 \
MTP_WARMDOWN_THRESHOLD=0.2 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful comparisons:

- **Disable the idea entirely:** `MTP_NUM_HEADS=0`
- **Test stronger auxiliary pressure:** raise `MTP_LOSS_WEIGHT`
- **Fade MTP earlier or later:** tune `MTP_WARMDOWN_THRESHOLD`
- **Cut off MTP compute earlier or later:** tune `MTP_DISABLE_FRACTION`

## How to evaluate

The script keeps the base stack's export + validation flow:

1. train under the 600s cap,
2. apply EMA,
3. export without `mtp_heads`,
4. run roundtrip int6 validation,
5. run sliding-window evaluation.

The final leaderboard-comparable metric is printed in the `final_int6_sliding_window_exact` / `final_int8_zlib_roundtrip_exact` lines, matching the recent record conventions.

## Main risks and tradeoffs

- **Small-model risk:** the literature is positive on MTP, but the biggest gains are often reported at larger scales than this challenge.
- **Step-time risk:** even one auxiliary head adds extra vocab projections during training.
- **Objective-mismatch risk:** if MTP remains too strong late in training, it can hurt the final next-token objective; that is why this candidate fades it out during warmdown.
- **Tuning risk:** the best point may be `k=1` here, but `k=2` could still win if the throughput hit is smaller than expected.

## Validation

Commands run from the repository root:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604081655_mtp-warmdown/train_gpt.py
```

Outcome:

- **Passed** for the repository's existing Python entry points and the new candidate script.

Attempted smoke check:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202604081655_mtp-warmdown/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_mtp', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcome:

- **Not feasible on this runner** because the environment does not currently have PyTorch importable (`ModuleNotFoundError: No module named 'torch'`), so only compile-time validation was possible here.
