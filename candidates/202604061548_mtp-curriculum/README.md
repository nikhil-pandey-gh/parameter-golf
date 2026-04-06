# MTP Curriculum on the 11L EMA/XSA/GPTQ-lite Backbone

## Hypothesis

Small language models in this repo are already strong on quantization/export, sliding-window evaluation, and the 11-layer XSA/EMA/partial-RoPE backbone. The least-explored cheap lever is **train-only multi-token prediction (MTP)**: auxiliary future-token heads can improve sample efficiency during the 10-minute training budget, then be dropped at export for effectively zero artifact cost. Because recent work reports that **small models struggle with naive MTP**, this candidate uses a **forward curriculum** instead of turning the full MTP objective on at step 0.

## Why this is promising here

- The current strongest non-TTT training scaffold is the 11-layer EMA/XSA/GPTQ-lite stack from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, and that script already contains dormant MTP hooks with export-time exclusion of the auxiliary heads.
- Repo history suggests fixed-wallclock recurrence/depth-reuse is higher risk here, while MTP is still unsubmitted in any record even though the plumbing exists.
- MTP heads do **not** count toward the final artifact in this candidate because they are removed before export, so the idea attacks training efficiency without spending the 16 MB budget.

## Prior repository influences

- **Chosen base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- **Lineage behind the base:** the 11-layer XSA/EMA/partial-RoPE family from `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` and `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Current leaderboard context:** `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` shows the frontier is now mostly low-cost increments on top of a strong backbone, plus evaluation-time tricks
- **Prior candidates:** none; `candidates/` did not exist before this run

## External research that informed this candidate

1. **Better & Faster Large Language Models via Multi-token Prediction** (`arXiv:2404.19737`) argues that MTP improves sample efficiency and downstream capability while keeping inference/export overhead optional when treated as auxiliary heads.
2. **Pre-Training Curriculum for Multi-Token Prediction in Language Models** (`arXiv:2505.22757`) specifically finds that **small language models** benefit from a **forward curriculum** that ramps from NTP toward MTP instead of training with the full objective immediately.
3. I also considered recent recurrent-depth / shared-block work such as **Intra-Layer Recurrence in Transformers for Language Modeling** (`arXiv:2505.01855`), but repo history already contains negative evidence on fixed-wallclock recurrence, so this candidate prioritizes the lower-risk MTP path.

## What changed vs. the base

1. **Enabled train-only MTP by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`.
2. **Added a forward MTP curriculum**: the auxiliary objective ramps in from `MTP_START_FRACTION=0.25` to `MTP_FULL_FRACTION=0.70`, with per-head **loss weights** turning on progressively instead of applying the full MTP objective from step 0.
3. **Changed candidate defaults to the record-like regime** used by the strong 11-layer stack: `ITERATIONS=9000` and `TRAIN_LOG_EVERY=200`.
4. **Made default dataset/tokenizer paths relative to the candidate file location**, so `train_gpt.py` can be run from inside this candidate directory without extra path env vars.
5. **Added a FlashAttention fallback to PyTorch SDPA** so the module can still be imported and non-Flash smoke checks can use SDPA when `flash_attn_interface` is unavailable.
6. **Disabled the broken late-QAT toggle by default** with `LATE_QAT_THRESHOLD=0.0`. The script still supports always-on QAT through `QAT_ENABLED=1`, but it avoids the known compile-time “looks enabled, does nothing” hazard from late toggling.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202604061548_mtp-curriculum
RUN_ID=mtp_curriculum \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
MTP_NUM_HEADS=2
MTP_LOSS_WEIGHT=0.15
MTP_CURRICULUM=1
MTP_START_FRACTION=0.25
MTP_FULL_FRACTION=0.70
QAT_ENABLED=1            # optional: always-on fake quant
LATE_QAT_THRESHOLD=0.0   # default; keep late-toggle disabled
```

The script keeps the base stack's final export/eval path: EMA averaging, GPTQ-lite-style mixed int6 export, standard roundtrip eval, and sliding-window eval.

## Main risks / tradeoffs

- **Tiny-model sensitivity:** the ACL 2025 curriculum paper exists precisely because naive MTP can hurt small models, so the ramp schedule may still need tuning.
- **Training-time overhead:** the auxiliary heads are export-free, but they still spend compute during training. This implementation ramps supervision, not head matmul count, so early-step compute is still close to full MTP.
- **No evaluation-time add-on:** this candidate deliberately targets training efficiency, not legal TTT. It may improve the training-only stack without immediately challenging the full eval-time SOTA.
- **QAT interaction remains open:** this version defaults to PTQ/GPTQ-lite behavior and keeps always-on QAT optional instead of mixing in another unstable axis by default.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604061548_mtp-curriculum/train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path("candidates/202604061548_mtp-curriculum/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
```

Outcomes:

- `python -m compileall ...` **passed**
- The attempted CPU smoke import **could not run in this workflow environment** because `torch` is not installed here (`ModuleNotFoundError: No module named 'torch'`), so no runtime start-up check was feasible without installing heavyweight dependencies
