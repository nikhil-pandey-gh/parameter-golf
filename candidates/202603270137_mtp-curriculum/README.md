# Forward MTP Curriculum on the LeakyReLU^2 + Parallel Muon Stack

## Hypothesis

The strongest recent Parameter Golf stacks already have most of the obvious low-bit, eval, and architecture wins: XSA, partial RoPE, BigramHash, SmearGate, EMA/SWA, GPTQ-lite, and legal TTT. A more promising next step is to improve **sample efficiency during the 10-minute training window** without adding any artifact bytes at export time.

This candidate turns on the repo's dormant **multi-token prediction (MTP)** path, but does so with a **forward curriculum**: train as standard next-token prediction early, then gradually phase in auxiliary future-token heads once the trunk has already learned the basics. The key bet is that this gives a small model the optimization benefits of MTP without the usual early-training instability.

## Why this is promising for this repository

Three repo-specific facts make this a good fit:

1. Recent top records already carry `MTP_NUM_HEADS` / `MTP_LOSS_WEIGHT` code paths, but the published runs keep them disabled with `MTP_NUM_HEADS=0`.
2. The current stack already **excludes MTP heads from export**, so training with MTP can improve the trunk without paying artifact bytes at submission time.
3. Small-model MTP got a direct research update in 2025: **forward curricula help SLMs benefit from MTP**, whereas naive always-on MTP can underperform on small models.

That combination makes this a higher-upside and lower-infrastructure bet than a larger architecture rewrite.

## Prior records and repo patterns that influenced this candidate

There was **no existing `candidates/` directory** at review time, so the comparison set was the root baseline plus all prior `records/`.

The most relevant ancestors were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - current top stack
  - LeakyReLU(0.5)^2
  - parameter banking + Parallel Muon
  - legal score-first TTT
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest pre-TTT quantization-focused stack
  - GPTQ-lite clip search + EMA
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - partial RoPE and layerwise norm scaling
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - XSA on deepest layers, EMA replacing SWA
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - BigramHash + SmearGate + wider MLP as part of the stable winning recipe

The repo review also showed two useful guardrails:

- depth recurrence / looped layers were a negative result under the 10-minute wallclock budget
- quantization and eval improvements have mattered more than broad architectural rewrites

## External research that informed this candidate

- **Better & Faster Large Language Models via Multi-token Prediction**  
  Fabian Gloeckle et al., 2024  
  <https://arxiv.org/abs/2404.19737>  
  Core result: predicting multiple future tokens can improve sample efficiency and representation quality.

- **Pre-Training Curriculum for Multi-Token Prediction in Language Models**  
  Ansar Aynetdinov and Alan Akbik, 2025  
  <https://arxiv.org/abs/2505.22757>  
  Most relevant result here: **small language models benefit from a forward MTP curriculum**, where the objective gradually ramps from next-token prediction to multi-token prediction.

I also considered more quantization-heavy ideas from newer work, especially rotation-based PTQ, but chose MTP curriculum because it fits the current codebase more directly and can be implemented without broad new export infrastructure.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps that overall stack and changes only the parts needed for curriculum MTP:

1. **MTP is enabled by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Forward curriculum over wallclock progress**
   - new flags:
     - `MTP_CURRICULUM=1`
     - `MTP_START_FRAC=0.25`
     - `MTP_END_FRAC=0.70`
   - before 25% progress: auxiliary heads are off
   - from 25% to 70%: heads phase in gradually
   - after 70%: full 2-head MTP is active

3. **Compile-safe dynamic head weighting**
   - head activity is controlled by a runtime tensor buffer instead of Python-side branching
   - this avoids the same kind of `torch.compile` constant-folding problem that previously killed late-QAT in an earlier record

4. **MTP heads are actually optimized**
   - the base script's comment said MTP heads belonged in the Adam bucket, but the weights were not wired into the optimizer
   - this candidate adds the MTP head weights to the AdamW scalar/small-matrix optimizer group

5. **Bootstrap MTP heads from the tied embedding head**
   - when embeddings are tied, the auxiliary heads start from a copy of `tok_emb.weight` instead of zero
   - this should make the auxiliary loss useful earlier once the curriculum begins

6. **Local fallback for non-FlashAttention environments**
   - if `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA
   - this is only to make the script more smoke-testable off-H100; the intended challenge path is still the FlashAttention-3 CUDA setup

As in the parent record, **MTP heads are excluded from export**, so the candidate still pays only for the trunk model and code bytes.

## Why this differs from existing records and prior experiments

This is not just "turn MTP on":

- the repo already had dormant MTP code paths in several strong records
- published runs left them off
- small-model MTP research says the missing piece is often the **curriculum**

So the novelty here is the combination of:

- a current SOTA-style compact stack,
- export-free auxiliary heads,
- actual optimizer wiring for those heads,
- and a small-model-friendly forward MTP schedule.

## How to run / evaluate

From the repository root:

```bash
cd candidates/202603270137_mtp-curriculum
SEED=1337 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_CURRICULUM=1 \
MTP_START_FRAC=0.25 \
MTP_END_FRAC=0.70 \
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

If you want to isolate the training-time effect before legal TTT, run the same command with `TTT_ENABLED=0` and compare the pre-TTT validation number to the current record's pre-TTT baseline.

## Validation

Validation attempted in this environment:

```bash
python -m compileall candidates/202603270137_mtp-curriculum/train_gpt.py
```

Outcome: **passed**.

Attempted CPU smoke check:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch
...
PY
```

Outcome: **not feasible in this runner**, because the available `python` / `python3` interpreters do not have `torch` installed, so an import-time forward/backward smoke test could not be executed locally here.

## Main expected risks and tradeoffs

- **Training throughput risk**: even excluded-from-export MTP heads still cost training FLOPs. The curriculum needs to earn back any step-count loss through better sample efficiency.
- **Schedule sensitivity**: the best `MTP_START_FRAC`, `MTP_END_FRAC`, number of heads, and loss weight may differ from the defaults here.
- **TTT interaction ambiguity**: legal TTT can improve the final score enough to mask whether MTP helped the trunk itself, so pre-TTT comparisons matter.
- **Bootstrap choice risk**: initializing MTP heads from the tied embedding matrix is plausible but not yet ablated in this repo.

## Suggested next experiments

1. Sweep `MTP_NUM_HEADS` in `{1, 2, 3}` with the same curriculum window.
2. Sweep `MTP_START_FRAC` / `MTP_END_FRAC`, especially earlier starts like `0.15 -> 0.60`.
3. Compare bootstrapped MTP heads vs zero-init heads.
4. Measure both **pre-TTT** and **post-TTT** deltas, not just the final number.
5. If MTP helps the trunk cleanly, try stacking it with the `2026-03-22` GPTQ-lite export path as a lower-complexity branch.
