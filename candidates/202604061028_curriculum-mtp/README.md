# Candidate: forward-curriculum MTP on the LeakyReLU² + Legal TTT stack

## Hypothesis

The current frontier already spends almost all of the 16MB budget on a strong 11-layer export path, so the next gain is more likely to come from a **training-only objective** than from another artifact-heavy architectural add-on. This candidate turns on a **single multi-token prediction (MTP) head** and ramps its weight in gradually, so the trunk gets extra future-token supervision only after the base next-token objective has stabilized.

Because the extra MTP head is already excluded from export in this codebase, the idea aims for **better sample efficiency at essentially zero artifact-cost increase**.

## Why this is promising here

Repository review showed a clear pattern:

- early wins came from context length, scheduling, and evaluation strategy,
- mid-cycle wins came from mixed quantization, bigger MLPs, BigramHash, SmearGate, and tighter export heuristics,
- the latest record line is already very saturated on artifact size, with recent gains coming from small training/eval refinements such as EMA, GPTQ-lite, LeakyReLU², and legal TTT.

That makes an export-free auxiliary loss attractive. The top record lineage also already contains dormant MTP support in `train_gpt.py`, but all reviewed record stacks kept it effectively off with `MTP_NUM_HEADS=0`. This candidate is a clean attempt to cash in on that unused path.

## Influential prior work in this repo

### Chosen base

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

I used this as the base because it is the strongest current stack and already combines:

- LeakyReLU(0.5)^2,
- parameter banking + Parallel Muon,
- partial RoPE + LN scale,
- shared value embeddings,
- GPTQ-lite-ready mixed export,
- legal score-first TTT.

### Other repo findings that shaped the choice

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: showed that low-cost post-training and averaging tweaks still matter on top of the 11-layer stack.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: showed that compile-time flag behavior can silently kill intended dynamic training features, so this candidate uses a tensor buffer for the MTP schedule rather than a Python-side toggle.
- `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` and related quantization-heavy runs: showed that the artifact budget is already tightly packed, so ideas that do **not** survive into the exported model are especially valuable.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`: made me avoid recurrence/weight-sharing as the main bet here, since its layer-recurrence sweep was clearly negative under fixed wallclock.

There were **no prior `candidates/` directories** in the repository at review time.

## External research that informed the idea

- **Gloeckle et al., 2024 — “Better & Faster Large Language Models via Multi-token Prediction”** (`arXiv:2404.19737`): motivates MTP as a sample-efficiency-improving auxiliary objective on top of a shared trunk.
- **Aynetdinov & Akbik, 2025 — “Pre-Training Curriculum for Multi-Token Prediction in Language Models”** (`arXiv:2505.22757`): specifically relevant because it argues that **small language models struggle with static MTP**, and that a **forward curriculum** helps them benefit from MTP more reliably.

I also reviewed parameter-sharing literature (for example ALBERT and later transformer-sharing work), but repo evidence pushed me away from that direction for this candidate.

## What changed versus the base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes four focused changes:

1. **Turns on one MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.12`

2. **Adds a forward curriculum for the MTP weight**
   - `MTP_WARMUP_FRACTION=0.20`
   - `MTP_FULL_FRACTION=0.55`
   - the MTP weight ramps smoothly from `0.0` to the target weight over the early/mid part of training.

3. **Makes the curriculum compile-safe**
   - the live MTP weight is stored in a tensor buffer (`mtp_aux_weight`) and updated from the training loop, instead of relying on a Python flag that may be constant-folded by `torch.compile`.

4. **Makes candidate-local execution cleaner**
   - dataset and tokenizer defaults resolve from the repository root, so the script can be launched directly from this candidate directory.

The export path is intentionally unchanged: the extra MTP head remains a **training-only** component and is still excluded from the final artifact.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202604061028_curriculum-mtp
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
cd candidates/202604061028_curriculum-mtp
SEED=1337 MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=1337 MTP_LOSS_WEIGHT=0.08 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=1337 MTP_WARMUP_FRACTION=0.30 MTP_FULL_FRACTION=0.70 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the top-stack defaults (including legal TTT at evaluation time). For faster training-only ablations, set `TTT_ENABLED=0`.

## Validation

Commands run in this workspace:

```bash
python -m compileall candidates/202604061028_curriculum-mtp/train_gpt.py
python - <<'PY'
import importlib.util
mods = ['flash_attn_interface', 'zstandard', 'sentencepiece', 'torch']
for m in mods:
    print(f"{m}: {'yes' if importlib.util.find_spec(m) else 'no'}")
PY
```

Observed outcomes:

- `python -m compileall ...` **passed**.
- A real smoke run was **not feasible in this container**:
  - `torch`, `sentencepiece`, and `flash_attn_interface` are not installed here,
  - this script is written for the repository's CUDA/FlashAttention evaluation environment rather than a CPU-only fallback path.

## Main risks / tradeoffs

- **Step-time tradeoff**: even one extra MTP head adds forward/backward work. If the sample-efficiency gain is too small, fewer steps in 600s could erase the benefit.
- **Small-model sensitivity**: the curriculum is the point of the candidate, but a 512-dim model may still over-regularize if the target MTP weight is too high.
- **Attribution blur with TTT**: because the best stack already includes strong legal TTT, the cleanest first ablation is likely `TTT_ENABLED=0` to measure whether curriculum MTP improves the base model itself before post-hoc adaptation.
- **Compile interactions**: this candidate specifically avoids Python-side scheduling flags for MTP, but the broader stack still contains other compile-sensitive paths.
