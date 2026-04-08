# Curriculum MTP on the 11L EMA + GPTQ-lite Base

## Hypothesis

Tiny language models in this repo are already close to the point where extra training efficiency matters more than another architectural rewrite. Multi-token prediction (MTP) is a strong candidate for that efficiency gain, but recent work suggests small models benefit when MTP is introduced gradually rather than at full strength from step 0. The hypothesis here is that a **training-budget-aware forward MTP curriculum** will improve the 11-layer pre-TTT stack's sample efficiency without increasing exported artifact size, because the extra heads are training-only and are excluded from export.

## Why this is promising here

- The cleanest strong base in-repo is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which already combines the best durable ideas from the earlier record line: 11 layers, MLP3x, XSA, partial RoPE, LN scale, EMA, GPTQ-lite, BigramHash, and VE128.
- That base already contains dormant MTP hooks, so this idea fits the repository's current implementation style instead of requiring new infrastructure.
- Prior records improved by attacking evaluation, quantization, or small architecture knobs; none of the published record READMEs used a real MTP curriculum as the main idea.
- The repo also contains evidence that compile-folded late-QAT can be misleading, so moving the "training-only extra complexity" budget to curriculum MTP is a cleaner bet than leaning on another QAT variant.

## Prior experiments that influenced this candidate

- **Primary base:** `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - Best clean pre-TTT stack in the repo (`val_bpb` mean `1.1233`).
  - Already has the right backbone for a low-risk training-objective change.
- **Supporting influences:**
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` clarified that the late-QAT branch can be compile-fragile, so this candidate does not rely on it by default.
  - `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` showed there is still headroom in training-only or eval-only additions, but also that TTT adds meaningful evaluation complexity; this candidate keeps the simpler pre-TTT path.
  - `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` and `2026-03-20_11L_EfficientPartialXSA_FA3_SWA120` established the 11L/XSA/EMA line as the strongest reusable architecture family.

## External research

- **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737, 2024).**
  - Reports that predicting multiple future tokens improves sample efficiency and encourages stronger induction-style behavior while keeping the main trunk unchanged.
- **Ansar Aynetdinov and Alan Akbik, "Pre-Training Curriculum for Multi-Token Prediction in Language Models" (arXiv:2505.22757, ACL 2025).**
  - Specifically relevant here because it argues that **small language models struggle with static MTP**, and that a **forward curriculum** helps them benefit from MTP rather than being hurt by it.

## What changed vs. the chosen base

1. **Enabled MTP by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`.
2. **Added a training-budget-aware forward curriculum** via `MTP_START_FRAC` and `MTP_END_FRAC`.
   - The first extra horizon ramps in before the second one.
   - The schedule keys off the same measured training-time fraction used by the inherited `MAX_WALLCLOCK_SECONDS` stop logic, so it stays aligned with the trainer's existing budget accounting.
3. **Plumbed per-step MTP head scales through the compiled model call** instead of treating MTP as a static on/off feature.
4. **Disabled late QAT by default** (`LATE_QAT_THRESHOLD=0.0`) so the candidate does not depend on the compile-fragile branch noted in prior repo work.
5. **Made the script runnable from this candidate directory** by resolving the dataset/tokenizer defaults relative to the repository root.
6. **Added a FlashAttention fallback** to PyTorch SDPA so the script can still import and run when `flash_attn_interface` is unavailable.

The exported artifact still excludes the MTP heads, so the idea targets training efficiency rather than artifact compression.

## How to run

From this candidate directory:

```bash
cd candidates/202604081123_curriculum-mtp
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
MTP_NUM_HEADS=2
MTP_LOSS_WEIGHT=0.15
MTP_START_FRAC=0.30
MTP_END_FRAC=0.80
LATE_QAT_THRESHOLD=0.0
```

The script defaults its dataset/tokenizer paths to the repository's shared `data/` directory, so it can be launched directly from this folder.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202604081123_curriculum-mtp/train_gpt.py
```

Outcome: **passed**

```bash
/tmp/gh-aw/agent/pg-venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202604081123_curriculum-mtp/train_gpt.py').resolve()
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.GPT(
    vocab_size=64,
    num_layers=4,
    model_dim=64,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2.0,
    tie_embeddings=True,
    tied_embed_init_std=0.01,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    mtp_num_heads=2,
    mtp_loss_weight=0.15,
    bigram_vocab_size=32,
    bigram_dim=16,
    xsa_last_n=1,
    rope_dims=8,
    ln_scale=True,
    dtg=False,
    ve_enabled=False,
).float()
model.train()
input_ids = torch.randint(0, 64, (2, 16), dtype=torch.int64)
target_ids = torch.randint(0, 64, (2, 16), dtype=torch.int64)
mtp_head_scales = torch.tensor([0.5, 0.0], dtype=torch.float32)
loss = model(input_ids, target_ids, mtp_head_scales)
print(loss.item())
PY
```

Outcome: **passed** (sample run printed `cpu_smoke_loss=4.3205`)

```bash
/tmp/gh-aw/agent/pg-venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202604081123_curriculum-mtp/train_gpt.py').resolve()
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.GPT(
    vocab_size=64,
    num_layers=2,
    model_dim=32,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2.0,
    tie_embeddings=True,
    tied_embed_init_std=0.01,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    mtp_num_heads=2,
    mtp_loss_weight=0.15,
    bigram_vocab_size=16,
    bigram_dim=8,
    xsa_last_n=1,
    rope_dims=4,
    ln_scale=True,
    dtg=False,
    ve_enabled=False,
).float()
compiled = torch.compile(model, dynamic=False, fullgraph=True)
compiled.train()
input_ids = torch.randint(0, 64, (2, 8), dtype=torch.int64)
target_ids = torch.randint(0, 64, (2, 8), dtype=torch.int64)
mtp_head_scales = torch.tensor([0.5, 0.0], dtype=torch.float32)
loss = compiled(input_ids, target_ids, mtp_head_scales)
print(loss.item())
PY
```

Outcome: **passed** (sample run printed `compiled_cpu_smoke_loss=4.3205`)

Notes:

- The runner image did not have the repo's Python dependencies preinstalled in system Python, so the smoke test used a temporary virtualenv in `/tmp/gh-aw/agent/pg-venv`.
- A full training run was not attempted in this environment because the script requires CUDA and the repository's real dataset shards.

## Main risks and tradeoffs

- **Throughput risk:** even training-only MTP heads add extra projection work, so the schedule has to improve sample efficiency enough to pay for fewer steps.
- **Curriculum sensitivity:** the `0.30 -> 0.80` ramp is plausible but still heuristic; too-early MTP can hurt optimization, while too-late MTP may not matter.
- **Quantization tradeoff:** disabling late QAT by default may give up a tiny quantization gain, although the repo evidence suggests that path is not trustworthy unless reworked to be compile-safe.
- **Fallback speed:** the SDPA fallback is useful for portability and smoke checks, but Hopper + FA3 should remain the intended fast path for real training.
