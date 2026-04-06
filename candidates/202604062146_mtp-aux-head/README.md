# Candidate: Single-head MTP on the LeakyReLU² + Parallel Muon stack

## Hypothesis

Add a **single training-only multi-token-prediction (MTP) head** to the current best record stack so the shared trunk gets a denser short-horizon supervision signal during the 10-minute training budget, while keeping the exported artifact unchanged because the auxiliary head is removed before quantized export.

## Why this is promising here

The repository's biggest durable wins already came from:

- better evaluation (`SlidingWindowEval`, legal TTT),
- better low-bit export (`MixedQuant`, QAT, GPTQ-lite),
- and careful architectural polish on the same 11-layer family (XSA, EMA, partial RoPE, LeakyReLU²).

That leaves **training sample efficiency at fixed wallclock** as one of the cleanest remaining levers. This candidate targets exactly that lever with a training-only objective that should stack with the existing quantization and evaluation pipeline instead of replacing it.

## Prior repository work that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the direct base because it is the strongest overall record and already bought back some training throughput with parameter banking + Parallel Muon.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` reinforced that the export path is already competitive, so the next experiment should avoid destabilizing quantization unless necessary.
- `records/track_10min_16mb/2026-03-17_LoRA_TTT/` and `records/track_10min_16mb/2026-03-19_SlidingWindowEval/` showed that the repo has already harvested many evaluation-only gains; this candidate instead tries to improve the trained trunk before evaluation.
- Across the March 20-23 records, the MTP code path existed but every logged run kept `mtp_num_heads:0`, so it remains an untried lever in this repo.

## External research that informed it

- **Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737)** argues that multi-token prediction improves sample efficiency and downstream capability while keeping inference overhead out of the base next-token model.
- **Frantar et al., "GPTQ" (arXiv:2210.17323)** and the repo's own GPTQ-lite follow-up support leaving the weight-only export path alone rather than coupling a new candidate to another export rewrite.
- **Tang et al., "AWQ" (arXiv:2306.00978)** reinforces the same general lesson: once the low-bit pipeline is competitive, preserving it and improving the trained representation can be a better next move than yet another quantizer tweak.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Set `MTP_NUM_HEADS` default from `0` to `1`.
2. Set `MTP_LOSS_WEIGHT` default from `0.2` to a more conservative `0.1`.
3. Wired MTP head parameters into the head optimizer and replicated-gradient all-reduce path so the auxiliary loss actually updates both the MTP heads and the shared trunk.
4. Added an inline code comment documenting that MTP heads are training-only and excluded from export.

Everything else stays on the proven March 23 stack so the experiment isolates the new idea.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected risks and tradeoffs

- The extra head adds training-time compute, so the main risk is losing enough steps to erase the sample-efficiency gain.
- MTP may help the pre-TTT trunk but interact weakly with the already-strong legal TTT path, reducing net gain.
- A single auxiliary head is deliberately conservative; if it works, the obvious follow-up is a sweep over `MTP_NUM_HEADS in {1,2}` and `MTP_LOSS_WEIGHT`.

## Validation

Commands run for this candidate:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604062146_mtp-aux-head/train_gpt.py
python - <<'PY'
from pathlib import Path

text = Path("candidates/202604062146_mtp-aux-head/train_gpt.py").read_text(encoding="utf-8")
assert 'MTP_NUM_HEADS", 1' in text
assert 'MTP_LOSS_WEIGHT", 0.1' in text
assert 'if "mtp_heads" not in k' in text
assert 'head_params.extend(list(base_model.mtp_heads.parameters()))' in text
print("static_candidate_checks_ok")
PY
```

Outcomes:

- repo + candidate `compileall`: passed
- static candidate checks: passed

I did **not** run a full CPU import/forward or training-step smoke test, because this runner does not have the repo's Python dependencies installed and the script expects the challenge CUDA/FlashAttention stack plus the real FineWeb shard layout. In this environment, a "smoke test" would only report missing runtime prerequisites rather than anything meaningful about the candidate logic.
