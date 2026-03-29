# Anchored TTT on the LeakyReLU2 + Legal TTT stack

## Hypothesis

The best in-tree result already gets a meaningful gain from legal score-first test-time training (TTT), but that sequential adaptation step is still vulnerable to drift: each chunk update can overfit to the most recent local distribution and gradually forget broadly useful weights.

This candidate adds a lightweight **proximal anchor** during TTT. Every chunk is still scored before any update, but the SGD adaptation loss now includes a small penalty that keeps the adapted model close to the pre-TTT quantized weights:

```python
loss = next_token_loss + TTT_ANCHOR_LAMBDA * sum((p - p0).square().sum() for p, p0 in zip(ttt_params, anchor_refs))
```

The expected result is a better tradeoff between local adaptation and global retention, especially late in the validation stream where forgetting compounds.

## Why this is promising for this repository

Repo history says two things very clearly:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the current best local result at **1.1194 val_bpb**, and its README attributes about **-0.0025 BPB** to legal score-first TTT.
- The 2026-03-22 and 2026-03-23 records already look close to saturated on the training-side recipe, so the highest-upside low-code intervention is probably inside the TTT loop rather than another large architecture rewrite.

This candidate keeps the same training stack, quantization path, and artifact shape as the 2026-03-23 record, so it aims for upside without giving back size or throughput.

## Prior repository work that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

Related influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` for the non-TTT best stack and the idea that most training-side gains are already harvested.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` for the cautionary note that seemingly helpful extra training tricks can silently do nothing under `torch.compile` if they rely on frozen class attributes.

There were no prior `candidates/` directories in the repository when this candidate was created.

## External research that informed it

- **End-to-End Test-Time Training for Long Context** (`arXiv:2512.23675`) frames long-context language modeling as continual learning and shows that next-token TTT can be a strong, scalable mechanism rather than just an eval hack.
- **Revisiting Realistic Test-Time Training** (`arXiv:2303.10856`) argues that sequential TTT is the realistic setting and that plain self-training is prone to confirmation bias; their stronger method adds an anchored regularizer.
- **Forgetting is Everywhere** (`arXiv:2511.04666`) gives a recent general account of forgetting as loss of self-consistency during adaptation, which is exactly the failure mode this candidate is trying to suppress in chunk-by-chunk TTT.

The implementation here is intentionally simpler than TTAC++: instead of source/target clustering, it uses a proximal weight anchor because that maps cleanly onto the existing Parameter Golf codepath and adds almost no infrastructure.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added a new hyperparameter: `TTT_ANCHOR_LAMBDA` (default `0.001`).
2. Snapshotted the unfrozen TTT parameters once before sequential adaptation begins.
3. Added an L2 anchor penalty to each TTT update so the adapted model stays near the pre-TTT checkpoint.
4. Logged the anchor setting and latest anchor penalty during TTT progress output.

What did **not** change:

- model architecture,
- quantization/export logic,
- training recipe,
- legal score-first evaluation ordering,
- artifact parameter count.

That means any gain would come from the TTT behavior itself, not from extra parameters or a different compression budget.

## How to run / evaluate

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
TTT_ANCHOR_LAMBDA=0.001 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

A good first sweep is `TTT_ANCHOR_LAMBDA` in `{0.0, 0.0005, 0.001, 0.002, 0.005}`.

## Expected risks / tradeoffs

- If `TTT_ANCHOR_LAMBDA` is too large, adaptation may become too conservative and give back the TTT gain.
- If it is too small, this candidate will collapse back to the existing 2026-03-23 behavior.
- The penalty adds extra TTT-time math proportional to the unfrozen parameter set, so evaluation becomes a bit more expensive even though training and artifact size stay essentially unchanged.
- This is a purely research-motivated candidate; it has not yet been tuned on the target 8xH100 setup.

## Validation

Commands run in this repository:

```bash
python -m compileall candidates/202603291713_anchored-ttt/train_gpt.py
python - <<'PY'
from pathlib import Path
import glob
import importlib.util
candidate = Path('candidates/202603291713_anchored-ttt/train_gpt.py')
print(f'candidate_exists={candidate.exists()}')
print(f'torch_installed={importlib.util.find_spec("torch") is not None}')
print(f'flash_attn_interface_installed={importlib.util.find_spec("flash_attn_interface") is not None}')
print(f'train_shards={len(glob.glob("data/datasets/fineweb10B_sp1024/fineweb_train_*.bin"))}')
print(f'val_shards={len(glob.glob("data/datasets/fineweb10B_sp1024/fineweb_val_*.bin"))}')
PY
```

Outcomes:

- `python -m compileall ...` **passed**.
- Feasibility probe reported:
  - `candidate_exists=True`
  - `torch_installed=False`
  - `flash_attn_interface_installed=False`
  - `train_shards=0`
  - `val_shards=0`

Because this runner is missing `torch`, `flash_attn_interface`, and the FineWeb shard dataset, a true runtime smoke test was **not feasible here**. The candidate script also hard-requires CUDA during execution, so a CPU-only launch would not be representative even with dependencies installed.
