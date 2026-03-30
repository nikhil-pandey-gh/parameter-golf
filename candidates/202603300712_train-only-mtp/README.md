# Training-only MTP on the LeakyReLU^2 + Legal TTT + Parallel Muon stack

## Hypothesis

The strongest next candidate in this repo is to turn on **training-only multi-token prediction (MTP)** on top of the current best stack.

The key idea is simple: use one auxiliary future-token head during training to improve sample efficiency under the fixed 600-second wallclock budget, then **drop the auxiliary head before export** so it does not consume artifact bytes at evaluation time.

## Why this is promising for this repository

The record history suggests that this challenge is no longer bottlenecked by "just train longer" or by broad architecture rewrites.

Instead, the winning runs cluster around three things:

- better sample efficiency inside the same 10-minute budget,
- evaluation-aware improvements such as sliding/TTT,
- compression-aware export and averaging.

That makes MTP attractive here:

- the current record line already has a strong trunk, strong eval recipe, and export logic;
- recent MTP work argues that predicting multiple future tokens improves sample efficiency and induction behavior;
- this codebase already contains an MTP implementation path, but every reviewed record kept `MTP_NUM_HEADS=0`.

So this candidate is deliberately conservative on infrastructure and aggressive on leverage: reuse the current best stack and activate the one underexplored training signal that should help most under a hard wallclock cap.

## Prior records and experiments that influenced this candidate

Primary local influences:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - current best record in this repo (`val_bpb 1.1194`), used as the implementation base.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - showed the strength of the 11-layer EMA + GPTQ-lite + warmdown stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - established partial RoPE + LN scale as durable zero-byte wins.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - shows the core 11-layer/XSA/EMA/int6 line that most later improvements stack on.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/`
  - reminder that evaluation-aware methods can deliver outsized gains without retraining.
- `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
  - argues that post-training quantization quality is a primary bottleneck.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - useful negative evidence that extra training time alone is not enough.

## External research that informed it

Primary source:

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"**, arXiv:2404.19737.
  - <https://arxiv.org/abs/2404.19737>
  - Key takeaway used here: auxiliary multi-token prediction can improve sample efficiency while sharing the same trunk.

Why it fits this repo specifically:

- the challenge is wallclock-limited, so sample efficiency matters more than elegant but slow architectural detours;
- the candidate already has the machinery to exclude `mtp_heads` from the serialized artifact, so the training signal is nearly "free" with respect to artifact bytes.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Change in this candidate:

- copied that script into this directory;
- changed the default `MTP_NUM_HEADS` from `0` to `1`;
- wired `mtp_heads` into the Adam head optimizer so the auxiliary objective actually updates its output heads during training;
- resolved the default dataset/tokenizer paths from the repository root so the script can be launched from inside this candidate directory.

Important detail:

- the base script already excludes `mtp_heads` from the exported state dict and rebuilds the final evaluation model with `mtp_num_heads=0`, so the auxiliary head is **training-only**.

In other words, this candidate intentionally changes the training objective while keeping the serialized inference artifact aligned with the challenge budget.

## How to run / evaluate

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.2 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The only intended hypothesis change relative to the base record command is `MTP_NUM_HEADS=1`.
- `MTP_LOSS_WEIGHT=0.2` is left at the script's existing value.
- If you want the cleanest ablation, compare the same command with `MTP_NUM_HEADS=0`.

## Main expected risks / tradeoffs

- **Training throughput risk**: the auxiliary head adds extra logits and cross-entropy work every step. If it slows step time too much, the sample-efficiency gain may be erased.
- **Scale mismatch risk**: the MTP paper's strongest gains were demonstrated on much larger models; the effect on an 11-layer 512-dim model may be smaller.
- **Metric mismatch risk**: MTP improves training efficiency, but this challenge is scored after quantization and compression. Better next-token learning does not automatically imply better post-quantization BPB.
- **Stack interaction risk**: the current best recipe already includes EMA, GPTQ-lite-style export, LeakyReLU^2, legal TTT, and Parallel Muon. The marginal return from one more training signal may be positive but small.

## Validation

Commands run locally in this workflow:

```bash
python -m compileall candidates/202603300712_train-only-mtp/train_gpt.py
```

Outcome:

- passed.

```bash
python - <<'PY'
from pathlib import Path
text = Path('candidates/202603300712_train-only-mtp/train_gpt.py').read_text(encoding='utf-8')
checks = {
    'default_mtp_enabled': 'MTP_NUM_HEADS", 1' in text,
    'export_strips_mtp': 'if "mtp_heads" not in k' in text,
    'eval_model_mtp_zero': 'mtp_num_heads=0, mtp_loss_weight=0.0' in text,
    'mtp_heads_optimized': 'head_params.extend(list(base_model.mtp_heads.parameters()))' in text,
    'repo_root_paths': 'REPO_ROOT / \"data\" / \"datasets\" / \"fineweb10B_sp1024\"' in text,
}
for k, v in checks.items():
    print(k, int(v))
PY
```

Outcome:

- passed (`default_mtp_enabled=1`, `export_strips_mtp=1`, `eval_model_mtp_zero=1`, `mtp_heads_optimized=1`, `repo_root_paths=1`).

Attempted smoke import:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('candidates/202603300712_train-only-mtp/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
```

Outcome:

- failed in this workflow environment before candidate execution with `ModuleNotFoundError: No module named 'numpy'`.
- Because the environment is also missing challenge runtime dependencies, a true CPU start-to-first-step smoke test was not feasible here.
