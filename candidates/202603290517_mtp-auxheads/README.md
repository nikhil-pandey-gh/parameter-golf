# 202603290517_mtp-auxheads

## Hypothesis

The strongest next candidate is to add **real multi-token prediction (MTP) auxiliary training** on top of the current best in-repo stack (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) while keeping the final artifact unchanged.

This repository already contains dormant MTP scaffolding in recent 11-layer record scripts, but no prior record actually enabled `MTP_NUM_HEADS > 0`, and the heads were not wired into any optimizer group. That means the code path existed, but the auxiliary heads would not learn. This candidate turns MTP into a real experiment by:

1. enabling training-only MTP heads by default,
2. adding the MTP head weights to the AdamW optimizer path,
3. warm-starting the MTP heads from the tied embedding / LM head weights so the auxiliary loss can influence the trunk immediately,
4. still stripping the MTP heads from the exported artifact so the 16MB budget is unchanged.

## Why this is promising for this repository

The repo history suggests the frontier is now dominated by **small, orthogonal improvements on the strongest 11-layer stacks** rather than broad architecture resets:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` shows the current best in-tree stack comes from stacking several small wins: LeakyReLU(0.5)^2, legal score-first TTT, and faster Parallel Muon.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` shows that even low-cost post-training refinements like GPTQ-lite clip search still matter.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md` and `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` show that the best-performing family is the 11-layer, MLP3x, XSA, EMA / warmdown-tuned line.

MTP is attractive here because it targets **training efficiency**, not model size. That is a strong fit for Parameter Golf: the auxiliary heads are cheap, they are only used during training, and they are already excluded from export in the base code path.

## Prior records that influenced this candidate

The main base is:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Important historical influences:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - strongest current stack
  - LeakyReLU(0.5)^2
  - score-first legal TTT
  - Parallel Muon / parameter banking
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - evidence that small orthogonal improvements still move the needle on this stack family
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/train_gpt.py`
  - confirms the repo had MTP hooks in code, but no record turned them on in practice

## External research that informed the choice

Primary sources considered:

- **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-Token Prediction" (arXiv:2404.19737)**
  - argues that predicting multiple future tokens through auxiliary heads improves sample efficiency and develops stronger induction / algorithmic behavior.
- **DeepSeek-V3 Technical Report (arXiv:2412.19437)**
  - explicitly reports using a multi-token prediction objective in a modern frontier model stack.
- **DeepSeek-V2 (arXiv:2405.04434)** and **QuaRot (arXiv:2404.00456)** were also reviewed.
  - Both are interesting, but MLA and rotation-based quantization would require broader attention / quantization refactors than this repo currently supports with a minimal candidate.

Why MTP won over those alternatives:

- it is the most compatible with the existing code,
- it adds no export-time artifact cost,
- it attacks training efficiency directly under the 10-minute cap,
- it was already half-present in the codebase, reducing implementation risk.

## What changed versus the chosen base implementation

Base copied from:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Candidate-specific changes in `train_gpt.py`:

- default `RUN_ID` set to this candidate name
- default `MTP_NUM_HEADS=2`
- default `MTP_LOSS_WEIGHT=0.15`
- added `MTP_INIT_FROM_BASE=1` default
- default `BIGRAM_VOCAB_SIZE=1536` to match the top record configuration
- default `TTT_ENABLED=1` and `TTT_FREEZE_BLOCKS=0` to match the top record recipe
- **fixed optimizer wiring** so MTP head weights are included in the AdamW parameter set
- **warm-started the MTP heads from the base output weights** (`tok_emb.weight` when embeddings are tied)
- kept the existing export behavior that removes `mtp_heads.*` from the serialized artifact

In short: this candidate converts the repo's dormant MTP path into an actual training feature while preserving the final submission footprint.

## How to run

From the candidate directory:

```bash
cd candidates/202603290517_mtp-auxheads

RUN_ID=202603290517_mtp_auxheads \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key candidate defaults already live in the script, but you can override them explicitly if desired:

```bash
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_INIT_FROM_BASE=1 \
BIGRAM_VOCAB_SIZE=1536 \
TTT_ENABLED=1 \
TTT_FREEZE_BLOCKS=0
```

## Expected tradeoffs / risks

- **Extra training compute**: MTP adds extra output projections and losses, so step time may rise slightly.
- **Objective mismatch**: if the auxiliary loss is weighted too heavily, next-token quality may worsen.
- **Warm-start bias**: initializing future-token heads from the base output weights may help early learning, but it could also reduce head diversity.
- **No GPU run in this environment**: this candidate has only static validation here; score impact is still an empirical question.

## Validation

Validation run in this workflow:

1. Syntax / bytecode validation for the candidate:

```bash
python -m compileall candidates/202603290517_mtp-auxheads/train_gpt.py
```

Outcome: **passed**.

2. Baseline repo compile check plus candidate:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603290517_mtp-auxheads/train_gpt.py
```

Outcome: **passed**.

3. Static structural sanity check for the new MTP path:

```bash
python - <<'PY'
from pathlib import Path
path = Path('candidates/202603290517_mtp-auxheads/train_gpt.py')
source = path.read_text()
compile(source, str(path), 'exec')
required = {
    'mtp_init_default': 'mtp_init_from_base = bool(int(os.environ.get("MTP_INIT_FROM_BASE", "1")))',
    'mtp_optimizer': 'for head in base_model.mtp_heads:\n        scalar_params.append(head.weight)',
    'mtp_warm_start': 'head.weight.data.copy_(base_weight.data)',
    'mtp_export_strip': 'if "mtp_heads" not in k',
}
missing = [name for name, needle in required.items() if needle not in source]
if missing:
    raise SystemExit(f'missing expected snippets: {missing}')
print('static_check_ok')
PY
```

Outcome: **passed** (`static_check_ok`).

4. Import / smoke-test probe:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('candidates/202603290517_mtp-auxheads/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('import_ok')
PY
```

Outcome: **not feasible in this runner** because the environment is missing Python runtime dependencies from `requirements.txt` (`ModuleNotFoundError: No module named 'numpy'`) before the script can be imported. A real CPU/GPU runtime validation should be done in the standard repo environment with dependencies installed.
