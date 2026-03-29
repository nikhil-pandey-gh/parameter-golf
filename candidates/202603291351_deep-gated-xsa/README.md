# Deep-Gated XSA Candidate

## Hypothesis

The strongest recent runs in this repository improve by making the deepest attention layers more selective: XSA removes self-value leakage, Partial RoPE limits unnecessary positional bias, and legal score-first TTT sharpens evaluation. This candidate pushes that same direction one step further by adding **per-token head gates on only the deepest XSA layers**, plus a **normalized value-residual highway** across those layers so XSA can suppress self-noise without completely discarding token-local information.

## Why this is promising here

Repository evidence says the best recent gains are small, composable changes on the 11-layer XSA stack rather than wholesale architecture changes. The `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` record already ships dormant `gated_attention` and `value_residual` hooks, but no prior record README reports using them. That makes this a low-infrastructure way to test a genuinely new attention-selectivity idea on top of the current best stack.

## Prior repository influences

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` supplied the base stack: 11L, LeakyReLU(0.5)^2, XSA on the last 4 layers, Partial RoPE, VE, EMA+SWA, GPTQ-lite-style int6 export, legal TTT, and Parallel Muon.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` reinforced that tiny export-quality improvements still matter at the top end.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md` and `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md` showed that targeting the deepest attention layers is a reliable source of incremental gains.
- The repository-wide review also highlighted dead ends: recurrence, brute-force longer training, and long-context-only changes have all underperformed relative to export-aware deep-stack improvements.

## External research that informed it

- **Exclusive Self Attention (XSA)**, arXiv:2603.09078, argues that deep self-attention improves when self-value information is explicitly excluded so attention focuses on contextual novelty.
- **Diff Transformer**, arXiv:2410.05258, motivates suppressing irrelevant-context noise in attention maps rather than only scaling model size.
- **FLASH / Gated Attention Unit**, arXiv:2202.10447, supports lightweight gating as an efficient way to improve attention selectivity without large parameter cost.
- **GLU Variants Improve Transformer**, arXiv:2002.05202, helps justify keeping the repo’s recent bias toward selective multiplicative control when a low-cost gate is available.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `GATED_ATTENTION_LAST_N` and `VALUE_RESIDUAL_LAST_N` hyperparameters, defaulting to `4`, so the new mechanism applies only to the deepest 4 layers instead of globally.
2. Kept the existing head-gate module but restrict it to the same late layers already benefiting from XSA.
3. Seeded the carried value anchor from the block immediately before the deep residual band, so all selected deep layers can actually mix against the same value reference.
4. Changed the value-residual mixer from an unconstrained 2-vector to a **softmax-normalized** 2-logit convex combination (`vr_logits`), making the deep value highway better behaved.
5. Left the rest of the 2026-03-23 stack intact: LeakyReLU(0.5)^2, XSA, Partial RoPE, VE, EMA+SWA, Parallel Muon, and legal TTT.

## How to run

From the candidate directory:

```bash
cd candidates/202603291351_deep-gated-xsa
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
GATED_ATTENTION_LAST_N=4 VALUE_RESIDUAL_LAST_N=4 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
python train_gpt.py
```

For actual leaderboard-style runs, use `torchrun --standalone --nproc_per_node=8 train_gpt.py` in the intended H100 environment, exactly as with the record folders.

## Validation performed

Commands run during candidate creation:

```bash
# From the repository root:
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603291351_deep-gated-xsa/train_gpt.py
python - <<'PY'
import ast
from pathlib import Path
path = Path("candidates/202603291351_deep-gated-xsa/train_gpt.py")
module = ast.parse(path.read_text(encoding="utf-8"))
# Verified that GPT.__init__ accepts the new layer-range args and forwards
# per-layer gated/value-residual settings into Block construction.
PY

# From candidates/202603291351_deep-gated-xsa:
python -m compileall train_gpt.py
```

Outcome so far:

- Baseline repo compile check: passed.
- Candidate script compile check: passed.
- Static AST wiring check: passed.
- Import/CPU smoke run: not feasible in this container because `torch` is not installed, so even a stubbed non-CUDA import path cannot be executed safely without adding new infrastructure.

## Main risks / tradeoffs

- The new gates may be too conservative if the open-bias initialization leaves them near identity throughout training.
- The deep value highway could partially undo XSA if the model over-relies on the carried value anchor.
- Even with negligible parameter cost, there is some extra per-step compute from late-layer gating.
- This is a targeted hypothesis, not a verified improvement yet; the main follow-up should be an ablation of gate-only vs gate+value-highway on the same seed/config.
