# MTP Auxiliary Head on the March 23 LeakyReLU2 + Legal TTT Stack

## Hypothesis

The strongest next low-risk idea for this repo is to make **multi-token prediction (MTP)** a real training signal on top of the current best published stack. Recent MTP work shows better sample efficiency from auxiliary future-token heads, and this repository already contains an MTP path in the March 23 record code. The catch is that the prior implementation never optimized those heads, so the auxiliary loss was effectively inert.

This candidate activates a conservative **1-head MTP objective** with a small loss weight, keeps the March 23 architecture and export path intact, and still drops the auxiliary heads before serialization so the submission artifact size stays unchanged.

## Why this is promising here

The repo history points to a clear pattern:

- the largest wins came from **evaluation-aware tricks** (sliding-window eval, then legal score-first TTT),
- the best training stacks are **compression-aware 10-11 layer models** with SmearGate, BigramHash, XSA, Partial RoPE, EMA/SWA, and aggressive int6 export,
- the current best record is the March 23 **LeakyReLU2 + Legal TTT + Parallel Muon** run at **1.1194 bpb**.

There were **no prior experiments under `candidates/`** at review time, so the comparison set is entirely the root baseline plus `records/`.

MTP fits this challenge unusually well because:

1. it improves the training objective without changing eval semantics,
2. the extra decoder heads are **training-only** and excluded from export,
3. the code path already exists in the strongest base implementation,
4. tiny models under fixed wall-clock budgets benefit most from better sample efficiency.

## Prior repository work that informed this candidate

- `train_gpt.py` in the repo root: baseline 9-layer 512d launch point, still using int8+zlib export.
- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`: established the 11-layer, MLP3x, compression-aware direction.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`: showed XSA + EMA was stronger than the prior SWA-only stack.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`: tightened the export path with GPTQ-lite clip search and longer warmdown.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`: best current base; this candidate forks its `train_gpt.py`.

## External research that informed it

- **Gloeckle et al., "Better & Faster Large Language Models via Multi-Token Prediction" (arXiv:2404.19737)**: argues that predicting multiple future tokens from a shared trunk improves sample efficiency and helps induction-head / algorithmic behavior.
- **DeepSeek-V3 Technical Report (arXiv:2412.19437)**: uses a multi-token prediction training objective in a modern high-performing LM stack, reinforcing that MTP remains a competitive training-side improvement rather than a one-off trick.

The key repo-specific observation is that MTP is attractive here precisely because it can be added as an **auxiliary training loss with zero artifact-cost increase** once the auxiliary heads are omitted from the exported checkpoint.

## What changed vs. the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Candidate changes:

1. **Default MTP is enabled** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15`.
2. **MTP heads are actually optimized.** The March 23 script defined MTP heads and loss terms but did not place those heads in any optimizer group, so they never trained.
3. **MTP heads are warm-started from the main decoder weights** (the tied embedding matrix when embeddings are tied), so the auxiliary loss contributes useful gradients immediately instead of waiting for a zero-initialized head to learn from scratch.
4. **Candidate-relative defaults now work from inside the candidate directory.** The copied script now resolves `DATA_PATH` and `TOKENIZER_PATH` from the repository root by default.
5. **Export behavior is unchanged.** `mtp_heads.*` tensors are still excluded before serialization, so the artifact budget story is the same as the March 23 base.

## How to run

From the repository root:

```bash
cd candidates/202604020059_mtp-aux-head
NUM_LAYERS=11 XSA_LAST_N=4 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 HEAD_LR=0.008 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- Running from the candidate directory is now supported by default because the script resolves repo-root `data/` and `tokenizers/` paths automatically.
- For direct A/B testing against the March 23 base, keep the non-MTP settings matched and sweep only `MTP_NUM_HEADS`, `MTP_LOSS_WEIGHT`, and optionally `HEAD_LR`.

## How to evaluate

The script keeps the March 23 evaluation flow:

1. train under the 600-second cap,
2. apply EMA (or LAWA if explicitly enabled),
3. export int6 + lzma while excluding `mtp_heads.*`,
4. run sliding-window validation on the round-tripped model,
5. optionally run legal score-first TTT if `TTT_ENABLED=1`.

## Main risks and tradeoffs

- **Training cost:** even one extra vocab projection head adds compute; the MTP gain has to beat the lost steps under a fixed 10-minute budget.
- **Short-run sensitivity:** this repo trains for only a few thousand effective steps, so auxiliary-loss weights that are harmless in larger-scale papers can still over-regularize here.
- **TTT interaction:** a better pre-TTT model is good, but the March 23 stack already gets a sizable post-TTT gain; the effects may not add linearly.
- **Implementation uncertainty:** the repo had dormant MTP support but no published ablation for it, so this candidate is evidence-backed but still genuinely exploratory.

## Validation

Commands run during candidate preparation:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604020059_mtp-aux-head/train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path("candidates/202604020059_mtp-aux-head/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_mtp", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcomes:

- both `compileall` commands succeeded,
- the live import smoke test was **not feasible in this runner** because the repo's Python runtime dependencies (for example `numpy`, `torch`, and `sentencepiece`) are not installed here,
- a full CPU-only run is also not a meaningful proxy for this script because the training and eval path is explicitly CUDA / FlashAttention based.
