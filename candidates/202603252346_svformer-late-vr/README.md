# SVFormer-style Late Value Residual on the Legal-TTT 11L Stack

## Hypothesis

The best local line already solved most of the obvious wins: sliding evaluation, 11-layer depth, Partial RoPE, XSA on late layers, VE128, aggressive export, and legal score-first TTT. The underexplored path is the **value stream**.

My hypothesis is that an **SVFormer-style value residual** will help this repository's deep 11-layer model preserve token identity through the deepest late-XSA layers without materially increasing parameter count or artifact size. In this candidate, the first captured value stream is reused only in the late stack, where information preservation is hardest and where this repo already concentrates specialized capacity.

## Why this is promising for this repository

Repository review showed three stable trends:

- More depth helps when export stays under control.
- Late specialized layers (`XSA_LAST_N=4`, `VE_LAYERS=9,10`, Partial RoPE) are where the best 11-layer runs keep adding capacity.
- Attention/logit stabilization is already well explored here via `q`/`k` RMS normalization, learned `q_gain`, and logit softcapping, so the next high-leverage idea is more likely to come from the **value pathway** than another q/k tweak.

This candidate adds only tiny control tensors while leaving the proven training/eval/export stack intact.

## Prior records and candidates that influenced this choice

There were no prior `candidates/` directories in this checkout.

The strongest local bases were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Best local score (`1.1194` mean post-TTT).
  - Supplies the exact training/eval/export path used here.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best clean non-TTT 11-layer stack.
  - Reinforced that the 11L + XSA + Partial RoPE + VE line is the right base family.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Important reminder that Partial RoPE helped, but late-QAT-as-implemented did not; the next idea should target a different bottleneck.

## External research that informed this candidate

- **Value Residual Learning / ResFormer / SVFormer** (`arXiv:2410.17897`): adds value residual connections to improve information flow in deep transformers, with the SVFormer variant specifically sharing the first layer's value embedding.
- **Methods of improving LLM training stability** (2024 arXiv search result): useful mainly as a rejection signal here; this repository already invests heavily in q/k/logit stabilization, so another q/k-centric candidate looked less novel.
- **Rope to Nope and Back Again** (2025 arXiv search result): another rejection signal; the repo already explored Partial RoPE and long-context variants, making value-path preservation the cleaner next move.

## What changed versus the chosen base implementation

Base file: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- Turn **value residual** on by default.
- Make the residual mixing **stable and bounded** by converting `vr_lambda` into a softmaxed convex combination instead of unconstrained raw coefficients.
- Capture the source value stream once, then apply value residuals only to the **late layers by default**.
  - Default behavior when `VALUE_RESIDUAL_LAYERS` is unset: reuse the source value stream in the last `XSA_LAST_N` layers.
  - With the current defaults, that means layers `7,8,9,10` in the 11-layer model.
  - If `XSA_LAST_N=0`, the script requires an explicit `VALUE_RESIDUAL_LAYERS=...` setting rather than silently broadening the residual path.
- Make default dataset/tokenizer paths **repo-relative**, so running from this candidate directory works without needing to `cd` back to the repo root.

## How to run / evaluate

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
VALUE_RESIDUAL=1 VALUE_RESIDUAL_SOURCE_LAYER=0 VALUE_RESIDUAL_LAYERS=7,8,9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- If `VALUE_RESIDUAL_LAYERS` is left empty, the script defaults to the last `XSA_LAST_N` layers.
- `VALUE_RESIDUAL_SOURCE_LAYER` must be earlier than every target in `VALUE_RESIDUAL_LAYERS`.
- Default `DATA_PATH` and `TOKENIZER_PATH` resolve from the repository root, so running this script from `candidates/202603252346_svformer-late-vr/` works without extra path overrides.

## Main expected risks / tradeoffs

- Reusing early-layer values may over-anchor late layers and reduce specialization if the residual is too strong.
- Legal TTT could interact nonlinearly with the late-layer residual path; this might help post-TTT more than pre-TTT, or vice versa.
- The candidate is intentionally narrow: it tests value-path preservation on top of the current best stack, not a broad architectural rewrite.

## Validation

Commands run in this repository:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603252346_svformer-late-vr/train_gpt.py
```

Outcomes:

- `python -m compileall train_gpt.py train_gpt_mlx.py data` ✅
- `python -m compileall candidates/202603252346_svformer-late-vr/train_gpt.py` ✅
- A true CPU smoke test was **not feasible** in this runner because the environment does not have `torch` installed, and this candidate also inherits the repository's FlashAttention/CUDA training path. Syntax validation was the safe low-cost check that did not distort runtime behavior.
