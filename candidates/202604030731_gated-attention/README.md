# 202604030731_gated-attention

## Hypothesis

Enable the dormant **gated attention** path on top of the strongest current record stack so attention heads can cheaply express "do nothing" behavior without pushing logits toward extreme values. The goal is to reduce attention outliers and improve the int6+lzma export path under the 16MB cap while keeping the rest of the proven 11L/XSA4/LeakyReLU^2/legal-TTT recipe intact.

## Why this is promising here

- Recent record gains in this repository were dominated by **quantization-aware architectural tweaks** rather than larger structural changes: XSA, partial RoPE, LN scaling, EMA, GPTQ-lite clip search, and then LeakyReLU^2 plus legal TTT.
- The repo's own non-record exploration showed that **layer recurrence/depth reuse regressed badly** under a fixed wall-clock budget, so another cheap attention-path improvement is a better fit than a bigger architectural swing.
- The current best record code already contains an untested `gated_attention` path, but no record README, submission metadata, or prior candidate actually turned it on.

## Prior repository work that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — chosen base implementation and overall recipe to preserve.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — evidence that quantization-focused refinements can still move the frontier on the 11L stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` — partial RoPE + LN scaling remain part of the carried-forward core.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` — especially its negative layer-recurrence result, which argues against spending this candidate on depth reuse.
- There were **no prior `candidates/` directories** in the repo at the time this candidate was created.

## External research

1. Bondarenko, Nagel, Blankevoort, **"Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing"** (arXiv:2306.12929). The paper motivates gated attention as a simple way to reduce activation outliers caused by attention heads trying to approximate a no-op through extreme softmax logits.
2. Liu et al., **"MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"** (arXiv:2402.14905). This reinforces the repository's empirical trend that compact LMs are highly architecture-sensitive, so a targeted attention change on a deep-thin GQA stack is a sensible next bet.

I considered the paper's clipped-softmax variant too, but kept this candidate to **gated attention only** because that integrates cleanly with the existing FlashAttention-oriented code path and avoids kernel-risk in the 10-minute training regime.

## What changed versus the base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Turn `GATED_ATTENTION` **on by default** and expose `ATTN_GATE_BIAS` as a tunable init knob.
2. Keep the rest of the best-record stack unchanged: 11L, XSA on last 4 layers, VE128, partial RoPE, LN scaling, LeakyReLU^2 MLP, EMA + tight SWA, GPTQ-lite int6 export, and legal score-first TTT.
3. Make default dataset/tokenizer paths **resolve relative to the repository root**, so the script still works when run from inside this candidate directory.
4. Add a **CPU/SDPA attention fallback** when `flash_attn_interface` is unavailable or tensors are not on CUDA, so the attention path itself no longer hard-depends on the FlashAttention binding for import or toy smoke checks.

## How to run / evaluate

Run from this candidate directory:

```bash
cd candidates/202604030731_gated-attention
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 ROPE_DIMS=16 LN_SCALE=1 \
LATE_QAT_THRESHOLD=0.15 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
GATED_ATTENTION=1 ATTN_GATE_BIAS=4.0 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- EMA is still applied at the inherited fixed decay of `0.997` inside the script.
- The candidate keeps `VALUE_RESIDUAL=0` by default so the main experimental variable stays focused on gated attention.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604030731_gated-attention/train_gpt.py
python - <<'PY'
import importlib.util
import pathlib
path = pathlib.Path("candidates/202604030731_gated-attention/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcomes:

- `python -m compileall ...` **parsed successfully**.
- A toy script import smoke test was **blocked by missing repo Python dependencies in this container** (`ModuleNotFoundError: No module named 'numpy'`; a separate dependency check also showed `torch` is unavailable here). Full training is also not runnable here because the challenge recipe expects CUDA, FlashAttention, and the prepared FineWeb shard export.

## Main risks / tradeoffs

- Gated attention may simply damp useful heads instead of mainly suppressing no-op outliers, in which case it could hurt the already-strong 2026-03-23 stack.
- The extra per-token gate projection is small, but it is still a little more work in the attention block.
- The candidate improves portability with an SDPA fallback, but the leaderboard-relevant path still assumes the normal CUDA + FlashAttention training environment.
