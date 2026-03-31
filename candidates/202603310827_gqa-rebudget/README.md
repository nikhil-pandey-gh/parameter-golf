# GQA Rebudget: 2-KV-head stack with lexical/value reinvestment

## Hypothesis

The current strongest stacks appear to overpay for KV-head capacity: every serious record keeps `NUM_KV_HEADS=4`, even while the repo's biggest consistent gains come from quantization-aware compression, lexical side channels (`BigramHash`, `SmearGate`), and deep-layer value reinjection. This candidate tests whether a more aggressive grouped-query setup (`NUM_KV_HEADS=2`) can reclaim enough parameter/artifact budget from the K/V path to fund stronger lexical/value priors without paying a meaningful quality penalty.

Concretely, the hypothesis is that **2 KV heads is still enough at 512d / 8 query heads**, and that spending the saved budget on a larger `BigramHash` table and a wider shared value embedding is a better trade for this challenge than keeping 4 KV heads.

## Why this is promising for this repository

Repository evidence points in two directions:

- The strongest record stack keeps accumulating gains from **cheap lexical/value-side structure**: `BigramHash`, `SmearGate`, `VE`, XSA, partial RoPE, EMA/SWA, and better quantization.
- At the same time, I could not find any serious sweep of **more aggressive KV sharing** under `records/`; the runs consistently stay at `NUM_KV_HEADS=4`.

That makes KV-head count a clean unexplored axis. This candidate is intentionally safer than recurrence or full shared-depth ideas: it keeps the winning 11-layer LeakyReLU^2 + XSA + partial-RoPE + GPTQ-lite + legal-TTT-capable stack intact, but shifts part of the parameter budget away from K/V projections and toward already-proven lexical/value helpers.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Chosen base implementation. It provides the strongest current stack: LeakyReLU(0.5)^2, XSA, partial RoPE, LN scaling, VE, GPTQ-lite int6 export, Parallel Muon, and optional legal score-first TTT.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Reinforced that VE and GPTQ-lite are worth preserving, and that post-training quantization quality matters enough to shape the architecture.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Showed partial RoPE + LN scale are real signal, not noise.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - Strong evidence that `BigramHash` is one of the best places to spend saved bytes.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Useful negative evidence: naive recurrence/shared-depth was already harmful in a short-budget setting, so this candidate deliberately avoids that direction.

## External research that informed it

- **Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need"** (`arXiv:1911.02150`)
  - Introduced multi-query attention: share K/V across heads, cut memory bandwidth sharply, and often lose only a small amount of quality.
- **Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"** (`arXiv:2305.13245`)
  - The most relevant primary source for this candidate. GQA is explicitly the middle ground between MHA and MQA: fewer KV heads with quality closer to full multi-head attention.
- **DeepSeek-AI, "DeepSeek-V2"** (`arXiv:2405.04434`)
  - Motivates the broader principle that aggressively compressing the KV path can be a worthwhile architectural trade when efficiency matters.

This candidate does **not** copy MLA or DeepSeek-V2's architecture directly. It only borrows the higher-level lesson that KV-path compression can be a good budget trade when paired with strong surrounding components.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Candidate changes:

1. **More aggressive grouped-query attention by default**
   - `NUM_KV_HEADS`: `4 -> 2`
2. **Reinvest some of the saved budget into lexical priors**
   - `BIGRAM_VOCAB_SIZE`: `2048 -> 3072`
3. **Reinvest some of the saved budget into shared value reinjection**
   - `VE_DIM`: `128 -> 192`
4. **Local fallback for environments without FlashAttention**
   - If `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA so the module can still import and run local smoke checks when `torch` is installed.

Everything else stays aligned with the strongest known stack: 11 layers, 512d model width, MLP 3x, LeakyReLU(0.5)^2, XSA on the deepest 4 layers, partial RoPE, LN scale, GPTQ-lite-style export, and optional legal TTT.

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603310827_gqa-rebudget

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate defaults already encode the core idea (`NUM_KV_HEADS=2`, `BIGRAM_VOCAB_SIZE=3072`, `VE_DIM=192`).
The script also resolves its default `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so running from this candidate directory does not require extra path overrides in the normal repo layout.

To be explicit, the equivalent override form is:

```bash
NUM_KV_HEADS=2 \
BIGRAM_VOCAB_SIZE=3072 \
VE_DIM=192 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
SWA_ENABLED=1 \
LATE_QAT_THRESHOLD=0.15 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to evaluate with the same legal TTT pathway used by the 2026-03-23 record, additionally enable:

```bash
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 \
TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

I ran the lowest-cost checks available in this workflow environment.

### Successful

```bash
python -m compileall candidates/202603310827_gqa-rebudget/train_gpt.py
```

Outcome: **passed**.

### Not feasible in this runner

Attempted a minimal CPU import/forward smoke test through a tiny 2-layer model instance.

Command form:

```bash
python - <<'PY'
# import candidate module, instantiate tiny GPT, run one forward loss
PY
```

Outcome: **not feasible here** because the runner's `python` environment does not have PyTorch installed:

```text
ModuleNotFoundError: No module named 'torch'
```

So this workflow could confirm syntax, but not a real runtime forward pass.

## Main expected risks and tradeoffs

- **Quality risk from fewer KV heads**: 2 KV heads may still be too aggressive at this model size, even if GQA literature suggests the trade can be favorable.
- **Rebudget placement may be wrong**: the saved parameters might be better spent on VE layers, bigram dimension, or another small structural helper rather than the particular `3072 / 192` split chosen here.
- **Interaction risk with XSA + VE**: XSA and shared value embeddings are both late-layer value-path interventions. Fewer KV heads could either make them more complementary or make them fight each other.
- **Top-stack variance**: because the base is already highly optimized, small architecture changes can help or hurt by very small margins. This candidate is best treated as a focused sweep target, not an assumed win.

## Suggested next experiments

If this candidate is promising after a first run, the next natural grid is small and surgical:

- `NUM_KV_HEADS in {1, 2, 4}`
- `BIGRAM_VOCAB_SIZE in {2048, 3072, 4096}`
- `VE_DIM in {128, 192, 256}`
- optional `TTT_ENABLED in {0, 1}`

That would quickly reveal whether the win comes from the more aggressive GQA choice itself or from the rebudget target.
