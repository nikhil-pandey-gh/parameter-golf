# Primer-style depthwise QKV on the strongest 11L stack

## Hypothesis

The current best stacks already show that **cheap local structure** matters a lot in this repo: SmearGate and BigramHash became standard, while deeper 11-layer XSA/Partial-RoPE models kept winning. My hypothesis is that a **tiny causal depthwise convolution applied directly to Q/K/V projections** can capture richer short-range patterns than hash-and-gate features alone, while preserving the rest of the winning recipe and adding only a very small artifact cost.

## Why this is promising here

1. Repo evidence says local context hacks pay off. The strongest records repeatedly keep SmearGate and BigramHash because short-range token structure is high leverage under the 16MB budget.
2. The current leaderboard already uses the **ReLU-squared family** aggressively. Primer's language-model result says the other simple win was to add **depthwise convolution after Q/K/V projection**.
3. The repo has already shown that applying expensive attention tweaks only in the deepest layers can work well (`XSA_LAST_N`). This candidate uses the same idea for Primer-style local mixing to keep step-time risk bounded.

## Influential prior records

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - strongest current stack; this candidate copies that implementation as the base
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - kept the 11L/XSA/Partial-RoPE/VE recipe intact and showed later wins mostly came from small targeted deltas
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA`
  - established the importance of lightweight local token-pair features
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120`
  - motivated restricting the new mixer to the deepest layers rather than paying the full-model overhead everywhere

## External research

- **Primer: Searching for Efficient Transformers for Language Modeling** ([arXiv:2109.08668](https://arxiv.org/abs/2109.08668))
  - Primer attributes much of its LM improvement to two simple changes: **ReLU-squared activations** and **depthwise convolution after Q/K/V projections**
- **MetaFormer Baselines for Vision** ([arXiv:2210.13452](https://arxiv.org/abs/2210.13452))
  - useful secondary evidence that tiny conv/attention hybrids can be strong without broad infrastructure changes

## What changed vs. the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `PRIMER_ENABLED`, `PRIMER_LAST_N`, and `PRIMER_KERNEL_SIZE` hyperparameters.
2. Added **causal depthwise Q/K/V mixing** inside `CausalSelfAttention`.
3. Applied that mixer only to the **deepest 4 layers by default**.
4. Initialized each depthwise kernel to an **identity causal filter**, so the model starts as the base stack and only learns deviations that help.
5. Adjusted default dataset/tokenizer paths so `train_gpt.py` can be launched **from this candidate directory** without editing paths.

Everything else stays intentionally close to the strong March 23 base: 11 layers, XSA on late layers, Partial RoPE, LN scaling, VE layers, EMA/SWA path, GPTQ-lite-style int6 export, and optional legal TTT.

## How to run

From this directory (`candidates/202604031336_primer-depthwise-qkv`):

```bash
PRIMER_ENABLED=1 PRIMER_LAST_N=4 PRIMER_KERNEL_SIZE=3 \
BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
SWA_ENABLED=1 SWA_EVERY=50 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you are starting from the repository root, `cd candidates/202604031336_primer-depthwise-qkv` first.

To compare directly against the current top stack, keep the same evaluation recipe and optionally enable legal TTT:

```bash
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604031336_primer-depthwise-qkv/train_gpt.py` | Passed |
| Repo-root path resolution check for `Path(__file__).resolve().parents[2]` | Passed; resolves back to the repository root when launched from the candidate directory |
| CPU import/forward smoke with a stubbed `flash_attn_interface` | Attempted, but this workflow runner does not have `torch` installed (`python -m pip show torch` returned nothing), so an import-based smoke test was not feasible here |

## Main risks / tradeoffs

- The extra depthwise conv may cost enough step time to erase any modeling gain.
- Identity initialization is safe, but it may also learn too slowly inside a 600-second training budget.
- The new kernels are tiny, but they still add a small amount of passthrough artifact bytes.
- The interaction between Primer-style local mixing and the existing BigramHash/SmearGate/VE stack may be redundant rather than additive.

## Code review status

Code review found one README path issue in the initial draft ("From this directory" plus a redundant `cd ...` line). That mismatch has been fixed; no remaining substantive issues were identified in the candidate files.
