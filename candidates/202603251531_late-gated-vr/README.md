# Late-Gated Value-Residual Tail

## Hypothesis

The strongest recent Parameter Golf runs improve quality by making **small, selective changes in the deepest layers** rather than rewriting the whole model. This candidate tests whether adding two very cheap late-layer attention controls on top of the strongest existing stack will improve stability and quantization robustness:

1. **Input-conditioned attention output gates** on the last 4 layers.
2. **A shared first-layer value anchor** mixed back into the last 4 layers.

The intuition is that the deepest layers are where attention concentration and residual over-scaling matter most, so explicit rescaling and a lightweight cross-layer value shortcut may help more than applying the same mechanism everywhere.

## Why this is promising for this repository

Repository review points in the same direction:

- The biggest durable wins after the baseline came from **selective deep-layer edits**: XSA only on the last layers, VE only on layers 9-10, partial RoPE, LN scale, EMA, and then LeakyReLU^2 and legal TTT on top.
- The latest banked training script already contains **dormant `GATED_ATTENTION`, `VALUE_RESIDUAL`, and `DTG_ENABLED` switches** that are not claimed in the record README. That makes them unusually cheap to test here.
- The best non-TTT stack is the 2026-03-22 GPTQ-lite/EMA/Partial-RoPE family, while the best overall stack is the 2026-03-23 LeakyReLU^2 + legal TTT + Parallel Muon stack. This candidate uses the latter as a base, but turns the unexplored attention controls into the main experiment.

## Prior records and candidates that influenced this

There were **no prior `candidates/` directories** in this repository when this candidate was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best overall score in this repo snapshot
  - already includes the banked Muon path and dormant attention-control feature flags
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best strong non-TTT reference stack
  - confirms the value of 11L + XSA4 + Partial RoPE + VE + EMA + better quantization
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - shows that small zero/near-zero parameter deep-layer changes can still move the needle

## External research that informed this

- **Affine-Scaled Attention: Towards Flexible and Stable Transformer Attention** (`arXiv:2602.23057`)
  - argues that modest, learned reweighting of attention outputs can improve stability and optimization.
- **A Unified View of Attention and Residual Sinks: Outlier-Driven Rescaling is Essential for Transformer Training** (`arXiv:2601.22966`)
  - argues that explicit gated rescaling can absorb or mitigate sink behavior while improving robustness, including under quantization.
- **RealFormer: Transformer Likes Residual Attention** (`arXiv:2012.11747`)
  - shows that lightweight cross-layer attention shortcuts can stabilize deeper transformers and produce sparser, more useful attention patterns.

This candidate is not a literal reimplementation of those papers. Instead, it adapts their core ideas to the existing repository code in the smallest viable way.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Late-only gated attention by default**
   - `GATED_ATTENTION=1`
   - `ATTN_GATE_LAST_N=4`
   - applies the learned attention-output gate only to the deepest 4 layers

2. **Late-only value residual by default**
   - `VALUE_RESIDUAL=1`
   - `VALUE_RESIDUAL_LAST_N=4`
   - captures a value anchor from layer 0 and mixes it into only the deepest 4 layers

3. **Separate capture vs. mix semantics**
   - the first block captures the anchor value tensor
   - only the configured late layers perform the residual value mix

4. **CPU-safe attention fallback for smoke testing**
   - if `flash_attn_interface` is unavailable, the script now falls back to PyTorch SDPA
   - this is mainly for local import/smoke use; the intended fast path on GPU is still FlashAttention

Everything else is intentionally left close to the chosen base so the experiment stays attributable.

## How to run or evaluate it

From this directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
GATED_ATTENTION=1 ATTN_GATE_LAST_N=4 \
VALUE_RESIDUAL=1 VALUE_RESIDUAL_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to stack the published legal TTT recipe from the 2026-03-23 record on top of this candidate during evaluation, reuse the same TTT flags from that record README.

## Validation

Commands run for this candidate in this workflow:

```bash
python -m compileall candidates/202603251531_late-gated-vr/train_gpt.py
```

Outcome:

- **Passed**.

Attempted additional smoke validation:

- I attempted a minimal CPU import/forward smoke test through the new late-gating/value-residual path.
- That could not run in this workflow container because the local Python environment does not currently have the repo's declared `torch` dependency installed (`requirements.txt` includes `torch`).

## Main expected risks and tradeoffs

- The two mechanisms may be **too redundant with XSA**, causing over-suppression in the deepest layers instead of better conditioning.
- A **layer-0 value anchor** may be too early / too lexical, especially once VE and BigramHash are already active.
- The SDPA fallback is meant for smoke testing and portability, not for claiming training-speed parity with FlashAttention on H100.
- This candidate has only been syntax-checked in this workflow; the real question is whether the late-only rescaling path survives 8xH100 training and then still helps after int6 export and optional TTT.
