# Candidate: Diagonal PIT Tying

## Hypothesis

The strongest pre-TTT Parameter Golf stacks still rely on hard tied embeddings, even though recent evidence suggests the shared token table is one of the last under-explored bottlenecks in compact LMs. A lightweight **diagonal pseudo-inverse tying (PIT)** transform should improve the input/output token interface with almost no artifact cost, helping both pre-quant training quality and post-quant robustness.

## Why this is promising for this repository

Two repo trends point at the same bottleneck:

1. `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600` showed that handling the embedding path more carefully nearly eliminated the quantization gap for `tok_emb.weight`.
2. The strongest pure training/export stack before TTT, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, already packs in 11L, 3x MLP, XSA, Partial RoPE, LN scale, EMA, GPTQ-lite, and value embeddings. Changing the token interface is one of the few high-upside axes that remains mostly untouched.

This candidate keeps that stable 2026-03-22 stack and changes only the tied embedding interface.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- **Embedding/quantization signal:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600`
- **Current frontier context:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`

I intentionally forked the 2026-03-22 stack rather than the 2026-03-23 record because 2026-03-22 is the strongest **pre-TTT, non-banked, standard-parameter** codebase in the repo. That makes it a better place to isolate a new representation-level idea.

## External research that informed it

- **Gu et al., "Rethinking Weight Tying: Pseudo-Inverse Tying for Stable LM Training and Updates" (arXiv:2602.04556).** PIT argues that hard tying does not preserve a stable token interface and proposes a pseudo-inverse-consistent shared memory plus hidden-space transform.
- **Lopardo et al., "Weight Tying Biases Token Embeddings Towards the Output Space" (arXiv:2603.26663).** This paper finds that tied embeddings are pulled toward the unembedding role early in training, weakening input representations and harming early-layer computation, with explicit relevance to smaller LMs.

The full PIT paper uses a richer SPD transform. For Parameter Golf, I adapted the core idea into a **bounded diagonal transform** so the candidate stays simple, cheap, and artifact-friendly.

## What changed versus the chosen base

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Adds `PIT_ENABLED` and `PIT_DIAG_BOUND` hyperparameters.
2. Adds a learned `pit_log_scale` vector (one scalar per hidden channel).
3. Scales token embeddings by the inverse diagonal transform on input.
4. Applies the matching forward transform before projecting hidden states through the tied token matrix on output.
5. Keeps the transform as a small fp32 control tensor during mixed-precision export.
6. Adds a PyTorch SDPA fallback when FlashAttention 3 is unavailable or unsupported on the active device, making the module easier to import outside the Hopper setup.

The rest of the model stays intentionally close to the proven 2026-03-22 recipe.

## How to run / evaluate

Run from this candidate directory:

```bash
RUN_ID=diag_pit_seed1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
PIT_ENABLED=1 PIT_DIAG_BOUND=0.35 \
BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
SWA_ENABLED=1 SWA_EVERY=50 \
LATE_QAT_THRESHOLD=0.15 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For an ablation, set `PIT_ENABLED=0`.

## Validation

Recorded in this environment:

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604081837_diag-pit-tying/train_gpt.py`  
  **Outcome:** Python compilation succeeded for the root scripts, `data/`, and this candidate script.
- Minimal CPU import/forward smoke test for the candidate module  
  **Outcome:** not feasible on this runner because the available Python environment does not have `torch` installed.
- Full training smoke test  
  **Outcome:** not feasible here without the challenge runtime dependencies (`torch`/CUDA stack); the local `data/datasets/fineweb10B_sp1024` dataset is also absent on this runner.

## Main risks / tradeoffs

- A diagonal PIT transform is much weaker than the full SPD PIT formulation, so the interface correction may be too small.
- The new scale vector could interact poorly with the tuned tied-embedding LR or with quantization if it drifts too aggressively.
- If the main gain from full PIT comes from richer cross-channel structure, this compact adaptation may not move the score enough.
