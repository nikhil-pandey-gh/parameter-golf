# Sandwich Bank Sharing + Bigger Hash Lexicon

## Hypothesis

The current best banked stack already keeps much of its layer-specific behavior in lightweight per-layer norms, scales, skip weights, XSA placement, and value-embedding scales. That makes it a good fit for **sandwich-style late-layer bank sharing**: reuse the heavy attention/MLP banks across the deepest refinement layers while keeping the lightweight per-layer modulators unique. The saved bytes can then be reinvested into a larger hashed lexical memory (`BIGRAM_VOCAB_SIZE=3072`) instead of simply shrinking the artifact.

## Why this is promising for this repository

The strongest records in this repository all follow the same pattern: use better evaluation, embedding-safe quantization, compression-aware training, and small architectural side channels to buy capacity under the 16 MB cap. A non-record recurrence sweep found that **extra looped depth hurts when it cuts the number of optimizer steps inside the 10-minute budget**. This candidate avoids that failure mode: logical depth and compute stay the same, while only the number of stored bank slices decreases.

## Which records or prior experiments influenced it

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the direct code base because it already packs the heavy transformer weights into four shared bank tensors, which makes selective sharing practical.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed that the non-TTT stack is already extremely competitive and that quantization/export details matter.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` established partial RoPE and LN scaling as durable wins.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` explicitly reported full layer recurrence as a negative result under a fixed wallclock budget, which is why this candidate reuses parameters without adding extra forward loops.

## Which external research informed it

- **ALBERT** ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942)) showed that cross-layer parameter sharing can substantially improve parameter efficiency.
- **Subformer** ([arXiv:2101.00234](https://arxiv.org/abs/2101.00234)) found that **sandwich-style sharing** works better than naive full tying for generative transformers.
- **Basis Sharing** ([arXiv:2410.03765](https://arxiv.org/abs/2410.03765)) showed that sharing across layers works best when each layer keeps lightweight unique coefficients rather than being forced into identical weights.
- **Two-Scale Latent Dynamics for Recurrent-Depth Transformers** ([arXiv:2509.23314](https://arxiv.org/abs/2509.23314)) is a useful contrast: recurrence can be valuable, but the repo's own fixed-wallclock evidence suggests borrowing the *parameter reuse* idea without adding extra passes.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate changes two things:

1. **Late sandwich bank sharing**
   - New env var: `BANK_LAYER_MAP`
   - If `BANK_LAYER_MAP` is unset, the script derives a compact default by leaving the early layers unique and reusing two bank slices in an `A,B,A,B` pattern across the deepest four logical layers. For the default 11-layer config, that implicit map is `0,1,2,3,4,5,6,7,8,7,8`.
   - Logical layers remain 11, but the heavy bank tensors now store only 9 physical slices.
   - The deepest four logical layers use an `A,B,A,B` bank pattern, while block-local scales, norms, skip weights, and value-embedding scales remain unique per logical layer.

2. **Bigger hashed lexical memory**
   - Default `BIGRAM_VOCAB_SIZE` increases from `2048` to `3072`.
   - The expectation is that bank sharing buys back artifact budget that can be spent on more lexical capacity.

The quantization/export path was also updated to quantize the bank tensors **directly**, so shared layers are not duplicated on disk during int6 export.
The compressed artifact `final_model.int6.ptz` stores the bank-layer map in its metadata, while raw `final_model.pt` stays a plain state dict for compatibility and is accompanied by a small `final_model.bank_map.pt` sidecar.

## How to run or evaluate it

Example training command:

```bash
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=3072 XSA_LAST_N=4 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
LATE_QAT_THRESHOLD=0.15 MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To disable sharing and recover the original bank layout:

```bash
BANK_LAYER_MAP=0,1,2,3,4,5,6,7,8,9,10
```

## Validation

Commands run for this candidate in this workflow:

```bash
python -m compileall candidates/202603280747_sandwich-bank-share/train_gpt.py
```

Outcome:

- `compileall` succeeds.
- A constructor-only smoke test was attempted next, but this workflow runtime does not have the training dependencies installed (`torch`, `numpy`, `sentencepiece`) and therefore cannot import the script deeply enough to instantiate `GPT`.

A true end-to-end CPU training smoke test is **not feasible** here because this script intentionally requires CUDA in `main()` and uses the FlashAttention interface in the forward path, and this workflow runtime also lacks the core Python ML dependencies needed for constructor-only import smoke tests.

## Main expected risks or tradeoffs

- Sharing the deepest banks may over-constrain late-layer specialization, especially because XSA and TTT both lean on upper-layer expressivity.
- The larger bigram table may not fully repay the representational capacity lost by bank sharing.
- Direct bank quantization should preserve sharing on disk, but the exact compression win depends on the entropy of the learned shared slices after training.
- The best sharing pattern is probably not the default `A,B,A,B`; future sweeps should test `A,B,C,A,B,C`, decoder-only sharing, and pairing bank sharing with larger VE settings or tighter GPTQ-lite clipping.
