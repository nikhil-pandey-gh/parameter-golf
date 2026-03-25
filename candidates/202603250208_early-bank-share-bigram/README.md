# Early Encoder Bank Sharing + Bigger Bigram Budget

## Hypothesis

The current frontier in this repo already suggests that the deepest layers are specialized and valuable: XSA is only helpful late, value embeddings are only used late, and the best stack keeps adding late-layer refinements rather than flattening the model. My hypothesis is that **one early encoder layer's heavy matrices can be shared across two effective depths with only a small quality hit**, while the saved unique-weight budget is better reinvested into a larger token-pair feature budget.

Concretely, this candidate reuses one early banked transformer layer via `BANK_LAYER_SCHEDULE=0,1,1,2,3,4,5,6,7,8,9`, keeps all per-layer norms / scales / residual mixing / q-gains unique, and raises `BIGRAM_VOCAB_SIZE` from the recent default regime to `3072`.

## Why this is promising for this repository

Recent records show a strong pattern:

- deeper `11L` stacks beat shallower models,
- late-layer specialization keeps helping (`XSA_LAST_N=4`, VE on layers `9,10`),
- compression quality is now good enough that the best remaining gains may come from **better budget allocation**, not just more of the same,
- larger bigram tables have already shown positive movement in prior runs.

That makes this repo a good fit for **selective** parameter sharing rather than naive full tying. If early layers are more generic while late layers stay specialized, sharing only one early heavy bank is a lower-risk way to buy extra artifact budget than tying the whole network.

## Prior records and history that influenced this candidate

This candidate is intentionally built on the strongest recent code path rather than the root baseline.

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Key influences:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` for GPTQ-lite export and the strong pre-TTT 11L stack.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for Partial RoPE + layerwise scaling.
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` for the late-layer XSA / EMA template.
- `2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` and the 2026-03-23 README ablations for evidence that **bigger bigram budgets** can pay off.

There were no prior `candidates/` folders in the repository when this candidate was created.

## External research that informed it

This design is mainly motivated by selective parameter sharing / recurrent-depth work:

- **Subformer** (Reid et al., arXiv:2101.00234) argues that *sandwich-style* sharing is better than naive cross-layer tying for generative transformers.
- **ALBERT** (Lan et al., arXiv:1909.11942) is the canonical evidence that cross-layer sharing can dramatically reduce parameter count while preserving quality.
- **Basis Sharing** (Wang et al., arXiv:2410.03765) shows that cross-layer sharing is especially attractive under strong compression pressure.
- **Intra-Layer Recurrence** (Nguyen and Lin, arXiv:2505.01855) reports that recurrence targeted at earlier layers can be especially effective.
- **Universal Transformers** (Dehghani et al., arXiv:1807.03819) provide the broader recurrent-depth framing for trading parameters for repeated computation.

## What changed vs. the chosen base implementation

Base file copied from:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `BANK_LAYER_SCHEDULE` and compact bank allocation.
2. Only the heavy banked matrices (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`) are shared.
3. Per-layer control parameters stay unique because each effective layer still has its own `Block` instance.
4. Quantization/export now unbanks and re-banks **unique bank layers** rather than duplicating shared weights.
5. Default `BIGRAM_VOCAB_SIZE` is increased to `3072`.
6. Default dataset/tokenizer paths are resolved relative to the repository root so the script can be run directly from this candidate directory.
7. Added a `SMOKE_TEST=1` path and a CPU-safe attention fallback so the file can do a cheap local startup check in environments that have `torch` installed but do not have FlashAttention.

## How to run

From the repository root:

```bash
cd candidates/202603250208_early-bank-share-bigram

NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=3072 \
BANK_LAYER_SCHEDULE=0,1,1,2,3,4,5,6,7,8,9 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
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

For a lightweight local startup check from the repository root in an environment with `torch` available:

```bash
cd candidates/202603250208_early-bank-share-bigram
SMOKE_TEST=1 python train_gpt.py
```

## Expected tradeoffs / risks

- Sharing even one early heavy layer may still cost more modeling quality than the bigger bigram table buys back.
- The exact shared-layer placement (`0,1,1,2,3,4,5,6,7,8,9`) is a research guess, not a tuned optimum.
- A larger bigram table can improve modeling but may compress worse than expected if its learned distribution is less zstd/lzma-friendly.
- The TTT interaction with shared early banks is untested; it may help, be neutral, or partially erase the architectural gain.

## Validation run in this environment

Executed here:

```bash
python -m compileall candidates/202603250208_early-bank-share-bigram/train_gpt.py
```

Outcome:

- Passed.

Attempted but not feasible in this runner:

```bash
cd candidates/202603250208_early-bank-share-bigram
SMOKE_TEST=1 python train_gpt.py
```

Outcome:

- Not runnable in this workflow container because the Python environment does not have `torch` installed, so a real model-instantiation smoke test could not be completed here.
- The script still includes the `SMOKE_TEST=1` path for use in a proper repo runtime.
