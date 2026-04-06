# SVFormer-Style Value Residual on the March 23 Stack

## Hypothesis

The strongest current stack in this repository already gets a lot from zero- or tiny-parameter tricks: XSA, partial RoPE, LN scaling, EMA/SWA, BigramHash, shared value embeddings, and LeakyReLU(0.5)^2. A **value residual** is a natural next step because it adds another cheap information-preserving path: later layers can blend their own value stream with the first block's raw values, preserving token identity deeper into the network without growing the large parameter banks.

The candidate specifically tests an **SVFormer-style first-layer value reuse** on top of the current best March 23 architecture by promoting an already-present dormant value-residual path into the default experimental configuration for this fork.

## Why this is promising here

This challenge is bottlenecked by compressed artifact size and 10-minute wallclock, so the best ideas have been the ones that improve quality without adding many bytes or much infrastructure. Recent records in this repo show exactly that pattern:

- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` proved deep zero-parameter attention surgery plus EMA mattered.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` showed partial RoPE + LN scaling were additive and cheap.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved mostly through better averaging/export quality instead of larger architecture changes.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` added LeakyReLU(0.5)^2 and reached the current best stack.

Value residuals fit the same pattern: very small extra state, no extra big matrix banks, and a direct claim that deeper transformers preserve information better under tight parameter budgets.

## Records and prior work that influenced this candidate

- **Chosen base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Additional repository influences:**
  - `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
- **Prior candidates reviewed:** none existed in `candidates/` when this candidate was created.

## External research that informed the choice

1. **Value Residual Learning** (Zhou et al., 2024/2025, arXiv:2410.17897) argues that standard hidden-state residuals do not preserve token-level information well enough in deep transformers, and that adding value residual connections can match the same validation loss with fewer parameters and less data. Its SVFormer variant explicitly reuses the first layer's value signal.
2. **Intra-Layer Recurrence in Transformers for Language Modeling** (Nguyen & Lin, 2025, arXiv:2505.01855) was considered because parameter reuse is attractive under a hard artifact cap, but repo evidence already shows recurrence/depth-reuse attempts can lose too much wallclock.
3. **Scaling Law for Quantization-Aware Training** (Chen et al., 2025, arXiv:2505.14302) reinforced the decision to keep the existing proven mixed quantization/export stack intact instead of changing multiple compression variables at once.

## What changed versus the chosen base

Relative to the March 23 record code, this candidate keeps the same overall 11-layer LeakyReLU(0.5)^2 + XSA + partial RoPE + LN scale + shared value embedding + GPTQ-lite/TTT stack, but turns the existing dormant **value residual path into the default candidate behavior**:

- `VALUE_RESIDUAL` now defaults to `1`
- `BIGRAM_VOCAB_SIZE` now defaults to `1536` to match the March 23 run command
- `TTT_FREEZE_BLOCKS` now defaults to `0` to match the stronger March 23 legal-TTT setting
- a short code comment documents the intended first-layer value reuse

The actual value residual implementation is intentionally unchanged from the latent path already present in the March 23 script: later blocks blend their current values with the first block's raw values through a learned 2-vector `vr_lambda`. This candidate is therefore a **configuration/defaults fork of a previously unsubmitted code path**, which is exactly what makes it attractive as a fast next experiment.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
VALUE_RESIDUAL=1 SWA_ENABLED=1 SWA_EVERY=50 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a cheaper training-side check, leave `TTT_ENABLED=0` and inspect the post-EMA / post-quant metrics before paying the full legal-TTT evaluation cost.

## Expected risks and tradeoffs

- Reusing first-layer values may over-anchor token identity and reduce useful layer specialization.
- The extra cross-layer dependency could raise activation memory and slightly hurt steps/second.
- The best `vr_lambda` behavior is still untuned; if this underperforms, the next sweep should target the initial mix and which layers receive the residual.
- Value residuals may interact non-trivially with XSA and shared value embeddings because all three manipulate the value path.

## Validation

Executed validation in this workflow:

1. **Passed:** `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604060308_svformer-value-residual/train_gpt.py`
2. **Blocked by environment:** attempted a CPU smoke test using a temporary `flash_attn_interface` shim, but this runner does not have `torch` installed, so an actual forward-pass startup check was not feasible here
3. **Passed:** candidate-only code review after implementation; the two review findings were addressed (restored unique default `run_id`, and clarified that this is a defaults/config fork of an already-present dormant value-residual path)
