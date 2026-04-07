# Neural Cache Evaluation on the 11L GPTQ-lite Stack

## Hypothesis

The next strong lever in this repo is an **evaluation-time continuous cache**: reuse already-scored validation-prefix hidden states and realized next tokens to sharpen next-token probabilities, without changing weights and with almost no artifact-byte cost.

## Why this is promising here

This repository has already shown that **evaluation changes can be first-class gains**:

- `2026-03-19_SlidingWindowEval` turned the naive baseline into a much stronger scorer purely through better prefix usage at eval time.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` pushed further with legal score-first test-time adaptation.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` is the strongest non-TTT stack that is still straightforward to extend.

The cache idea fits those constraints well: it is prefix-only, requires no extra training infrastructure, and layers on top of the existing 11-layer recipe instead of replacing it.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Eval-only gains:** `records/track_10min_16mb/2026-03-19_SlidingWindowEval/`
- **Legal prefix-only adaptation precedent:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Core architectural stack:** the 03-20 to 03-22 line of 11-layer XSA/EMA/Partial-RoPE/SmearGate/BigramHash/GPTQ-lite records

## External research that informed it

- Grave et al., **Improving Neural Language Models with a Continuous Cache** (arXiv:1612.04426)  
  <https://arxiv.org/abs/1612.04426>
- Merity et al., **Pointer Sentinel Mixture Models** (arXiv:1609.07843)  
  <https://arxiv.org/abs/1609.07843>
- Khandelwal et al., **Generalization through Memorization: Nearest Neighbor Language Models** (arXiv:1911.00172)  
  <https://arxiv.org/abs/1911.00172>

These papers all point at the same opportunity: a strong parametric LM can often improve next-token prediction by interpolating with a small non-parametric memory over the recent prefix.

## What changed vs. the chosen base

This candidate starts from the `2026-03-22` 11-layer EMA + GPTQ-lite stack and keeps its training/export recipe intact. The new pieces are:

1. **Cache-ready eval interface**
   - Refactored the model backbone so eval can return both final hidden states and logits.
   - Added `forward_hidden_and_logits()` for cache scoring.

2. **Legal continuous-cache evaluation**
   - Added `eval_val_sliding_cache()`.
   - The cache stores normalized final hidden states plus realized next tokens.
   - Cache updates happen **only after** those tokens have been scored.
   - The current implementation evaluates the cache on **rank 0 only** for correctness under distributed runs.

3. **Config knobs for cache tuning**
   - `CACHE_EVAL_ENABLED`
   - `CACHE_EVAL_LAMBDA`
   - `CACHE_EVAL_THETA`
   - `CACHE_EVAL_SIZE`
   - `CACHE_EVAL_BLOCK_SIZE`

4. **Portability/smoke-test improvements**
   - Defaults for `DATA_PATH` and `TOKENIZER_PATH` are now resolved relative to the repository root, so the script can be run from this candidate directory directly.
   - Added an SDPA fallback when `flash_attn_interface` is unavailable.
   - Added `SMOKE_TEST=1` to exercise the backbone, attention fallback, and cache-ready forward path on CPU without dataset shards.

## How to run

From this candidate directory:

```bash
cd candidates/202604070746_neural-cache-eval
CACHE_EVAL_ENABLED=1 \
CACHE_EVAL_LAMBDA=0.20 \
CACHE_EVAL_THETA=12.0 \
CACHE_EVAL_SIZE=2048 \
CACHE_EVAL_BLOCK_SIZE=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The inherited defaults target the same 11-layer stack as the `2026-03-22` record and now resolve the dataset/tokenizer from the repository root automatically.
`CACHE_EVAL_ENABLED` stays **off by default** in code so a plain `torchrun --standalone --nproc_per_node=8 train_gpt.py` still gives a budget-safer baseline path, but the candidate should be evaluated with the explicit cache-enabled command above.

Useful cache knobs:

```bash
CACHE_EVAL_ENABLED=1 \
CACHE_EVAL_LAMBDA=0.20 \
CACHE_EVAL_THETA=12.0 \
CACHE_EVAL_SIZE=2048 \
CACHE_EVAL_BLOCK_SIZE=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

With cache eval disabled, the script preserves the base record's canonical final-score behavior. When cache eval is enabled, it still logs the standard roundtrip and sliding-window metrics, then emits the cache metric and uses it for the final `final_int8_zlib_roundtrip_exact` line.

## Validation run for this candidate

Validation was intentionally lightweight:

1. Syntax compilation:

   ```bash
   python -m compileall train_gpt.py
   ```

   Outcome: **passed**

2. Minimal CPU smoke test:

   ```bash
   SMOKE_TEST=1 \
   NUM_LAYERS=2 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 \
   TRAIN_SEQ_LEN=32 EVAL_SEQ_LEN=32 \
   BIGRAM_VOCAB_SIZE=256 BIGRAM_DIM=64 \
   VE_ENABLED=0 XSA_LAST_N=0 \
   python train_gpt.py
   ```

   Outcome:

   ```text
   smoke_test_ok loss:6.9385 hidden_shape:(2, 32, 128) logits_shape:(2, 32, 1024)
   ```

No full GPU run was attempted here, so the cache hyperparameters remain uncalibrated.

## Main risks and tradeoffs

- **Eval-time budget:** the cache evaluator is more expensive than plain sliding-window eval and is currently implemented on rank 0 only.
- **Hyperparameter sensitivity:** `lambda`, cache size, and similarity temperature will likely matter a lot.
- **Overlap with existing inductive bias:** BigramHash + SmearGate already help local repetition, so the cache may help mainly on longer or rarer recurring patterns.
- **Conservative distributed implementation:** correctness came first; a future version could parallelize cache eval more carefully.
