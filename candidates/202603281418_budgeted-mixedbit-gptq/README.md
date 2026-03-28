# Budgeted Mixed-Bit GPTQ-lite Search

## Hypothesis

The 16 MB artifact cap is still a quantization problem more than a training problem. Instead of using a fixed rule like "all MLP and attention weights are int6", this candidate assumes a small number of tensors are worth promoting to int8 if the promotion buys more reconstruction fidelity than it costs in compressed bytes.

## Why this is promising for this repository

Repository history shows that quantization decisions repeatedly moved the leaderboard:

- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

But those records still use fixed quantization recipes. The repo-wide review found no prior `candidates/` directory and no prior experiment that performs a compressed-size-aware, per-tensor promotion search under a global byte budget.

## Prior experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  This is the direct base. It already has the strongest clean pre-TTT stack in the repo: 11L, MLP3x, XSA4, Partial RoPE, LN Scale, VE128, SmearGate, BigramHash, EMA, warmdown 3500, and GPTQ-lite clip search.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  This run established that Partial RoPE and LN Scale mattered, and also documented that the old late-QAT path was ineffective under `torch.compile`. That made this candidate focus on export-side quantization rather than reviving that training path.

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  This is the current best overall result, but it adds TTT plus parameter-banking / parallel-Muon plumbing. I intentionally did not use it as the direct base because the goal here is to isolate a new compression idea, not a larger training-stack rewrite.

## External research that informed it

- Mengzhao Chen et al., **"Scaling Law for Quantization-Aware Training"** (`arXiv:2505.14302`)
  The main signal I used is that mixed precision helps when quantization error is concentrated in a subset of layers/tensors, especially around MLP bottlenecks and outlier-heavy paths.

- Mengzhao Chen et al., **"EfficientQAT: Efficient Quantization-Aware Training for Large Language Models"** (`arXiv:2407.11062`)
  I did not import their training algorithm directly, but I borrowed the broader idea that quantization quality improves when you stop treating all blocks uniformly.

- Yehui Tang et al., **"A Survey on Transformer Compression"** (`arXiv:2402.05964`)
  This reinforced that mixed-precision compression is often most effective when it is architecture-aware instead of globally uniform.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

- Replaced the fixed `mixed_quantize_int6(..., {"mlp", "attn"})` export rule with a **budgeted mixed-bit search**.
- The search starts from the original "MLP + attention = int6" baseline, then greedily tests **promoting individual tensors to int8**.
- Promotions are ranked by **reconstruction error reduction per added compressed byte**, using actual compressed bundle size checks instead of only raw tensor size proxies.
- A promotion is only kept if the resulting compressed payload still fits under:
  - `MAX_SUBMISSION_BYTES`
  - minus code bytes
  - minus `MIXEDBIT_SAFETY_MARGIN_BYTES`
- If even the all-int6 starting point is still over budget, the script now **raises instead of exporting and evaluating an invalid artifact**.
- Added a **flash-attn fallback** to PyTorch `scaled_dot_product_attention` for CPU-safe code paths.
- Added `SMOKE_TEST=1` mode for a tiny random-token sanity run.
- Set `LATE_QAT_THRESHOLD` default to `0.0` so this candidate stays focused on the export-side mixed-bit idea instead of carrying a legacy late-QAT default that was not central to the hypothesis.

## How to run / evaluate

Training / export on the intended GPU path:

```bash
cd candidates/202603281418_budgeted-mixedbit-gptq
SEED=1337 \
MAX_SUBMISSION_BYTES=16000000 \
MIXEDBIT_SAFETY_MARGIN_BYTES=32768 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

- `MAX_SUBMISSION_BYTES`: overall code + artifact budget, default `16000000`
- `MIXEDBIT_SAFETY_MARGIN_BYTES`: reserve headroom for compression variance, default `32768`
- `MIXEDBIT_MAX_PROMOTIONS`: cap the number of int8 promotions, default `0` (unlimited)
- `MIXEDBIT_MIN_ERROR_DELTA`: ignore tiny promotions, default `0.0`

Lightweight smoke mode:

```bash
cd candidates/202603281418_budgeted-mixedbit-gptq
SMOKE_TEST=1 python train_gpt.py
```

## Validation commands and outcomes

Passed:

```bash
python -m compileall candidates/202603281418_budgeted-mixedbit-gptq/train_gpt.py
```

Not feasible on this runner because repository runtime dependencies are not installed:

```bash
cd candidates/202603281418_budgeted-mixedbit-gptq
SMOKE_TEST=1 python train_gpt.py
```

Observed outcome:

```text
ModuleNotFoundError: No module named 'torch'
```

So the smoke-test path is implemented, but I could not execute it end to end in this workflow environment.

## Main expected risks / tradeoffs

- The search objective is **weight reconstruction error**, not validation BPB, so the chosen promotions may not perfectly track leaderboard gain.
- The selection is **greedy**, while full-bundle compression has interactions across tensors, so it may miss a better global combination.
- This candidate intentionally stays on the cleaner 2026-03-22 stack instead of the stronger but more invasive 2026-03-23 TTT / parameter-banking stack, so any gain here has to come from better export efficiency rather than the absolute best training recipe.
