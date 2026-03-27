# Adaptive Mixed-Bit GPTQ-lite

## Hypothesis

The strongest next step for this repository is not another uniform-precision export rule, but a byte-budgeted mixed-precision export that spends precision where this 11-layer stack is most sensitive.

This candidate starts from the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack and replaces its fixed `int6-for-mlp+attn` export with a lightweight greedy allocator:

- start from `int5` MLP, `int6` attention/projection, `int8` embeddings,
- score each possible promotion by normalized reconstruction gain,
- weight that gain by layer depth and tensor role,
- then promote only the best tensors until the compressed artifact budget is nearly full.

The intuition is simple: previous records already showed that hand-tuned mixed precision helps, and recent mixed-precision quantization work keeps pointing toward sensitivity-aware precision assignment instead of uniform rules.

## Why this looks promising here

Repository history suggests that the largest wins came from better evaluation and better compression-aware modeling, not from small optimizer tweaks alone:

- the baseline moved from `1.2244` to `1.1925` mostly through sliding-window evaluation,
- then to the `1.16x -> 1.14x -> 1.12x` range through int6/mixed-bit export, bigger MLPs, SmearGate/BigramHash, EMA/SWA, XSA, Partial RoPE, and GPTQ-lite,
- the `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` record showed that fixed mixed precision can buy meaningful capacity,
- the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` record showed that even pure post-training clip search is still worth measurable BPB.

So the natural next step is to combine those ideas: keep the strong 11-layer stack, but replace static bit rules with a compressed-budget-aware mixed-bit policy.

## Prior repository influence

This candidate was primarily influenced by:

- `train_gpt.py` in the repo root for the baseline structure and export flow.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md` for the core observation that fixed mixed `int5/int6` rules can buy back bytes usefully.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` for the strong 11-layer Partial RoPE + LN-scale stack.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` as the chosen base implementation.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` for the LeakyReLU(0.5)^2 activation win and the `BigramHash 2048 -> 3072` positive ablation.

There was no pre-existing `candidates/` directory in this repository when this candidate was created.

## External research that informed this candidate

- **GPTQ** (`arXiv:2210.17323`) motivates using reconstruction-aware post-training quantization rather than naive clipping.
- **AWQ** (`arXiv:2306.00978`) emphasizes that not all weights are equally important; protecting a small sensitive subset can matter a lot.
- **A Survey on Transformer Compression** (`arXiv:2402.05964`) reinforces that quantization plus architecture-aware compression remains one of the most practical routes under tight artifact constraints.
- **Mixed-Precision Quantization for Language Models: Techniques and Prospects** (`arXiv:2510.16805`) frames mixed precision as the right tool when some transformer components are more precision-sensitive than others.
- **Beyond Outliers: A Data-Free Layer-wise Mixed-Precision Quantization Approach Driven by Numerical and Structural Dual-Sensitivity** (`arXiv:2603.17354`) is especially close in spirit: use lightweight structural and numerical cues to drive layerwise precision choices without expensive calibration-heavy search.

This implementation deliberately uses a very lightweight version of those ideas so it stays compatible with the repository's current training/export setup.

## What changed versus the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Replaces fixed mixed-int6 export with `mixed_quantize_budgeted(...)`.

   - Base policy: `int5` for MLP matrices, `int6` for attention/projection matrices, `int8` for embeddings.
   - Promotion policy: greedily rescue selected tensors upward (`int5 -> int6`, `int6 -> int8`) by ranking normalized reconstruction improvement per estimated compressed byte.
   - Sensitivity weighting is intentionally simple: later layers are weighted more heavily, and attention / qk / non-MLP projections are weighted above generic tensors.

2. Keeps the 11-layer Partial-RoPE/XSA/EMA/GPTQ-lite stack intact as the main training backbone.

3. Adopts **LeakyReLU(0.5)^2** in the MLP, following the later repository win.

4. Increases the default `BigramHash` bucket count from `2048` to `3072`, again following the later repository evidence.

5. Fixes the candidate defaults so it can be launched **from inside this candidate directory** while still finding the repo-root dataset and tokenizer.

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script defaults now point at the repo-root cached dataset and tokenizer:

- `data/datasets/fineweb10B_sp1024`
- `data/tokenizers/fineweb_1024_bpe.model`

Useful knobs:

```bash
SUBMISSION_BUDGET_BYTES=16000000 \
MIXED_QUANT_SAFETY_MARGIN=16384 \
ARTIFACT_COMPRESSOR=zlib \
BIGRAM_VOCAB_SIZE=3072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`ARTIFACT_COMPRESSOR` defaults to `zlib` here for portability. If you have `zstandard` installed and want to chase extra compression headroom, you can explicitly set `ARTIFACT_COMPRESSOR=zstd`.

For a smaller local run, reduce `TRAIN_BATCH_TOKENS`, `ITERATIONS`, and `nproc_per_node`.

## Expected tradeoffs / risks

- The allocator uses **weight reconstruction error**, not activation-calibrated loss deltas, so it may still mis-rank some matrices.
- The promotion search is deliberately lightweight and greedy; it is not a global knapsack solver.
- Because artifact size is governed by compressed bytes, local byte estimates can be imperfect; the script compensates by checking the final compressed blob and backing off if needed.
- The `LeakyReLU^2 + Bigram3072 + adaptive export` combination has not been jointly tuned on 8xH100 yet.
- This candidate only changes export-time precision allocation, so if the next bottleneck is training-time quantization robustness, a future follow-up may still need a truly working late-QAT phase.

## Validation

Ran:

```bash
python -m compileall candidates/202603271212_adaptive-mixedbit-gptq/train_gpt.py
```

Outcome:

- **Passed**.

Attempted an additional lightweight import/constructor smoke test, but the workflow environment did not have the repository's Python runtime deps installed (`ModuleNotFoundError: No module named 'numpy'`), so a deeper import-time smoke test was not feasible here.

I did **not** run a CUDA training smoke test in this environment.
