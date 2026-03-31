# FC2 Kurtosis Late-QAT

## Hypothesis

The best reusable non-TTT stack in this repository is already strongly compression-aware, but it still leaves a gap between the EMA checkpoint and the final GPTQ-lite int6 artifact. My hypothesis is that a **late-only, compile-safe QAT path plus a light FC2/MLP outlier regularizer on the deepest blocks** can reduce that export gap without paying for a broader architecture rewrite.

In practice, the candidate tries to make the model **easier to quantize**, not just better to quantize afterward.

## Why this is promising for this repository

The repo history shows a very clear pattern:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` established the durable 11-layer, EMA, XSA, int6-export family.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` improved the same family with partial RoPE and layerwise scaling, but its README also notes that its late-QAT branch was effectively compiled away.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` is the cleanest strong base: GPTQ-lite percentile search, EMA, warmdown 3500, value embeddings, and a stable 3-seed mean.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` shows the next big gain came from a more complex eval-heavy stack, not from replacing the core 11-layer recipe.

That made the `2026-03-22` stack the best base for a surgical candidate: it is strong, stable, and still simple enough to modify precisely.

## Prior repo influence

This candidate is mainly derived from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`

It was also informed by a negative-result trail:

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` reported that naive layer recurrence was actively harmful in a short-wallclock regime, so I deliberately avoided architecture-level weight tying/recurrence here.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

- **Nrusimha et al., 2024 — _Mitigating the Impact of Outlier Channels for Language Model Quantization with Activation Regularization_** (`arXiv:2404.03605`)  
  Key takeaway: outlier channels make low-bit quantization harder, and activation regularization can stop that difficulty from migrating into the weights and hurting PTQ.

- **Chen et al., 2025 — _Scaling Law for Quantization-Aware Training_** (`arXiv:2505.14302`)  
  Key takeaway: FC2-layer outliers are a major QAT bottleneck, and reducing weight quantization error remains important even when activation-side issues are addressed.

- **Xiao et al., 2025 — _Exploring Layer-wise Information Effectiveness for Post-Training Quantization in Small Language Models_** (`arXiv:2508.03332`)  
  Key takeaway: quantization sensitivity is not uniform across layers, which supports focusing this regularizer on the deepest few blocks instead of spreading it everywhere.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Compile-safe late QAT**
   - The base family relied on toggling a class attribute from the training loop.
   - This candidate threads `qat_active` through the compiled forward path directly, so the QAT branch is controlled by runtime inputs instead of a Python-side class flag.
   - That is specifically meant to avoid the `torch.compile` constant-folding failure mode called out in the `2026-03-21` README.

2. **Late FC2/MLP kurtosis regularization**
   - The MLP now computes a small centered-kurtosis penalty on the **pre-activation feeding the FC2/projection path**.
   - The penalty is only applied to the **last `OUTLIER_REG_LAST_N` blocks** and only once the training schedule reaches the configured late threshold.
   - To keep the overhead bounded at the repository's default batch size, the statistic is computed on a **sampled token view** rather than the full late-stage activation tensor.
   - Defaults:
      - `OUTLIER_REG_LAST_N=4`
      - `OUTLIER_REG_THRESHOLD=0.15`
      - `OUTLIER_REG_WEIGHT=1e-4`
      - `OUTLIER_REG_TARGET=4.0`
      - `OUTLIER_REG_TOKEN_STRIDE=16`
      - `OUTLIER_REG_MAX_TOKENS=4096`

3. **Candidate-directory portability**
   - The default dataset and tokenizer paths now resolve relative to `train_gpt.py` itself, so the script can be launched from inside this candidate directory without manually rewriting `DATA_PATH` / `TOKENIZER_PATH`.

Everything else intentionally stays close to the base stack:

- 11 layers, 512 width, 8H / 4KV
- MLP3x
- XSA on the last 4 layers
- Partial RoPE
- LN scale
- SmearGate + BigramHash
- Value embeddings on deep layers
- EMA
- GPTQ-lite int6 export + zstd

## How to run

From the repository root:

```bash
cd candidates/202603311127_fc2-kurtosis-qat

RUN_ID=fc2_kurtosis_qat \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
LATE_QAT_THRESHOLD=0.15 \
OUTLIER_REG_THRESHOLD=0.15 OUTLIER_REG_WEIGHT=1e-4 OUTLIER_REG_TARGET=4.0 OUTLIER_REG_LAST_N=4 \
OUTLIER_REG_TOKEN_STRIDE=16 OUTLIER_REG_MAX_TOKENS=4096 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If the repository layout is unchanged, the script will automatically find:

- `../../data/datasets/fineweb10B_sp1024/`
- `../../data/tokenizers/fineweb_1024_bpe.model`

because it resolves defaults relative to the candidate file location.

## Evaluation notes

The export path is unchanged from the chosen base:

- GPTQ-lite-style int6 per-row quantization for attention/MLP weights
- gentler handling for embeddings / small tensors
- zstd-compressed artifact
- full roundtrip eval plus sliding-window eval

This means any win from this candidate should come from a smaller **train-to-quantized** degradation, not from a new compression format.

## Validation

Commands run for this candidate in the current workflow:

- `python -m compileall candidates/202603311127_fc2-kurtosis-qat/train_gpt.py`  
  Outcome: **passed**

- Attempted a minimal CPU smoke test by importing the candidate with a local FlashAttention stub and exercising `forward(..., qat_active=True, outlier_reg_active=True)`  
  Outcome: **not feasible in this runner** because the local Python environment does not have `torch` installed, and installing heavyweight new runtime dependencies was outside the safe lightweight-validation scope here.

## Main risks and tradeoffs

- The kurtosis penalty may still over-smooth useful high-magnitude channels and slightly hurt the pre-quant model if the coefficient is too large.
- Passing runtime QAT/regularizer flags through the compiled graph may trigger an additional compile path once the late phase begins.
- Because the export format is unchanged, this candidate is betting entirely on **better quantization robustness**, not on a better compressor or eval trick.
- The new defaults are intentionally conservative; the likely next tuning surface is the `(weight, target, last_n, threshold, token_stride, max_tokens)` set for the outlier penalty.
