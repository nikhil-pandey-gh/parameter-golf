# Candidate: Sandwich-Shared MLP Banks

## Hypothesis

The strongest non-TTT line in this repository already looks like a compression-aware 11-layer Transformer with 3x MLPs, bigram features, SmearGate, XSA in the deepest layers, partial RoPE, EMA, and GPTQ-lite export. My hypothesis is that the **middle feed-forward blocks are over-parameterized relative to the artifact budget**, so we can **share only those middle MLPs**, keep every attention block unique, and spend the recovered bytes on a **higher-precision export path for the shared banks**.

This is intentionally **not** a recurrent-depth candidate. The local non-record 1x5090 sweep found that doubling depth through layer recurrence was strongly negative because it cut step count too much in a fixed wall-clock budget. This candidate instead keeps the same number of forward passes and the same 11-layer schedule, so throughput should stay close to the proven XSA/EMA/GPTQ-lite stack.

## Why this is promising for this repository

Three local patterns motivated this candidate:

1. The best compact runs are now dominated by **quantization-aware architecture tuning**, not by large model-family swaps.
2. The best non-TTT stack already relies on **late-layer attention specialization** (`XSA_LAST_N=4`, partial RoPE, LN scale), so sharing full blocks would be higher risk than sharing only the MLP path.
3. The repo has negative evidence against **extra recurrent depth**, but not against **parameter sharing at fixed compute**.

So this candidate makes the smallest structural bet that still feels genuinely new in this codebase: **share only the middle MLP banks, keep unique attention everywhere, and keep the rest of the winning training/export recipe intact**.

## Prior records that influenced this candidate

### Main base implementation

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

This is the direct base for `train_gpt.py`. It contributed the 11-layer 512-dim stack, XSA on the deepest layers, EMA-before-export, GPTQ-lite percentile clip search, warmdown3500, bigram features, SmearGate, and the mixed int6/int8 export path.

### Closely related architectural wins

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`

These records established the exact lane this candidate is trying to preserve: **11 layers, 3x MLP, deep-layer-only attention specialization, low-overhead improvements, and strong compression-aware export**.

### Negative evidence reused here

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`

That sweep explicitly found **layer recurrence x2** to be a bad trade because extra recurrent depth hurt step count too much under a fixed training-time budget. That is why this candidate uses **parameter sharing without extra unrolled passes**.

## External research that informed the idea

- **Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers** (`arXiv:2101.00234`)

  This is the most direct inspiration. The paper argues that **naive cross-layer sharing is weak in generative Transformers**, but **sandwich-style sharing** can outperform a standard Transformer with fewer parameters.

- **ALBERT** (`arXiv:1909.11942`)

  ALBERT is the classic evidence that cross-layer sharing can meaningfully improve parameter efficiency. This candidate borrows that spirit, but in a much more conservative form aimed at autoregressive compact models.

- **Universal Transformer** (`arXiv:1807.03819`) and **Intra-Layer Recurrence in Transformers for Language Modeling** (`arXiv:2505.01855`)

  These papers support the broader idea that recurrence or reuse can be helpful, but they also imply that *where* reuse happens matters. In this repo, compute is tightly capped, so the implementation here favors **reuse without more passes**.

- **A Survey on Transformer Compression** (`arXiv:2402.05964`)

  This was useful as a broader reminder that the best compact models often come from **co-designing architecture and quantization**, not treating them as separate problems.

## What changed versus the chosen base

Starting from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes the following changes:

1. **Repo-root-relative defaults**

   The script now resolves dataset and tokenizer defaults relative to the repository root, so it can be launched directly from the candidate directory without manually rewriting paths.

2. **Sandwich-style shared middle MLPs**

   A new `SHARED_MLP_GROUPS` setting defaults to:

   ```bash
   SHARED_MLP_GROUPS=2-3,4-5,6-7
   ```

   That means layers `2-3`, `4-5`, and `6-7` each reuse one shared MLP bank, while the first two layers and last three layers keep unique MLP weights.

   Important detail: only the **MLP matrices** are shared. Each layer still keeps its own:

   - attention projections,
   - RMSNorms,
   - residual mixing,
   - learned MLP/attention scaling,
   - skip-connection placement,
   - late-layer XSA behavior.

   In other words, the candidate shares the most byte-heavy part of the middle trunk while preserving the per-layer attention specialization that seems important in the record line.

3. **Shared banks are serialized once and exported at int8**

   Shared MLP banks live under `shared_mlps.*` instead of under `blocks.*.mlp.*`, so they appear only once in the model state dict.

   The export classifier marks `shared_mlps.*` as a separate category, which means they follow the **int8 path** instead of the default int6 MLP path. The goal is to convert parameter savings from sharing into **lower roundtrip quantization error** on the reused banks.

4. **Late QAT is disabled by default**

   `LATE_QAT_THRESHOLD` now defaults to `0.0`. Prior repository notes showed that compile-time constant folding can make late-QAT paths brittle in fully compiled runs, and this candidate is meant to test the sharing/precision-allocation hypothesis as cleanly as possible.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603311258_sandwich-shared-mlp
RUN_ID=sandwich_shared_mlp \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If your dataset or tokenizer lives somewhere else, override the defaults explicitly:

```bash
cd candidates/202603311258_sandwich-shared-mlp
RUN_ID=sandwich_shared_mlp \
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To ablate the idea, you can disable sharing entirely:

```bash
SHARED_MLP_GROUPS= \
RUN_ID=no_shared_mlp \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- **Over-regularization risk**: the middle FFNs may still need per-layer specialization, so sharing them could hurt pre-quant loss even if it helps the artifact budget.
- **Quantization-allocation risk**: exporting the shared banks at int8 might not pay back enough bpb if the real bottleneck is elsewhere in the model.
- **Interaction risk with XSA/skip topology**: even though attention remains unique, changing the capacity of the middle feed-forward path may alter how the U-Net-style skip structure is used.
- **Compile/QAT caveat**: late QAT is intentionally out of scope for this candidate by default; if re-enabled, it should be audited carefully in compiled mode.

## Validation

### Commands run

```bash
python -m compileall candidates/202603311258_sandwich-shared-mlp/train_gpt.py
python - <<'PY'
try:
    import torch
    print('torch:available')
except Exception as exc:
    print(f'torch:missing:{type(exc).__name__}:{exc}')
PY
```

### Outcomes

- `python -m compileall ...` **passed**.
- A minimal CPU smoke test was **not feasible in this workflow runner** because `torch` is not installed here (`ModuleNotFoundError: No module named 'torch'`).
- I also ran a baseline repository syntax check before editing:

  ```bash
  python -m compileall train_gpt.py train_gpt_mlx.py data
  ```

  That baseline syntax check also passed.
