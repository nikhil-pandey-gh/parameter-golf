# LeakyReLU² + GPTQ-lite Int5-MLP + BigramHash8192

## Hypothesis

The strongest non-TTT stack in this repository already gets most of the way there with 11 layers, EMA, XSA, partial RoPE, shared value embeddings, and GPTQ-lite export. This candidate pushes on the part of that stack that still looks under-exploited: the MLP block.

The hypothesis is that three changes should stack well:

1. **LeakyReLU(0.5)^2** improves the 11-layer stack's train-time optimization by preserving negative-slope gradient flow inside the MLP.
2. **Mixed GPTQ-lite export with int5 MLP / int6 attention** should spend fewer artifact bits on the most compressible weights while keeping attention at the more precision-sensitive setting.
3. **BigramHash8192** reinvests some of the saved artifact headroom into larger lexical pair memory, which has already shown repeated gains in earlier records.

The target is lower **post-roundtrip sliding-window val_bpb** under the same 16MB artifact limit, without depending on legal TTT or parameter-banking infrastructure.

## Why this is promising for this repository

This repository's record history is dominated by three themes:

- small, reliable training deltas on a strong 11-layer stack,
- export-time quantization quality dominating final score,
- modest lexical augmentations like BigramHash continuing to pay off.

The 2026-03-23 record showed that **LeakyReLU^2 alone** was worth about `-0.002 BPB` on a stronger descendant stack. The 2026-03-20 `10L_Int5MLP_MuonWD04_SWA50` record showed that **int5 MLP + larger BigramHash** is a viable compression/capacity trade. The 2026-03-22 record showed that **GPTQ-lite clip search + EMA + warmdown3500** is a very strong base. This candidate combines those threads into one self-contained script.

## Prior repository experiments that informed it

### Chosen base

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

This is the main base because it is the strongest pre-TTT stack in the repo summary, with a stable 11-layer recipe and a clean standalone implementation.

### Additional influences

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the **LeakyReLU(0.5)^2** MLP activation.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - contributed the idea of **int5 MLP / int6 attention** plus a much larger BigramHash.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - reinforced that the 11-layer XSA/EMA/partial-RoPE line is the right family to keep building on.

There were no prior runs under `candidates/` when this candidate was created.

## External research that informed it

- **Scaling Law for Quantization-Aware Training** (Chen et al., 2025, arXiv:2505.14302)
  - argues that quantization error depends strongly on precision assignment and identifies FC/MLP outliers as a core bottleneck, motivating **mixed precision rather than uniform low-bit export**.
- **A Survey on Transformer Compression** (Tang et al., 2024, arXiv:2402.05964)
  - reinforces the broader design principle that quantization and efficient architectural changes should be co-designed rather than optimized independently.
- The repo's own later ablation evidence around **LeakyReLU^2** is treated as the most directly relevant primary empirical signal for this exact codebase.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **MLP activation**
   - changed from `relu(x)^2` to `leaky_relu(x, 0.5)^2`.

2. **Bigger lexical memory**
   - increased the default `BIGRAM_VOCAB_SIZE` from `2048` to `8192`.

3. **Mixed GPTQ-lite export**
   - kept the base record's multi-percentile row-clipped GPTQ-lite search idea,
   - but now quantizes:
     - MLP matrices to **int5** (`[-16, 15]`),
     - attention-style matrices to **int6** (`[-32, 31]`),
     - everything else as in the inherited mixed path.

4. **MLP-aware fake quantization**
   - MLP `fc` / `proj` layers are tagged to use the int5 clip range during inherited late-QAT.

Everything else stays intentionally close to the 2026-03-22 stack: 11L, XSA on last 4 layers, shared value embeddings, partial RoPE, LN scale, EMA, warmdown3500, seq 2048, sliding eval stride 64, and GPTQ-lite-style clip search.

## How to run

From this candidate directory:

```bash
RUN_ID=leaky_int5_bigram_8192 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves its default `DATA_PATH` and `TOKENIZER_PATH` from the repository root via `__file__`, so this command works from inside the candidate directory without extra path overrides.

The key defaults are baked into `train_gpt.py`. The main candidate-specific defaults are:

- `BIGRAM_VOCAB_SIZE=8192`
- `NUM_LAYERS=11`
- `TRAIN_SEQ_LEN=2048`
- `XSA_LAST_N=4`
- `LATE_QAT_THRESHOLD=0.15`
- `EVAL_STRIDE=64`

If artifact size ends up tighter than expected on a real H100 run, the first fallback to try is reducing `BIGRAM_VOCAB_SIZE` from `8192` to `4096` before changing the rest of the stack.

## Validation

### Commands run in this environment

Run from the repository root:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603280816_leaky-int5-bigram/train_gpt.py
python -m py_compile candidates/202603280816_leaky-int5-bigram/train_gpt.py
```

### Outcomes

- `compileall`: passed (`python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603280816_leaky-int5-bigram/train_gpt.py`).
- `py_compile`: passed (`python -m py_compile candidates/202603280816_leaky-int5-bigram/train_gpt.py`).
- Minimal CPU smoke test: not run. This candidate inherits the CUDA + `flash_attn_interface` runtime path and expects the challenge tokenizer/shards, so a meaningful CPU-only start test was not feasible in this environment.

## Expected risks and tradeoffs

- **Artifact-size risk**: `BigramHash8192` is intentionally larger; if the mixed int5/int6 savings are smaller than expected on this exact 11-layer stack, the model may need a smaller bigram table.
- **Training/export mismatch risk**: the inherited late-QAT path is now MLP-int5-aware, but this remains a fairly aggressive compression regime.
- **Incremental rather than architectural risk**: this candidate is a strong compositional bet, not a brand-new architecture. If it fails, the next step should probably be either compile-safe progressive QAT or a more aggressive evaluation-time adaptation variant.
