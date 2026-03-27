# QuaRot-lite export on the 2026-03-22 EMA + GPTQ-lite stack

## Hypothesis

The strongest open lever in this repository is still the quantization gap on the already-strong 11-layer stack. This candidate tests a **QuaRot-lite** export path: before int6 quantization, eligible 2D attention and MLP matrices are compared in both their raw form and a block-Hadamard-rotated form, and the lower-MSE encoding is kept. The model function is unchanged after load-time de-rotation, so any win should come from better compressibility / lower roundtrip error rather than training-side luck.

I also carry over the **LeakyReLU(0.5)^2** MLP activation from the current best overall record because it was one of the clearest positive ablations in the repo and is orthogonal to the export-side idea.

## Why this is promising for this repository

The record progression shows that the repo's best gains now come from increasingly compression-aware design:

- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` showed that architecture refinements like partial RoPE and LN scaling still help, but only in basis points.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` got its gain mainly from smarter export and smoother weights.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` showed LeakyReLU(0.5)^2 was a real win on top of a strong stack.

The records review also suggests there were **no prior `candidates/` directories yet**, so this is meant to be a first serious candidate rather than a rerun of an abandoned line.

## Prior records that influenced this candidate

The base implementation is the 2026-03-22 non-TTT frontier stack:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Specific influences:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - strongest pure train/export stack without legal TTT
  - already has the right ingredients for a quantization-focused follow-up: EMA, GPTQ-lite clip search, partial RoPE, XSA4, VE, BigramHash, and the 11L / 3x MLP layout
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - contributed the `LeakyReLU(0.5)^2` MLP activation
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - reinforced that export-aware tweaks are now competing in a narrow margin regime and need to be low-risk and orthogonal
- `2026-03-19_smeargate_orthoinit_muonwd`
  - the original breakthrough showing that quantization-aware stacks can dominate brute-force compute

## External research that informed it

This candidate is primarily motivated by function-preserving quantization preconditioning:

- **QuaRot**: <https://arxiv.org/abs/2404.00456>
  - shows that orthogonal rotations can remove outliers and make LLM quantization easier without changing the represented function
- **SmoothQuant**: <https://arxiv.org/abs/2211.10438>
  - frames the core idea of mathematically equivalent transformations to shift quantization difficulty into a friendlier representation
- **Data-Free Quantization through Weight Equalization and Bias Correction**: <https://arxiv.org/abs/1906.04721>
  - older but useful evidence that scale-/transform-preserving reparameterizations can materially improve PTQ without retraining

This implementation is intentionally lighter than those papers: it only tests a deterministic block-Hadamard transform on eligible attention/MLP matrices during export, keeps whichever domain gives lower reconstruction MSE, and stores enough metadata to invert the transform after load.

## What changed versus the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

- Added **block-Hadamard rotation-aware PTQ** for large 2D int6 attention/MLP tensors.
  - export compares raw int6 GPTQ-lite style quantization against a rotated-domain version
  - dequantization inverts the rotation before loading the model back for evaluation
  - env knobs: `ROTATE_PTQ`, `ROTATE_PTQ_BLOCK`, `ROTATE_PTQ_MIN_DIM`
- Switched the MLP activation to **`LeakyReLU(0.5)^2`**.
  - env knob: `MLP_LEAKY_SLOPE`
- Made the script more self-contained for local validation.
  - file-relative defaults for `DATA_PATH` / `TOKENIZER_PATH`
  - fallback from `flash_attn_interface` to `torch.nn.functional.scaled_dot_product_attention`
  - CPU-safe smoke path for tiny local runs

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603272312_quarot-lite

SEED=1337 \
ROTATE_PTQ=1 ROTATE_PTQ_BLOCK=256 ROTATE_PTQ_MIN_DIM=256 \
MLP_LEAKY_SLOPE=0.5 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- By default the script resolves data paths relative to the repository root, so running it directly from the candidate directory should still find `../../data/...` automatically.
- For a pure ablation against the base stack, disable the new export path with `ROTATE_PTQ=0` or revert to ReLU^2 with `MLP_LEAKY_SLOPE=0.0`.

## Expected risks and tradeoffs

- The biggest uncertainty is whether **6-bit** quantization is still high enough precision that rotation buys only a tiny gain; the strongest literature gains are usually reported at 4-bit.
- Block size matters. `256` is a conservative default because it cleanly tiles the main 512 / 256 / 1536 dimensions in this stack, but a different block size may work better.
- The current rotation search is intentionally lightweight and reconstruction-MSE-based, not calibration-loss-based. That keeps the implementation minimal but may leave some gains on the table.
- Carrying over LeakyReLU^2 helps make this candidate stronger, but it also means any final improvement would need to be disentangled from the export-side win with ablations.

## Validation

Commands run locally:

```bash
cd candidates/202603272312_quarot-lite
python -m compileall train_gpt.py
```

Outcome:

- `python -m compileall train_gpt.py` passed.
- A CPU smoke run was **attempted but not completed** because this runner currently lacks the repository runtime dependencies (`torch`, `numpy`, and `sentencepiece`) in both `python` and `python3`, and it also does not contain the cached challenge dataset / tokenizer under `data/`.
- Because of those environment limits, the validation here is syntax-level only. The script was still written with a CPU-safe smoke path and file-relative defaults so that a follow-up local run is straightforward once the normal repo dependencies and data are present.
