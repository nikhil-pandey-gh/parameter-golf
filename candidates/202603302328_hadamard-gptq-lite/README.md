# Hadamard-GPTQ-lite on top of LeakyReLU2 + legal TTT + Parallel Muon

## Hypothesis

The current frontier in this repository is strongly quantization-limited: exporter improvements keep buying measurable BPB even when the training stack is already strong. This candidate tests whether a fixed block-wise Hadamard rotation can make the existing per-row int6 GPTQ-lite exporter more robust by flattening row outliers before clip selection, then undoing the rotation after dequantization.

Because the Hadamard transform is orthogonal and self-inverse, this changes only the quantization basis, not the represented float model. The expected upside is lower reconstruction error for the same int6 payload and no training-time slowdown.

## Why this is promising for this repository

Repository history points to quantization as the main remaining bottleneck:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` improved the best static stack by about `-0.0013` BPB largely from better clip search and EMA, with essentially zero extra training cost.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the strongest overall result here (`1.1194` mean) and still exports the model through the same mixed int6 path, so exporter-side gains should stack cleanly with its training recipe.
- The broader record trend is consistent: mixed low-bit export, protecting sensitive tensors, and better compression-aware postprocessing beat many larger architectural changes.

This candidate therefore targets the place where the repo has been winning most reliably lately: getting a better artifact out of an already strong training run.

## Prior records that influenced this candidate

- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: chosen as the direct base because it is the strongest current record and already bundles the best known training-side stack in this repo.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: showed that export-only GPTQ-lite improvements can be worth a real BPB gain.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: reinforced that the best static architecture is already quite mature, so the next incremental win should probably be export- or evaluation-side rather than a broad model rewrite.
- Full history review across `records/` also suggested avoiding dead ends like recurrent layer reuse, which was explicitly bad in `track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`.

No prior `candidates/` directory existed when this candidate was created.

## External research that informed it

- **QuaRot** (arXiv:2404.00456): showed that orthogonal rotations can remove outliers and substantially improve low-bit transformer quantization while preserving the original computation.
- **SpinQuant** (arXiv:2405.16406): strengthened the same thesis, showing that rotation choice materially affects quantization quality and that rotation-based PTQ can outperform earlier baselines on hard-to-quantize LLMs.
- **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference** (arXiv:2004.09602): useful background for why clipping and scale choice matter so much in low-bit deployments.

I also briefly reviewed broader compact-model directions such as matmul-free transformers, but those would require much broader architectural changes than this repository's recent winning pattern justifies for a next incremental candidate.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. The script now defaults `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root so it can be launched directly from this candidate directory.
2. Added two exporter knobs:
   - `HADAMARD_GPTQ=1` to enable the rotated int6 path
   - `HADAMARD_BLOCK_SIZE=128` to control the block size
3. Before int6 GPTQ-lite clip search, eligible 2D int6 matrices are rotated with a fixed normalized block Hadamard transform along the input dimension.
4. The transform metadata is stored in the mixed-int6 export manifest.
5. During dequantization, the same Hadamard transform is applied again to invert the rotation before loading weights back into the eval model.

Importantly, embeddings and passthrough control tensors are left alone. This candidate only changes the mixed int6 exporter path for the large attention/MLP matrices.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603302328_hadamard-gptq-lite
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 HADAMARD_GPTQ=1 HADAMARD_BLOCK_SIZE=128 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablation:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 HADAMARD_GPTQ=0 HADAMARD_BLOCK_SIZE=128 \
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- Rotation may reduce quantization error but hurt `lzma` compressibility enough to erase the gain.
- The current implementation only rotates tensors whose last dimension is divisible by `HADAMARD_BLOCK_SIZE`; other tensors fall back to the original path.
- This adds extra CPU-side export and dequantization work, though no training-time cost.
- Because the base stack still includes legal TTT, total evaluation time remains substantial even if the exporter improves.

## Validation run for this candidate

Succeeded:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603302328_hadamard-gptq-lite/train_gpt.py
```

Attempted but not feasible in this workflow environment:

```bash
cd candidates/202603302328_hadamard-gptq-lite && python - <<'PY'
# attempted import-level CPU smoke test with a stubbed flash_attn_interface
PY
```

That smoke test failed immediately with `ModuleNotFoundError: No module named 'torch'` because the session Python environment does not have the repo's runtime dependency stack installed. So this candidate is syntax-validated here, but a real import/CPU forward smoke test still needs an environment with `torch` available.
