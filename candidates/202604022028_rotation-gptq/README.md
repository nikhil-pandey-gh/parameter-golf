# Rotation-GPTQ-lite export on the 2026-03-23 stack

## Hypothesis

The current best stack is already quantization-sensitive: most of the big gains in this repo came from better compression-aware training, better eval, and better post-training quantization rather than from raw pre-quant loss alone. A lightweight **rotation-aided export** should reduce the int6 roundtrip error on the largest attention and MLP matrices by smoothing column outliers before row-wise GPTQ-lite quantization, while keeping the trained network and inference graph unchanged after dequantization.

## Why it is promising for this repository

- The repo is evaluated after loading and dequantizing the stored artifact, so we can change the **storage basis** without changing the trained function.
- The top records already converged on a strong 11-layer training recipe; the remaining headroom is increasingly in **post-training quantization quality**.
- Prior records show repeated wins from fp16 embeddings, mixed int6/int8 export, GPTQ-lite clip search, and quantization-friendly training. This candidate attacks that same bottleneck, but in a way that does **not** require new training infrastructure.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current full stack
  - supplies the candidate base script, LeakyReLU^2 MLP, legal TTT path, and parameter-banked optimizer path
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best clear non-TTT fork point for quantization work
  - introduced GPTQ-lite percentile search and quantified its benefit
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - useful warning that the earlier late-QAT toggle was effectively a no-op under `torch.compile`
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - strong evidence that this challenge is often won or lost at export time

There were **no prior `candidates/` directories** in this checkout when this run started.

## External research that informed it

- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs**  
  https://arxiv.org/abs/2404.00456
- **SpinQuant: LLM Quantization with Learned Rotations**  
  https://arxiv.org/abs/2405.16406

Both papers use orthogonal rotations to make low-bit quantization easier by redistributing outlier structure. This candidate applies the same core idea in a narrower repo-friendly form: only inside the export/dequant path, and only when the rotated basis actually lowers reconstruction error for a given tensor.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in `train_gpt.py`:

1. **Rotation-aware int6 export**
   - For eligible 2D attention/MLP matrices whose input dimension is divisible by 512, the exporter now compares:
     - plain GPTQ-lite row-wise int6 quantization
     - GPTQ-lite row-wise int6 quantization after a normalized 512-wide block Hadamard rotation on the column axis
   - The candidate stores whichever version reconstructs the original tensor with lower MSE.
2. **Inverse rotation on load**
   - Rotated tensors are dequantized and then mapped back to the original basis before the eval model is built.
3. **Repo-root-relative defaults**
   - Default dataset and tokenizer paths resolve relative to the repository root, so the script can be run from inside this candidate directory.
4. **Import-safe top section**
   - `flash_attn_interface` import is now optional at import time so helper functions can be inspected in non-GPU environments; the actual model path still requires FlashAttention.
5. **Late QAT disabled by default**
   - `LATE_QAT_THRESHOLD` defaults to `0.0` here so this candidate isolates the export experiment instead of depending on the previously unreliable late-QAT toggle.

## How to run or evaluate

From this directory:

```bash
ROTATION_INT6=1 \
ROTATION_BLOCK_SIZE=512 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.0 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `DATA_PATH` and `TOKENIZER_PATH` default to the repo-root `data/` tree, so they do not need to be set when running from this candidate directory.
- `ROTATION_INT6=0` is the easy ablation.

## Validation run for this candidate

Commands run in this environment:

```bash
python -m compileall candidates/202604022028_rotation-gptq/train_gpt.py
python3 - <<'PY'
import sys
print(sys.version)
try:
    import torch
    print(torch.__version__)
except Exception as exc:
    print(type(exc).__name__, exc)
PY
```

Outcomes:

- `python -m compileall ...` **passed**
- A deeper helper smoke test was **not feasible here** because the available Python runtime in this container does not have `torch` installed
- A full CPU run is also **not feasible** for this candidate as written because the training/eval path requires CUDA plus FlashAttention

## Main expected risks and tradeoffs

- The Hadamard rotation may help some matrices and hurt others; this is why the exporter uses **per-tensor error selection** instead of forcing rotation everywhere.
- The current implementation only rotates tensors whose input dimension is divisible by 512; this deliberately avoids padding logic and keeps the change small.
- The improvement may be modest if GPTQ-lite clip search already captures most of the available quantization headroom.
- Because this candidate focuses on export, not training, it may improve roundtrip fidelity more than raw pre-quant loss.
