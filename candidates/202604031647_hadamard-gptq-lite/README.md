# Hadamard GPTQ-lite export on the 11L EMA/XSA stack

## Hypothesis

The strongest non-TTT stack in this repo is already training-side mature; the next cheap win is likely **post-training quantization quality**, not another large architecture rewrite. This candidate adds an **export-only Hadamard rotation search** on top of the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` base so each large int6 matrix can be quantized in the coordinate system that gives the lowest de-rotated roundtrip MSE.

The goal is to reduce the post-quantization BPB gap at essentially zero training-time cost and negligible artifact-size overhead.

## Why this is promising for this repository

- Repo history shows repeated gains from better compression-aware export rather than wholesale model redesign: sliding eval, fp16-sensitive tensors, mixed int6/int8, GPTQ-lite clip search, EMA, and tighter warmdown all produced meaningful wins.
- The `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` run is the strongest clean pre-TTT training stack, while `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` suggests the 11-layer family itself is already strong enough to justify focusing on the quantized artifact.
- Unlike learned rotations or new training objectives, an export-only rotation search fits this repo's constraints well: it preserves the training loop, keeps the candidate self-contained in one script, and only touches the serialization roundtrip already used by the records.

## Prior repo runs that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Architecture family:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Quantization sensitivity / mixed precision precedent:** `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
- **Best overall current record in the same family:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

There were **no prior `candidates/` experiments** in the repository when this candidate was created.

## External research that informed the idea

- **QuIP** ([arXiv:2307.13304](https://arxiv.org/abs/2307.13304)): quantization benefits from making weights more incoherent via orthogonal preprocessing.
- **QuaRot** ([arXiv:2404.00456](https://arxiv.org/abs/2404.00456)): fixed rotations can remove outlier concentration and make low-bit Transformer quantization much easier.
- **SpinQuant** ([arXiv:2405.16406](https://arxiv.org/abs/2405.16406)): the choice of rotation matters materially for quantized accuracy; better rotations can close a large part of the quantization gap.

This candidate uses the smallest repo-friendly adaptation of those ideas: a deterministic normalized Hadamard transform, tried only during export, with the best no-rotation / row-rotation / column-rotation choice selected per tensor by reconstruction MSE.

## What changed vs the chosen base

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. Added `HADAMARD_GPTQ` (default `1`) and `HADAMARD_MIN_DIM` (default `256`) knobs.
2. Added a normalized fast Walsh-Hadamard transform helper for power-of-two dimensions.
3. For each large int6-exported matrix, the script now tries:
   - no rotation,
   - right/column Hadamard preprocessing when the input width is power-of-two,
   - left/row Hadamard preprocessing when the output width is power-of-two.
4. Each candidate is quantized with the existing GPTQ-lite percentile clip search, de-rotated back to the original basis, and scored by roundtrip MSE; the best candidate is kept.
5. The dequantization path now applies the inverse Hadamard transform when a rotated encoding was chosen.
6. Export logs now include a `gptq_rotations:` summary showing how many int6 tensors used each path.

Training, architecture, EMA/SWA, XSA, Partial RoPE, VE, and evaluation remain otherwise unchanged.

## How to run

From this candidate directory:

```bash
cd candidates/202604031647_hadamard-gptq-lite
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
HADAMARD_GPTQ=1 HADAMARD_MIN_DIM=256 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This candidate resolves its default `DATA_PATH` and `TOKENIZER_PATH` from the repository root, so the command works when launched directly from the candidate directory.

If the rotation search is neutral or harmful on a particular system, set `HADAMARD_GPTQ=0` to recover the plain GPTQ-lite export path.

## Main expected risks and tradeoffs

- The repo already uses per-row GPTQ-lite clipping, so rotation gains may be smaller here than in lower-bit or activation-quantized papers.
- Fixed Hadamard rotations are less expressive than learned rotations (SpinQuant), so some tensors may still prefer the unrotated basis.
- Export/dequant roundtrip is a little more CPU-expensive because each large int6 tensor may evaluate multiple candidates.
- This is intentionally an export-side candidate; if the real bottleneck is training-time quantization robustness instead of export basis choice, gains may be limited.

## Validation

Commands run locally in this workflow:

```bash
python -m compileall candidates/202604031647_hadamard-gptq-lite/train_gpt.py

python - <<'PY'
import glob, importlib.util
for name in ('torch', 'flash_attn_interface', 'sentencepiece'):
    print(name, bool(importlib.util.find_spec(name)))
print('train_shards', len(glob.glob('data/datasets/fineweb10B_sp1024/fineweb_train_*.bin')))
print('val_shards', len(glob.glob('data/datasets/fineweb10B_sp1024/fineweb_val_*.bin')))
PY
```

Outcomes:

- `python -m compileall .../train_gpt.py` **succeeded**.
- End-to-end CPU smoke was **not feasible in this runner** because the local Python environment did not have `torch`, `flash_attn_interface`, or `sentencepiece`, and there were no local FineWeb train/validation shards under `data/datasets/fineweb10B_sp1024/`.
