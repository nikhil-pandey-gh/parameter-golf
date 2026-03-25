# QuaRot-lite + LeakyReLU2 on the 11L EMA/GPTQ-lite stack

## Hypothesis

The recent record trend in this repository says the remaining gains mostly come from reducing post-training export error rather than inventing a completely new backbone. This candidate tests a small, repository-friendly version of rotation-based quantization: apply deterministic signed block-Hadamard rotations to quantized 2D tensors before GPTQ-lite clip search, then undo the rotation after dequantization.

The expectation is that the rotations spread column outliers across each row, making the existing per-row int6/int8 quantizers more accurate without adding learned parameters or artifact bytes. I also carry over `LeakyReLU(0.5)^2` from the current best record because it is a one-line, already-supported training improvement.

## Why this is promising here

The strongest non-TTT record in this repo is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which improved mainly by polishing export quality rather than changing the 11-layer architecture. The next record on 2026-03-23 improved further with `LeakyReLU(0.5)^2` and legal TTT, again reinforcing that the stack is mature and small quality deltas matter.

Rotation-based PTQ is attractive for this challenge because it attacks the same bottleneck as GPTQ-lite while staying cheap in code size and training cost. Unlike learned rotations, a fixed signed Hadamard transform is deterministic, storage-free, and easy to fit into the current export path.

## Prior repository work that influenced this candidate

The main local influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` for the 11L EMA + GPTQ-lite export stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for the observation that the real win was Partial RoPE + LN scale, while late QAT was effectively dead due to compile-time specialization.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` for the `LeakyReLU(0.5)^2` MLP activation.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/` for the strong evidence that quantization/export fidelity, especially for sensitive matrices, is the central bottleneck.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` for the reminder that depth reuse / recurrence is likely a dead end under the fixed 10-minute budget.

## External research that informed it

The main papers behind the choice were:

- `GPTQ` (`arXiv:2210.17323`), which established one-shot second-order-aware weight quantization as a strong baseline for transformer PTQ.
- `QuaRot` (`arXiv:2404.00456`), which shows that orthogonal rotations can remove outliers and make low-bit quantization much easier while preserving the model function.
- `SpinQuant` (`arXiv:2405.16406`), which finds that some rotations are much better than others and that rotation quality materially affects low-bit accuracy.
- `The case for 4-bit precision` (`arXiv:2212.09720`), which argues that the bit-allocation tradeoff matters as much as raw parameter count, reinforcing why export quality is worth optimizing in a fixed artifact budget challenge.

This candidate deliberately chooses the smallest adaptation of that literature that fits this repo: deterministic signed Hadamard rotations inside the existing GPTQ-lite-style exporter, not a new learned-rotation training pipeline.

## What changed versus the chosen base implementation

Chosen base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`.

Changes in this candidate:

- Added deterministic signed block-Hadamard rotation before quantizing large 2D tensors.
- Stored only tiny rotation metadata (`rotate=hadamard`, `block_size`) and regenerated the sign pattern from the tensor name at load time, so the rotation adds effectively no model bytes.
- Applied the rotation-aware path to both the int6 matrix path and the int8 fallback path used for non-int6 quantized tensors.
- Swapped the MLP activation to `LeakyReLU(0.5)^2`.
- Added an optional CPU + SDPA fallback and optional compile disable flag so the script can be smoke-tested without FlashAttention 3 or GPUs.
- Disabled late-QAT by default (`LATE_QAT_THRESHOLD=0.0`) because the 2026-03-21 record documented that the dynamic enable path was constant-folded away under `torch.compile`; this candidate focuses on the export-side idea instead of relying on that brittle path.

## How to run or evaluate it

From this candidate directory:

```bash
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
ROTATE_QUANT=1 ROTATE_MIN_BLOCK=64 \
LATE_QAT_THRESHOLD=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

- `ROTATE_QUANT=0` to isolate the new export path.
- `ENABLE_COMPILE=0` if you want to experiment with dynamic QAT logic locally.
- `ROTATE_MIN_BLOCK=128` or `256` if the embedding path proves too sensitive to smaller blockwise rotations.

## Main expected risks or tradeoffs

- The idea is strongest when export error, not raw fp32 quality, is the limiting factor. If the current stack is already close to the best quantization frontier, the gain may be small.
- Signed Hadamard rotations are cheaper than learned rotations, but `SpinQuant` suggests that rotation choice matters. A deterministic pseudo-random sign pattern may help less than a learned rotation.
- Applying the rotation to tied embeddings may or may not be net-positive; this repo has repeatedly found embeddings unusually sensitive.
- The CPU/SDPA fallback is only for smoke validation. The intended competitive path is still the CUDA + FlashAttention stack.

## Validation

Planned / executed validation commands for this candidate:

```bash
python -m compileall candidates/202603251657_quarot-lite/train_gpt.py
```

Outcome: **passed** in this workflow runner.

```bash
python3 - <<'PY'
mods = ['numpy', 'sentencepiece', 'torch']
for m in mods:
    __import__(m)
PY
```

Outcome: **failed in this workflow runner** because the local Python environment did not have `numpy`, `sentencepiece`, or `torch` installed, so a real CPU launch could not be completed here without the normal repository runtime environment.

Because of that dependency gap, I could not complete the temporary toy-tokenizer/toy-shard smoke run in this workflow. On a normal Parameter Golf runtime image with the listed repo dependencies installed, the candidate should be smoke-tested with a tiny local SentencePiece model and synthetic shard pair before any real GPU run.
