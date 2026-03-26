# Learned Bank-QAT Clip Multipliers

## Hypothesis

The current best stack in this repository is already strong on architecture and evaluation, but its main weight banks are still quantized only after training. Earlier repository results showed that STE int6 QAT can largely erase the quantization gap, while a later 11-layer record explicitly documented that its `late_qat` path was accidentally dead under `torch.compile`. This candidate tests a focused fix: apply compile-safe late QAT directly to the banked weights, and let each bank slice learn a small clip multiplier that is reused by the export quantizer.

## Why this is promising for this repository

Three repository trends point in the same direction:

1. `records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/README.md` showed that STE int6 QAT can eliminate the quant gap on strong int6 runs.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` noted that its late-QAT branch was dead-code-eliminated by `torch.compile`, so late QAT never actually hit the main model.
3. The best current stack, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`, still relies on GPTQ-lite export over parameter banks, so quantization robustness on those banks remains a plausible source of headroom.

That makes bank-aware late QAT a high-leverage, low-infrastructure follow-up: it targets the exact place where most artifact bytes live without disturbing the winning XSA / Partial RoPE / VE / TTT structure.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Current best repository result.
  - Supplies the banked optimizer path, LeakyReLU² MLP, legal score-first TTT, and the latest export/eval stack.
- **Quantization motivation:** `records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR/`
  - Strong evidence that STE int6 QAT can remove post-quant degradation.
- **Failure mode being fixed:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Important warning that late QAT can silently do nothing in compiled graphs.
- **Clean pre-TTT 11-layer lineage:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Reinforces that GPTQ-lite export is the right place to wire quantization-aware improvements.

There were no prior experiments under `candidates/` when this candidate was created.

## External research that informed it

- **AWQ** (`arXiv:2306.00978`) argues that not all weights are equally quantization-sensitive and that protecting the important parts of the weight distribution matters disproportionately.
- **SmoothQuant** (`arXiv:2211.10438`) highlights that quantization difficulty is uneven and can often be managed by changing how scale is assigned.
- **LLM.int8()** (`arXiv:2208.07339`) shows that transformer quality is often dominated by a small set of outlier-sensitive directions.
- **LSQ** (`arXiv:1902.08153`) motivates learning quantizer scale parameters jointly with model weights instead of treating them as fixed post-hoc constants.

This candidate does not implement those methods literally. Instead, it borrows a lightweight version of their shared idea: learn small quantizer-scale adjustments on the most important matrices, then reuse them at export time.

## What changed versus the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate adds:

1. **Candidate-directory-safe defaults**
   - Dataset and tokenizer defaults now resolve relative to the repository root, so running from this candidate directory works without extra path edits.

2. **Compile-safe bank late QAT**
   - Added `_ste_fake_quantize_int6_per_row(...)` for the main bank slices.
   - The model keeps bank QAT fully off during early training, then activates it once the warmdown factor falls below `LATE_QAT_THRESHOLD`.
   - At that activation point, the script recompiles the model once and thereafter uses a runtime `bank_qat_mix` tensor to ramp the fake-quant blend upward during the rest of late training.

3. **Learned per-slice clip multipliers**
   - Added four small parameter vectors:
     - `qo_qat_log_scale`
     - `kv_qat_log_scale`
     - `mlp_up_qat_log_scale`
     - `mlp_down_qat_log_scale`
   - Each bank slice learns a scalar multiplier (clamped exponential) that rescales its int6 clipping range.
   - These learned clip tensors are optimized with the scalar/control parameter group and then reused by the export quantizer.

4. **Export quantizer wiring**
   - `mixed_quantize_int6(...)` now accepts a `clip_multipliers` map.
   - The learned clip multipliers are exported from the trained model and applied during the final GPTQ-lite-style int6 quantization pass.

## How to run or evaluate

Run from this candidate directory:

```bash
cd candidates/202603260836_learned-bank-qat
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
BANK_QAT_MIN_MULT=0.5 BANK_QAT_MAX_MULT=1.5 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If your local dataset/tokenizer layout differs from the repository default, override `DATA_PATH` and `TOKENIZER_PATH` explicitly.

## Validation

### Commands run in this workflow

1. Syntax check for the repository baseline and candidate:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603260836_learned-bank-qat/train_gpt.py
```

Outcome: **passed**.

2. Candidate-only syntax recheck after making the path-resolution fix:

```bash
python -m compileall candidates/202603260836_learned-bank-qat/train_gpt.py
```

Outcome: **passed**.

3. Attempted CPU-only import/quantization smoke test with a temporary `flash_attn_interface` stub.

Outcome: **blocked by environment**, because the workflow container does not currently have `torch` installed even though it appears in the repository `requirements.txt`.

### Why a fuller smoke test was not feasible here

This script imports PyTorch at module import time and requires CUDA in `main()`. A true start-up smoke test therefore needs the repository Python dependencies available locally, especially `torch`. In this workflow container, that dependency is absent, so the lowest-cost reliable validation available was static compilation.

## Main expected risks or tradeoffs

- **Step-time overhead:** fake-quantizing bank slices late in training could reduce the number of steps reached within the 10-minute cap.
- **Clip-multiplier instability:** per-slice scale learning may overfit or drift toward a worse local quantizer if the clamp range is too loose.
- **Interaction with EMA/SWA/TTT:** the stack is already complex; a gain in post-quant roundtrip quality may not translate linearly to the final TTT score.
- **Compile behavior:** this candidate avoids doing bank fake-quant work from step 0 by activating late QAT once and recompiling at that point, but the only way to fully confirm the intended path is a real torch+CUDA run.
