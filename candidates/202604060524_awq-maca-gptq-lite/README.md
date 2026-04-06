# Activation-Aware Multi-Scale GPTQ-lite Export

## Hypothesis

The current best non-TTT stack is already close to the 16 MB ceiling, so the highest-ROI next step is to improve **how the trained model is packed**, not to spend more training budget on a riskier architecture change. Replacing the existing weight-only GPTQ-lite clip search with **activation-aware, multi-scale calibration** should better protect the channels that matter most for BPB, especially when evaluation uses long contexts and sliding windows.

## Why this is promising here

This candidate keeps the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` training stack intact and only changes the export-time int6 search:

- that record already showed that a better clip search can buy measurable BPB at **zero training-step cost**;
- earlier records repeatedly showed that quantization quality, fp16/int8 embedding treatment, and export details are decisive under the 16 MB artifact cap;
- repository review also showed that heavier QAT/recurrent ideas can cost too many steps or fail silently under `torch.compile`, while export-side improvements are safer.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
   Base implementation and strongest pure train/export record.
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`  
   Confirms that the current backbone gains came from the architecture stack, not the earlier broken late-QAT branch.
3. `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`  
   Reinforced that quantization/export choices dominate the final submission metric.

## External research that informed it

- Ji Lin et al., **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** (2023, MLSys 2024).  
  <https://arxiv.org/abs/2306.00978>
- Elias Frantar et al., **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (2022).  
  <https://arxiv.org/abs/2210.17323>
- Seungwoo Son et al., **On the Importance of a Multi-Scale Calibration for Quantization** (2026).  
  <https://arxiv.org/abs/2602.07465>
- Bruce Changlong Xu, **Activation Sensitivity as a Unifying Principle for Post-Training Quantization** (2026).  
  <https://arxiv.org/abs/2601.11663>

The concrete takeaway applied here is simple: use **activation statistics**, and gather them across **multiple sequence lengths**, before choosing low-bit clipping scales.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. Added **activation-aware int6 clip selection**: per-row clip candidates are now scored with activation-weighted reconstruction error instead of plain weight MSE.
2. Added **multi-scale calibration**: activation RMS is collected from short and long sequence windows before export.
3. Kept the base record's model/training stack unchanged: 11L, XSA4, Partial RoPE, LN scale, VE128, EMA, warmdown 3500, late QAT, int6 block packing, int8 embeddings.
4. Added small **portability helpers** so the script can be launched from this candidate directory and can fall back to PyTorch SDPA when FlashAttention is unavailable.
5. Added optional **`VAL_TOKEN_LIMIT`** and compile gating for cheap local syntax/startup checks without changing the default training path.

## How to run or evaluate it

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs for ablation/debugging:

```bash
AWQ_ENABLED=0                       # fall back to the weight-only export path
AWQ_CALIBRATION_SEQ_LENS=256,512,1024,2048
AWQ_CALIBRATION_TOKENS=32768
AWQ_CLIP_PERCENTILES=0.9990,0.9995,0.9999,0.99999,1.0
VAL_TOKEN_LIMIT=0                   # keep full validation by default
USE_TORCH_COMPILE=1
```

## Main expected risks and tradeoffs

- Export now pays an extra calibration pass after EMA, so packing time increases.
- Activation-aware calibration can still overfit if the sampled windows are not representative enough.
- This does **not** change the backbone or optimizer stack, so gains may be incremental rather than dramatic.
- The CPU/SDPA path is only for smokeability; the intended competitive path is still the original CUDA + FlashAttention setup.

## Validation

### Commands run

```bash
python -m compileall candidates/202604060524_awq-maca-gptq-lite/train_gpt.py
```

```bash
cd candidates/202604060524_awq-maca-gptq-lite && \
USE_TORCH_COMPILE=0 AWQ_ENABLED=1 AWQ_CALIBRATION_TOKENS=512 \
AWQ_CALIBRATION_SEQ_LENS=32,64 ITERATIONS=1 MAX_WALLCLOCK_SECONDS=0 \
TRAIN_SEQ_LEN=64 EVAL_SEQ_LEN=64 TRAIN_BATCH_TOKENS=64 VAL_BATCH_SIZE=128 \
VAL_TOKEN_LIMIT=4097 VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=1 WARMUP_STEPS=0 \
python train_gpt.py
```

### Outcomes

- `compileall` **passed**.
- The minimal runtime smoke test was **not feasible in this checkout**:
  - the environment does not have the runtime Python deps installed yet (`ModuleNotFoundError: No module named 'numpy'`);
  - the checkout also does not include `data/datasets/` or `data/tokenizers/`, so even with dependencies installed there is no local shard/tokenizer payload to boot a real train/eval run.
