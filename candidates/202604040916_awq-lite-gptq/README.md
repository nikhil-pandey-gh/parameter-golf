# AWQ-lite Activation-Aware GPTQ-lite

## Hypothesis

The current 11-layer stack is already strong enough that another small training-side tweak is less attractive than reducing the **post-training int6 export error**. A lightweight activation-aware quantizer should preserve the same model architecture and training recipe while improving quantized `val_bpb` by protecting channels that matter most to actual layer inputs, not just to raw weight MSE.

## Why this is promising here

- The repo history shows repeated wins from **better export/compression**, especially embedding protection and GPTQ-lite clip search, while several architecture directions already look saturated.
- The strongest recent records still rely on a mostly **weight-only** export path, so activation-aware quantization is a relatively clean unexplored axis.
- This repository evaluates the dequantized roundtrip artifact, so even a modest reduction in quantization loss can matter directly.

## Prior records and candidates that influenced this

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
  I used this as the direct base because it already isolates the GPTQ-lite export path on top of the modern 11-layer XSA/EMA/Partial-RoPE stack.
- **Related record trends:**
  - `2026-03-18_FP16Embed_WD3600` showed that export quality, not just training loss, is often the bottleneck.
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` and `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` suggest the main training stack is already quite mature.
- **Prior candidates:** none existed in `candidates/` when this candidate was created.

## External research that informed it

- **AWQ**: activation-aware weight quantization uses activation statistics to protect salient channels rather than treating all weights equally.  
  <https://arxiv.org/abs/2306.00978>
- **GPTQ**: one-shot post-training quantization motivated keeping the change in the export path rather than building a large new training system.  
  <https://arxiv.org/abs/2210.17323>
- **SmoothQuant**: reinforced the broader idea that moving quantization difficulty using activation information can preserve model quality.  
  <https://arxiv.org/abs/2211.10438>
- **SpinQuant / QuaRot**: recent evidence that outlier-aware equalization or rotations are a strong direction for low-bit LLM export, even though this candidate implements a much simpler repo-friendly variant.  
  <https://arxiv.org/abs/2405.16406>  
  <https://arxiv.org/abs/2404.00456>

## What changed vs the chosen base

1. Added a short **post-EMA activation calibration pass** over a few training batches.
2. Replaced weight-only GPTQ-lite percentile search with an **AWQ-lite search**:
   - collect per-layer input RMS statistics,
   - try a small grid of input-channel scaling exponents,
   - score candidate int6 clips with an activation-weighted reconstruction objective,
   - store per-matrix input scales only when the activation-aware path wins.
3. Adjusted the script defaults so it can be run directly from this candidate directory without manually rewriting `DATA_PATH` and `TOKENIZER_PATH`.

Everything else stays intentionally close to the 2026-03-22 11L EMA GPTQ-lite record so the candidate isolates the export-side idea.

## How to run

From this candidate directory:

```bash
cd candidates/202604040916_awq-lite-gptq
RUN_ID=awq_lite_gptq \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
AWQ_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
AWQ_CALIBRATION_BATCHES=8 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script defaults to the repository dataset/tokenizer under `../../data/...` via a resolved repo root, so it is runnable from inside this folder. You can still override `DATA_PATH` and `TOKENIZER_PATH` explicitly if needed.

## Main risks and tradeoffs

- The export path is slower because it adds calibration plus a small clip/alpha search per matrix.
- Per-matrix input scales consume some artifact bytes, so the quality gain must outweigh the extra metadata.
- The improvement is expected mostly in **post-quant** quality; pre-quant validation loss may be unchanged.
- This version does not yet port the same idea onto the parameter-banked top record stack, so the best next follow-up would be testing that transfer if the export trick looks promising.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604040916_awq-lite-gptq/train_gpt.py` | Passed |
| `python -m compileall train_gpt.py train_gpt_mlx.py data` | Passed |
| Import / tiny construction smoke | Not feasible on this runner: the environment is missing the repository's baseline Python deps (`numpy`, `torch`, `sentencepiece`), so even a non-training import check fails before model execution. |

