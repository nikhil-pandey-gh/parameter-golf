# Activation-Aware GPTQ-lite + LeakyReLU²

## Hypothesis

The current 11-layer EMA + GPTQ-lite stack is already close to the artifact limit, so the best next gain is likely to come from **shrinking the quantization gap** rather than adding more train-time machinery. This candidate tests an **AWQ-lite** idea: use a short post-training calibration pass on train tokens to measure per-channel activation RMS, then use those activation statistics to choose the GPTQ-lite clip percentile that minimizes **activation-weighted** reconstruction error instead of plain weight MSE.

I also carry over the repo's strongest recent MLP-local win, **LeakyReLU(0.5)^2**, because it was the clearest training-side improvement on the next record family.

## Why this is promising here

This repository's recent gains have come from:

1. keeping the 11-layer/XSA/EMA/partial-RoPE stack stable,
2. making export smarter (`GPTQ-lite`, better warmdown, EMA), and
3. finding cheap local modeling wins such as LeakyReLU².

An activation-aware clip search fits those trends well:

- it leaves the main training loop almost unchanged,
- it spends extra work only in the post-training export path,
- it targets the same remaining weakness that GPTQ-lite already attacked,
- and it keeps the artifact format self-contained.

## Prior repo influences

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Training-side carry-over:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` for LeakyReLU(0.5)^2
- **No prior `candidates/` directory existed** when this candidate was created.

## External research

- **AWQ** (Lin et al., 2023, arXiv:2306.00978): activation-aware protection of salient channels during low-bit quantization.
- **SmoothQuant** (Xiao et al., 2022, arXiv:2211.10438): use activation statistics to make weights easier to quantize.
- **GPTQ** (Frantar et al., 2022, arXiv:2210.17323): post-training quantization with reconstruction-aware objectives.

This candidate intentionally implements a **small, single-file AWQ-inspired variant** instead of full channel folding: it reuses the existing GPTQ-lite clip search, but scores each clip candidate with **activation-weighted row error** gathered from a short calibration pass.

## What changed vs the chosen base

1. **LeakyReLU² MLP**
   - Replaces `relu(x)^2` with `leaky_relu(x, 0.5)^2`.
2. **Activation-aware calibration pass**
   - Adds `collect_activation_rms(...)`, which runs a short eval-mode pass over training shards after EMA is applied.
   - Collects per-input-channel RMS for large `CastedLinear` weights.
3. **Activation-weighted int6 clip search**
   - `quantize_int6_per_row(...)` now prefers the clip percentile with the lowest activation-weighted reconstruction error when calibration stats are available.
4. **New env knobs**
   - `AWQ_LITE_ENABLED` (default `1`)
   - `AWQ_CALIBRATION_TOKENS` (default `131072`)
   - `AWQ_CALIBRATION_SEQ_LEN` (default `2048`)

The rest of the 11-layer EMA/GPTQ-lite/XSA/partial-RoPE/value-embedding stack stays aligned with the 2026-03-22 record.

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
AWQ_CALIBRATION_TOKENS=262144 AWQ_CALIBRATION_SEQ_LEN=2048 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

By default the script resolves `data/` relative to the repository root, so the command above works from inside the candidate directory as long as the repository keeps the usual layout:

- `./data/datasets/fineweb10B_sp1024/fineweb_train_*.bin`
- `./data/datasets/fineweb10B_sp1024/fineweb_val_*.bin`
- `./data/tokenizers/fineweb_1024_bpe.model`

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall candidates/202604080604_activation-aware-gptq/train_gpt.py` | Passed |
| Minimal CPU smoke test | Not feasible in this workflow environment because the runtime dependencies required by the script (`torch`, `numpy`, `sentencepiece`) were not installed |

## Main risks and tradeoffs

- The calibration pass adds some post-training wall-clock time.
- Activation-weighted clip selection may help only a subset of layers, or overfit a too-small calibration sample.
- This candidate combines one known training win (LeakyReLU²) with one new export idea, so any eventual gain should be re-ablated to separate the two effects.
