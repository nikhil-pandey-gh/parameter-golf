# LSQ-style Late QAT on the 11L GPTQ-lite EMA Base

## Hypothesis

The strongest pure-training record in this repo already wins by making the final artifact more quantization-friendly: EMA, warmdown tuning, partial RoPE, GPTQ-lite clip search, and int6 export all reduce the post-training quantization penalty. This candidate tests the next direct step in that direction: replace fixed row-max fake quantization with **learned step-size int6 QAT** on transformer block matrices, then enable it only late in training so the model keeps using full-precision weights for most of training.

## Why this is promising for this repo

- The record history shows that quantization robustness is still one of the highest-leverage knobs:
  - `2026-03-18_FP16Embed_WD3600` found the tied embedding is extremely quant-sensitive.
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` and `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` kept squeezing gains out of the quantization path even after the architecture largely stabilized.
- The chosen base already exports int6 block weights with GPTQ-lite clip search, but its training-time fake quant path still used a fixed row-max scale.
- Learned Step Size Quantization (LSQ) is explicitly designed to improve low-bit training with a small code change instead of a new training stack.

## Influences from prior records

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Quantization cautionary input:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` (late-QAT compile caveat)
- **Earlier quant-gap work:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
- **Repository review result:** there were **no prior `candidates/` runs** to incorporate

## External research

- **LSQ**: Esser et al., *Learned Step Size Quantization* ([arXiv:1902.08153](https://arxiv.org/abs/1902.08153)) motivated replacing fixed fake-quant scales with learned step sizes.
- During the research pass I also compared ALBERT-style sharing, Universal Transformer / recurrent depth reuse, linear-attention replacements, and pruning/compression-first schemes. Those looked less compatible with this repo's combination of a 16MB artifact limit and a strict 10-minute training cap, especially because prior local evidence already showed depth reuse/regression risk.

## What changed vs the chosen base

This candidate keeps the `2026-03-22` architecture and export path intact, but changes the training-side quantization logic:

1. **LSQ-style per-row learned step sizes for `CastedLinear` block weights**
   - each transformer block matrix gets a trainable per-output-row `lsq_log_step`
   - step sizes are initialized from row mean absolute weight magnitude and refreshed when late QAT turns on
2. **Late-QAT activation without a shared class flag**
   - the old shared class-flag switch is replaced with a module-local late-QAT toggle on each block matrix
   - LSQ scales are refreshed from the current weights at activation time instead of reusing initialization-time values
3. **Export keeps the challenge-relevant artifact unchanged**
   - LSQ parameters are training-only and are excluded from export
   - post-training export is still the same GPTQ-lite int6 path as the base run
4. **Run-from-candidate-dir defaults**
   - default data/tokenizer paths resolve relative to the repository root via `__file__`, so `train_gpt.py` can be launched from this candidate directory directly

## How to run / evaluate

From the repository root:

```bash
cd candidates/202604030142_lsq-late-qat
RUN_ID=lsq_late_qat SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable LSQ entirely
LSQ_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Enable LSQ from step 0 instead of late activation
QAT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still performs:

- EMA application before export
- GPTQ-lite int6 export
- roundtrip validation
- sliding-window evaluation

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604030142_lsq-late-qat/train_gpt.py
```

Outcome: **passed**

```bash
python - <<'PY'
from pathlib import Path
cand = Path('candidates/202604030142_lsq-late-qat/train_gpt.py').resolve()
ns = {'Path': Path, '__file__': str(cand)}
text = cand.read_text(encoding='utf-8')
prefix = text.split('try:\n    import zstandard', 1)[0]
exec(prefix, ns)
print('repo_root', ns['REPO_ROOT'])
PY
```

Outcome: `repo_root /home/runner/work/parameter-golf/parameter-golf`

```bash
python - <<'PY'
import runpy
runpy.run_path('candidates/202604030142_lsq-late-qat/train_gpt.py', run_name='candidate_import_check')
print('candidate_import_ok')
PY
```

Outcome: **failed immediately** in this runner with `ModuleNotFoundError: No module named 'numpy'`

```bash
python - <<'PY'
from pathlib import Path
root = Path('data')
print('train_bins', len(list(root.glob('datasets/fineweb10B_sp1024/fineweb_train_*.bin'))))
print('val_bins', len(list(root.glob('datasets/fineweb10B_sp1024/fineweb_val_*.bin'))))
print('tokenizer_exists', (root / 'tokenizers/fineweb_1024_bpe.model').exists())
PY
```

Outcome:

- `train_bins 0`
- `val_bins 0`
- `tokenizer_exists False`

So a local dataset-backed smoke test was **not feasible** in this workflow run: the runner lacked both the Python runtime dependencies and the FineWeb/tokenizer artifacts that the trainer expects, and the script's actual training path is CUDA-oriented.

## Main risks / tradeoffs

- **Extra training compute:** LSQ fake-quant math runs inside every transformer block matrix once active.
- **Optimizer sensitivity:** late low-bit adaptation may help or may over-regularize if the learned steps collapse too aggressively.
- **Train/export mismatch remains possible:** training now uses learned step sizes, but final export still uses GPTQ-lite post-training quantization.
- **`torch.compile` interactions:** this candidate explicitly avoids the previous late-QAT class-flag pitfall, but compile behavior is still the main engineering risk to watch on real H100 runs.
