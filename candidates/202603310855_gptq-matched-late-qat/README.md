# GPTQ-matched late QAT

## Hypothesis

The strongest non-TTT stack in this repo already uses a smarter export quantizer than its training-time fake quantizer.

This candidate tests a simple idea: during the late-QAT phase, make the model adapt to the **same percentile-searched per-row int6 scales** that the export path uses, instead of training against a cruder row-max fake quantizer.

If the remaining gap is mostly a train/deploy mismatch, matching those quantizers should reduce the post-quantization penalty without adding new model parameters, new eval-only logic, or a large architectural rewrite.

## Why this is promising for this repository

Repository review suggests that recent wins come from stacking many small improvements on the same 11-layer int6-aware backbone rather than from wholesale architecture changes.

The most relevant local evidence was:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
  - showed that better GPTQ-style clip selection was worth roughly `-0.0006` BPB even *without* extra training.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - explicitly documented that a prior late-QAT path was effectively dead because `torch.compile` constant-folded the class-attribute flag.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - showed there is still headroom above the March 22 stack, but part of that stack's score comes from TTT and parameter banking complexity that I did not want to entangle with this candidate.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - reported that layer recurrence hurt badly in a fixed wall-clock regime, which pushed this candidate away from recurrent-depth ideas and toward a quantization-alignment idea instead.

There were no prior `candidates/` in the repository when this candidate was created, so the comparison set was the root baseline plus the existing `records/` tree.

## External research that informed this candidate

- **GPTQ** — Frantar et al., 2022 (`arXiv:2210.17323`): motivated the focus on using a more accurate low-bit weight quantizer instead of a naive max-based scale.
- **Learned Step Size Quantization (LSQ)** — Esser et al., 2019 (`arXiv:1902.08153`): reinforced the idea that quantizer-step choice matters during training, not only at export time.
- **SmoothQuant** — Xiao et al., 2022 (`arXiv:2211.10438`): further evidence that scale calibration is a first-order issue in quantized language models.
- I also reviewed parameter-sharing / recurrent-depth literature such as **ALBERT** (`arXiv:1909.11942`) and the **Universal Transformer** (`arXiv:1807.03819`), but the repo's own recurrence ablation looked weak enough that quantizer matching felt like the stronger next bet for this codebase.

## Chosen base implementation

This candidate starts from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

I chose that file because it is the strongest clean training-side stack in the repo without dragging in the March 23 record's TTT, parameter banking, and larger code surface.

## What changed versus the base

1. **Late QAT now targets the export quantizer more faithfully.**
   - Attention and MLP `CastedLinear` layers now carry cached per-row int6 scales.
   - Those scales are chosen with the same 5-percentile search used by export-time GPTQ-lite quantization.

2. **The late-QAT activation path no longer depends on a compile-fragile class attribute.**
   - Each linear layer now has non-persistent QAT buffers (`_qat_active`, `_qat_scale`) instead of a global boolean that can be constant-folded away.
   - This directly addresses the failure mode documented in the March 21 partial-RoPE record.

3. **Scale search is refreshed periodically, not every forward.**
   - New knob: `QAT_REFRESH_EVERY` (default `32`).
   - This keeps the fake-quant path aligned with current weights without making every forward pay the full percentile-search cost.
   - Forward passes use a cached dequantized int6 weight tensor, so the expensive quantizer reconstruction only runs on refresh steps after QAT is active.
   - The late-QAT enable signal is also synchronized across DDP ranks before the first refresh so all workers enter the matched-QAT phase together.

4. **The export helper reuses the same scale-selection logic.**
   - The candidate factors the int6 scale search into a shared helper so train-time QAT and export-time int6 quantization are synchronized.

5. **The script is runnable from inside the candidate directory.**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root inferred from `__file__`, so `cd candidates/... && torchrun ... train_gpt.py` works without having to override paths immediately.

## How to run

From the candidate directory:

```bash
cd candidates/202603310855_gptq-matched-late-qat
RUN_ID=gptq_matched_late_qat \
SEED=1337 \
QAT_REFRESH_EVERY=32 \
INT6_CLIP_PERCENTILES=0.999,0.9995,0.9999,0.99999,1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults in this script already match the March 22-style 11-layer stack closely, including:

- `NUM_LAYERS=11`
- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=786432`
- `XSA_LAST_N=4`
- `ROPE_DIMS=16`
- `LN_SCALE=1`
- `VE_ENABLED=1`
- `WARMDOWN_ITERS=3500`
- `MATRIX_LR=0.025`, `SCALAR_LR=0.025`, `TIED_EMBED_LR=0.035`

## Expected tradeoffs / risks

- **Refresh overhead:** percentile-searching scales every `QAT_REFRESH_EVERY` steps adds some warmdown overhead. If this is too expensive on 8xH100, the refresh interval may need to increase.
- **Compile interaction risk:** this candidate avoids the specific class-attribute issue from March 21, but `torch.compile` can still be subtle around mutated module buffers, so this is worth watching carefully in a real GPU run.
- **No training-side architecture change:** if the remaining gap versus the March 23 record mostly comes from activation changes or TTT, quantizer matching alone may not be enough.
- **Quantizer locality:** this only applies the matched-QAT path to int6-exported attention and MLP matrices, not to every float tensor in the model.

## Validation

I ran the following lightweight checks in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603310855_gptq-matched-late-qat/train_gpt.py
```

Outcome: **passed**.

I also checked whether a runtime smoke test was feasible:

```bash
python - <<'PY'
import importlib.util
mods = ['torch', 'sentencepiece', 'flash_attn_interface']
for m in mods:
    print(f'{m}:', 'present' if importlib.util.find_spec(m) else 'missing')
PY
```

Outcome in this execution environment:

- `torch: missing`
- `sentencepiece: missing`
- `flash_attn_interface: missing`

Because those required runtime dependencies are unavailable here, and this script requires CUDA, I did **not** run a CPU smoke launch. The syntax validation above is therefore the strongest safe check I could perform locally in this workflow.
