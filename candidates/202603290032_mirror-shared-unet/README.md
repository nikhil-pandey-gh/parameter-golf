# Candidate: Mirror-Shared U-Net Trunk

## Hypothesis

The strongest non-TTT stack in this repo already gets excellent quality from an 11-layer U-Net-shaped transformer with XSA, Partial RoPE, LN scaling, VE128, EMA, and GPTQ-lite export. My hypothesis is that this stack can afford **cross-layer weight sharing in the mirrored encoder/decoder trunk** without paying the fixed-wallclock penalty that hurt earlier recurrence experiments, and that the recovered artifact bytes are best reinvested into a slightly richer **BigramHash** prior.

Concretely, this candidate shares the heavy `attn` + `mlp` modules across layer pairs `2<->8`, `3<->7`, and `4<->6`, while keeping the per-layer residual controls (`attn_scale`, `mlp_scale`, `resid_mix`), U-Net skip weights, XSA routing, and layer-index LN scaling distinct.

## Why this is promising for this repository

Recent records show a clear pattern:

- the repo's best non-TTT core is the 2026-03-22 `11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack;
- the latest record shows `LeakyReLU(0.5)^2` is a cheap, real win;
- older experiments show that **extra recurrent passes** can lose under a 10-minute wallclock because they reduce step count too much.

This candidate tries to take the parameter-sharing / recurrent-depth literature seriously **without** repeating that full-looping mistake. The model still executes the same 11 passes, so step-time should stay much closer to the proven 11-layer regime than a doubled-depth looped model would.

## Prior records and experiments that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Key ideas borrowed directly from prior wins:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: 11L U-Net stack, XSA4, Partial RoPE, LN scaling, VE128, EMA, GPTQ-lite, warmdown3500.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: `LeakyReLU(0.5)^2` MLP activation.
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`: the 11-layer/XSA/EMA transition that established the modern core.

Negative-result guidance I explicitly used:

- `2026-03-18_FP16Embed_WD3600`: notes that depth recurrence looked promising but needed more steps than the 10-minute budget allowed.
- `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`: full layer recurrence was a bad fixed-wallclock tradeoff in that exploration.

No prior `candidates/` directory existed in this repository snapshot, so there were no earlier candidate iterations to avoid duplicating.

## External research that informed this candidate

- **ALBERT** (`arXiv:1909.11942`): motivates cross-layer parameter sharing as a way to reduce parameters while preserving much of the representational benefit of deeper transformers.
- **Universal Transformer** (`arXiv:1807.03819`): motivates recurrent / shared-depth computation as a useful inductive bias, especially when full independent layers are too expensive.
- **BitNet b1.58** (`arXiv:2402.17764`): reinforces the repo's broader compression-aware direction; rather than inventing a new quantization stack here, I keep the proven low-bit export path and spend the novelty budget on model structure.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Mirror-shared trunk weights**
   - Added `SHARED_LAYERS` (default: `2:8,3:7,4:6`).
   - These pairs share the heavy attention and MLP submodules while leaving per-layer control tensors unique.
   - Export now prunes duplicated shared state entries and restores aliases on load, so the artifact actually benefits from sharing.

2. **LeakyReLU(0.5)^2 MLP activation**
   - Replaces the base `relu^2` MLP nonlinearity with the activation used in the 2026-03-23 record.

3. **Bigger default BigramHash table**
   - `BIGRAM_VOCAB_SIZE` default changed from `2048` to `3072` to spend some of the recovered bytes on a larger token-pair prior.

4. **Safer attention backend import**
   - `flash_attn_interface` import is now optional.
   - If unavailable, the attention path falls back to PyTorch `scaled_dot_product_attention` instead of crashing at import time.

5. **Run-from-candidate defaults**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repo root computed from `__file__`, so the script works when launched from this candidate directory.

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202603290032_mirror-shared-unet

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

A more explicit reproduction command, matching the intended defaults, is:

```bash
cd candidates/202603290032_mirror-shared-unet

NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=3072 \
SHARED_LAYERS=2:8,3:7,4:6 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
VE_ENABLED=1 \
VE_LAYERS=9,10 \
LATE_QAT_THRESHOLD=0.15 \
WARMDOWN_ITERS=3500 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- This candidate keeps the 2026-03-22 training/export regime, so it still expects the existing FineWeb shard format and SentencePiece tokenizer.
- The script still targets the standard CUDA challenge environment for real training/eval.
- `flash_attn_interface` is optional at import time, but the intended fast path is still the CUDA/FlashAttention setup used by the existing records.

## Validation run in this workflow

Lightweight checks performed here:

```bash
python -m compileall candidates/202603290032_mirror-shared-unet/train_gpt.py
wc -c candidates/202603290032_mirror-shared-unet/train_gpt.py
python - <<'PY'
import importlib.util
print(importlib.util.find_spec('torch'))
PY
```

Outcomes:

- `compileall` **passed**.
- Script size is **71,395 bytes**.
- `torch` is **not installed** in this workflow container (`find_spec('torch') -> None`), so a truthful runtime smoke test was **not feasible here** without adding heavyweight infrastructure that the repository does not already use in this environment.

## Main expected risks / tradeoffs

- **Sharing can oversmooth the trunk.** Even with per-layer control vectors retained, mirrored encoder/decoder layers may want more specialization than this scheme allows.
- **The export aliasing path is new.** The training logic remains close to the base record, but serialization now depends on explicit alias-pruning / alias-restoration for shared blocks.
- **Bigger BigramHash may not be the best use of saved bytes.** If sharing regularizes too aggressively, those bytes might be better spent on VE capacity or another low-rank specialization path.
- **This is intentionally a non-TTT candidate.** It aims to improve the strongest reusable core first; if it works, the obvious next experiment is to layer legal score-first TTT on top.
