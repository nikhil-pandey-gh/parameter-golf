# LSQ-Late-QAT on the GPTQ-lite 11L Stack

## Hypothesis

The next meaningful gain is likely to come from **training the export quantizer itself**, not from adding yet another feature module. This candidate replaces the brittle fixed-step late fake-quant path with an **LSQ-inspired late QAT pass** that learns per-row int6 step sizes on the strongest pre-TTT 11-layer stack, then reuses those learned scales as an extra candidate during export-time GPTQ-lite clipping.

## Why this is promising here

Repository evidence keeps pointing at the same bottleneck:

- long-context and sliding-window evaluation delivered the early gains,
- 11 layers + MLP3x + XSA + EMA + Partial RoPE pushed the pre-quant model much lower,
- but post-training quantization remained a recurring source of loss, even in strong runs,
- and one prior "late QAT" implementation was explicitly called out as ineffective because `torch.compile` constant-folded the Python flag.

That makes learned export-aware quantization one of the few high-leverage ideas that is still underexplored in this repo and fits the codebase well.

## Prior records that influenced this candidate

- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`**  
  Chosen base implementation. It already has the strongest clean pre-TTT stack in this repository: 11L, XSA4, Partial RoPE, LN scale, VE128, EMA, warmdown3500, and GPTQ-lite export.
- **`records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`**  
  Important negative lesson: the prior late-QAT hook was present but effectively dead because the toggle was constant-folded.
- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`**  
  Influenced the decision to leave TTT and heavy systems work out of this candidate and instead isolate a training/export improvement on the strongest simpler stack.
- **`records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`** and **`records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`**  
  Both reinforce that export damage is real and worth attacking directly.

## External research that informed it

- **LSQ: Learned Step Size Quantization** — <https://arxiv.org/abs/1902.08153>  
  Motivation for learning quantizer step sizes instead of freezing them to heuristics.
- **EfficientQAT** — <https://arxiv.org/abs/2407.11062>  
  Motivation for a short late phase that trains quantization parameters rather than paying full-QAT cost across the whole run.
- **CrAM: Compression-Aware Minimization** — <https://arxiv.org/abs/2207.14200>  
  Motivation for explicitly biasing training toward weights that survive downstream compression better.

## What changed vs the chosen base implementation

1. **LSQ-style per-row late QAT for `CastedLinear` weights**
   - Each eligible linear gets a learnable per-row `qat_log_step`.
   - Late QAT is controlled by a per-module blend buffer instead of a static class flag, avoiding the earlier constant-folding failure mode.
   - QAT starts late, re-initializes scales from current weights, and ramps in over `LSQ_QAT_BLEND_STEPS`.

2. **Export now consults learned scales**
   - GPTQ-lite percentile search is still present.
   - For int6 matrices, export also evaluates the learned LSQ row scales and keeps whichever reconstruction is lower-MSE.

3. **Standalone fallback / smoke support**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA.
   - `SMOKE_TEST=1` runs a tiny CPU-only forward/backward/export roundtrip with random tokens and no dataset.

## How to run

### Full training / evaluation

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
LSQ_QAT_ENABLED=1 LSQ_QAT_THRESHOLD=0.18 LSQ_QAT_LR=0.004 \
LSQ_QAT_INIT_PERCENTILE=0.9999 LSQ_QAT_BLEND_STEPS=96 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script now resolves the dataset and tokenizer from the repository root by default, so it is runnable directly from this candidate directory without rewriting `DATA_PATH` / `TOKENIZER_PATH`.

### Minimal smoke path

```bash
SMOKE_TEST=1 python3 train_gpt.py
```

This does not need the dataset or tokenizer, only a Python environment with `torch`.

## Main risks / tradeoffs

- **Training-time overhead:** even late-only fake quant adds some extra math on every eligible linear once the blend ramps in.
- **Scale mismatch risk:** learned row scales may help some matrices and hurt others, so export keeps the GPTQ-lite fallback path.
- **Unverified full-run sensitivity:** the defaults are research-motivated and code-complete, but not yet GPU-swept in this environment.
- **Artifact budget interaction:** better int6 survivability may or may not translate into better final BPB once compression and sliding eval are included.

## Validation

Commands run here:

```bash
python3 -m compileall candidates/202604091053_lsq-late-qat/train_gpt.py
python3 - <<'PY'
import importlib.util
for mod in ['torch', 'numpy', 'sentencepiece']:
    print(f'{mod}={bool(importlib.util.find_spec(mod))}')
PY
python3 -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
```

Outcomes:

- `compileall` **passed**.
- The local Python 3 runtime in this container does **not** currently have `torch`, `numpy`, or `sentencepiece`.
- The one-step `pip` install attempt failed because the environment is externally managed, so I did **not** run the smoke path here.
- The script still includes `SMOKE_TEST=1` so a minimal CPU startup check is available in a repo-ready environment with `torch` installed.
