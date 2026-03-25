# Cyclic Shared XSA Tail

## Hypothesis

The current best training-only stacks in this repository already seem close to saturating the obvious 11-layer / 512-dim / XSA / EMA / GPTQ-lite recipe. The next useful lever is probably **better parameter allocation**, not just another local hyperparameter tweak. This candidate tests an **ALBERT-style shallow weight-sharing scheme** that reuses only the expensive early transformer block weights while keeping **cheap per-layer adapters** and a **fully unique deep XSA tail**.

Concretely, the first 6 logical layers are implemented with only 3 physical shared blocks used in a cyclic pattern `[0, 1, 2, 0, 1, 2]`, while the final 5 logical layers remain unique. The hypothesis is that early language-modeling features are redundant enough to share expensive attention/MLP weights, and that the saved bytes can be better spent on a slightly richer local signal (`BIGRAM_VOCAB_SIZE=3072`) without paying extra training-time FLOPs.

## Why this is promising for this repository

Repository evidence says the strongest direction so far is a mature 11-layer stack with deep-layer specialization, compression-aware export, and strong evaluation. The best training-only record and the current best overall record both keep improving the same basic recipe rather than replacing it:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

At the same time, the repository has already mined many of the lower-risk knobs:

- deeper 9L/10L/11L scaling,
- partial RoPE,
- LN scaling,
- XSA on late layers,
- BigramHash and SmearGate,
- EMA/SWA,
- GPTQ-lite / mixed-bit quantization,
- sliding eval and TTT.

What is still relatively underexplored is **parameter sharing without extra per-step compute**. A non-record 1x5090 experiment reported that naive layer recurrence with extra compute was bad, but that is a different failure mode than this candidate. This candidate does **not** add extra unrolled depth or extra forward passes; it keeps the same logical 11-layer execution and only shares the underlying expensive weights in the shallow prefix.

## Prior records that influenced this candidate

### Primary base

This candidate is a direct fork of the clean training-only 11-layer stack from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

That base already includes the strongest non-TTT ingredients in one place:

- 11 logical layers,
- 3x MLP,
- deep XSA,
- partial RoPE,
- LN scale,
- EMA,
- GPTQ-lite int6 export,
- Value Embeddings on late layers,
- SmearGate + BigramHash.

### Additional influences

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - reinforced that the late 11-layer XSA tail with partial RoPE and LN scaling is worth preserving.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - showed there is still marginal value in spending some of the artifact budget on a larger local-feature budget (`BIGRAM_VOCAB_SIZE=3072` there), so this candidate modestly grows the bigram table after recovering bytes via sharing.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - useful negative result: extra recurrent compute was bad in a fixed-wallclock regime, which is exactly why this candidate avoids adding more iterative steps.

## External research that informed it

This implementation is grounded in three primary-source ideas:

- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** (`arXiv:1909.11942`)
  - key takeaway: large transformer stacks can often tolerate aggressive parameter sharing much better than naive parameter counting would suggest.
- **Universal Transformers** (`arXiv:1807.03819`)
  - key takeaway: recurrence across depth can be a useful inductive bias even when computation is still parallel over sequence positions.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (`arXiv:2505.01855`)
  - key takeaway: recurrence is more promising when applied selectively, and earlier layers appear to benefit most from repeated/shared processing.

I did **not** implement full recurrent-depth execution from those papers because this repository is strongly wallclock-constrained. Instead, this candidate takes the part that seems most compatible with the challenge: **share early expensive weights at fixed compute, keep a unique deep tail, and preserve lightweight per-layer specialization**.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Added cyclic shallow block sharing**
   - New hyperparameters:
     - `SHARED_PREFIX_UNIQUE=3`
     - `SHARED_PREFIX_REPEATS=2`
   - Default logical-to-physical block plan becomes:
     - `[0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7]`
   - This reduces the expensive transformer core from 11 physical blocks to 8 while keeping 11 logical layer applications.

2. **Split each logical layer into shared heavy weights + cheap per-layer adapters**
   - Shared physical block contains:
     - attention projections,
     - MLP projections,
     - rotary setup,
     - XSA math.
   - Per-layer adapter keeps:
     - RMSNorms,
     - q-gain,
     - residual mix,
     - attention/MLP scales,
     - optional DTG gate.

3. **Kept the late tail unique**
   - The final 5 logical layers are unique physical blocks.
   - The final 4 logical layers are still the XSA tail.
   - This preserves the repo’s strongest empirical pattern: late-layer specialization matters more than early-layer uniqueness.

4. **Increased BigramHash budget slightly**
   - Default `BIGRAM_VOCAB_SIZE` changed from `2048` to `3072`.
   - The idea is to spend a portion of the recovered bytes on a better local lexical prior rather than simply shrinking the artifact.

5. **Disabled late QAT by default**
   - `LATE_QAT_THRESHOLD` now defaults to `0.0`.
   - The candidate is meant to stand on the sharing idea itself, and this avoids relying on the previously documented compile-sensitive late-QAT path.

## How to run / evaluate

From the candidate directory:

```bash
cd candidates/202603250929_cyclic-shared-xsa

RUN_ID=cyclic_shared_xsa \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=11 \
SHARED_PREFIX_UNIQUE=3 \
SHARED_PREFIX_REPEATS=2 \
BIGRAM_VOCAB_SIZE=3072 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
LATE_QAT_THRESHOLD=0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still performs the usual repository pattern:

- training,
- EMA application,
- int6 roundtrip export,
- sliding-window eval.

## Main expected risks / tradeoffs

- **Optimization risk**: even with per-layer adapters, shallow sharing could reduce useful specialization enough to hurt pre-quant quality.
- **Quantization interaction risk**: sharing might improve compression but also create broader weight distributions in shared blocks, which could hurt int6 export.
- **Compile risk**: this candidate changes the block wiring substantially; it compiles syntactically, but a real CUDA run is still needed to confirm `torch.compile` and FlashAttention behave exactly as expected.
- **Budget allocation risk**: spending recovered bytes on `BIGRAM_VOCAB_SIZE=3072` may or may not be the best use of the artifact headroom.

## Validation

### Commands run

1. Syntax check:

```bash
python -m compileall candidates/202603250929_cyclic-shared-xsa/train_gpt.py
```

Result: **passed**.

2. Static wiring sanity check:

```bash
python - <<'PY'
from pathlib import Path
path = Path('candidates/202603250929_cyclic-shared-xsa/train_gpt.py')
text = path.read_text()
checks = {
    'shared_prefix_unique': 'shared_prefix_unique' in text,
    'shared_prefix_repeats': 'shared_prefix_repeats' in text,
    'build_block_plan': 'def build_block_plan' in text,
    'layer_adapters': 'layer_adapters' in text,
}
print(checks)
PY
```

Result: **all expected candidate-specific symbols present**.

3. Runtime dependency probe:

```bash
python - <<'PY'
import importlib.util
for name in ('torch', 'flash_attn_interface', 'sentencepiece'):
    print(f'{name}={bool(importlib.util.find_spec(name))}')
PY
```

Result in this environment: `torch=False`, `flash_attn_interface=False`, `sentencepiece=False`.

### Why a CPU runtime smoke test was not feasible here

A true import-and-instantiate smoke test was **not feasible in this workflow environment** because the required runtime dependencies for the training script are not installed locally. The script is syntactically valid, but an actual start-up test will need the normal repository runtime environment with `torch`, `sentencepiece`, and FlashAttention available.
