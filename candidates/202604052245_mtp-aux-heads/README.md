# Candidate: Training-Only Multi-Token Prediction on the Banked LeakyReLU2 + Legal TTT Stack

## Hypothesis

The current best stack in this repository is already very strong on architecture, quantization, and evaluation. The next cheap lever is likely **training efficiency**, not another export tweak. This candidate adds **two training-only multi-token prediction (MTP) heads** to the current banked/XSA/partial-RoPE/LeakyReLU2/Legal-TTT stack so the model gets denser next-few-token supervision during the same 600-second training budget, while **excluding the auxiliary heads from export** so the final artifact budget is unchanged.

## Why this is promising here

Repository review suggests a clear trend:

- early gains came from **evaluation** and **quantization-aware export** (`SlidingWindowEval`, fp16 embedding export, int6 QAT, GPTQ-lite);
- later gains came from **incremental architecture improvements** on top of that stack (SmearGate, BigramHash, XSA, partial RoPE, LN scale, EMA);
- the current best record, `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, already compresses well and only gets a modest final boost from TTT, so another training-time-only improvement that does **not** consume artifact bytes is attractive.

MTP fits that gap especially well:

- the repository already carries a dormant MTP implementation in the latest record scripts, but no checked-in run actually enables it;
- MTP improves **sample efficiency** rather than export size, which matches this challenge's 10-minute training bottleneck;
- the extra parameters live only in auxiliary output heads and are stripped before serialization.

## Prior records that influenced this candidate

This candidate is based directly on the strongest current local base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`

It also inherits the design path established by:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` for EMA + GPTQ-lite + warmdown tuning,
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` for partial RoPE + LN scale,
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` for XSA + EMA,
- the earlier `SlidingWindowEval`, int6/QAT, fp16-embedding, and SmearGate/BigramHash runs that established the current winning stack.

Notably, the repo review also showed a few dead ends that made MTP more appealing than other options:

- lower LR alone barely moved the needle,
- longer training without stronger quantization still underperformed,
- several folders already exhaust the obvious quantization/export tweaks.

## External research that informed it

Primary sources:

1. **Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737)**  
   Proposes predicting several future tokens with independent auxiliary heads on top of a shared trunk, reporting improved sample efficiency and stronger generative behavior with no inference-time requirement to keep those heads.

2. **DeepSeek-V3 (arXiv:2412.19437)**  
   Explicitly includes a multi-token prediction training objective in a modern frontier training recipe, which is a strong signal that MTP remains a live optimization lever rather than an abandoned idea.

I also compared recent quantization papers such as **QuaRot** (arXiv:2404.00456) and **SpinQuant** (arXiv:2405.16406). Those are interesting, but in this repository they would require a more invasive reworking of the already-strong quantization/export path. By contrast, MTP is already partially wired, costs no export bytes when removed after training, and attacks a dimension the repo has not seriously explored yet.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS`: `0 -> 2`
   - `MTP_LOSS_WEIGHT`: `0.20 -> 0.15`

2. **Actually train the auxiliary MTP heads**
   - the copied base script already computed MTP loss and excluded `mtp_heads.*` from export,
   - but the `mtp_heads` weights were not routed into any optimizer group,
   - this candidate adds those weights to the existing AdamW-managed replicated parameter set.

3. **Align the default TTT freeze setting with the best published recipe**
   - `TTT_FREEZE_BLOCKS`: `2 -> 0`

The rest of the base behavior stays the same:

- computes auxiliary MTP losses only during training,
- excludes `mtp_heads.*` tensors from the exported artifact,
- reconstructs the eval model with `mtp_num_heads=0` for roundtrip validation and TTT.

## How to run

From this candidate directory:

```bash
RUN_ID=mtp_aux_heads \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
TTT_ENABLED=1 \
TTT_FREEZE_BLOCKS=0 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=1536 \
XSA_LAST_N=4 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
ROPE_DIMS=16 \
LN_SCALE=1 \
LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected behavior

- During training, the model optimizes the normal next-token loss plus a small auxiliary loss from two future-token heads.
- During export/eval, the candidate still serializes only the main model parameters, so the artifact remains comparable to the base record.
- If MTP helps, the expected win should show up primarily in **pre-TTT quality**, with a smaller but hopefully still positive effect after legal TTT.

## Main risks and tradeoffs

- MTP adds some training compute; if the extra objective slows steps enough, the sample-efficiency gain may be cancelled out.
- The best setting may not be exactly `2 x 0.15`; a single head or a lower weight may work better for this tiny-model regime.
- Because the current best stack already uses TTT, some of MTP's gain may overlap with what TTT already recovers at eval time.
- The strongest remaining bottleneck may still be quantization/export rather than pre-quant model quality, in which case MTP could underdeliver.

## Validation

Commands run locally for this candidate:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604052245_mtp-aux-heads/train_gpt.py
python - <<'PY'
from pathlib import Path
text = Path("candidates/202604052245_mtp-aux-heads/train_gpt.py").read_text(encoding="utf-8")
required = [
    'for head in base_model.mtp_heads:',
    'scalar_params.append(head.weight)',
    'if "mtp_heads" not in k',
    'mtp_num_heads=0, mtp_loss_weight=0.0',
]
missing = [needle for needle in required if needle not in text]
if missing:
    raise SystemExit(f"missing MTP wiring markers: {missing}")
print("static MTP wiring checks passed")
PY
```

Outcome:

- `compileall`: success
- static MTP wiring check: success
- Minimal CPU-only smoke test: **not feasible in this workspace** without adding extra scaffolding, because this script assumes the challenge dataset/tokenizer layout, CUDA execution, and the record stack's FlashAttention-style runtime dependencies
