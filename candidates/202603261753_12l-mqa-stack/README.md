# 12L MQA stack

## Hypothesis

A 12-layer model with **1 KV head (MQA-style attention)** can be close to parameter-neutral relative to the current 11-layer / 4-KV-head stack, letting this repository trade redundant KV projections for one extra transformation step without changing the tokenizer, training pipeline, or export path.

## Why this looks promising here

Recent records in this repo already appear saturated on the usual late-stage knobs: sliding-window evaluation, EMA/SWA, GPTQ-lite or mixed low-bit export, SmearGate, BigramHash, XSA on deep layers, partial RoPE, and legal TTT. External research suggests **shared or reduced K/V structure is a strong efficiency lever**, and local history shows no prior run that explicitly tries the simple `12L + 1KV` trade.

This is appealing for this challenge because the saved K/V projection budget is roughly the cost of adding one more 512d / 3x-MLP block. In other words, it is one of the few genuinely new architecture moves that is still easy to implement as a clean fork of an already-strong script.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Most relevant preceding lineage:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`

Those records establish the current winning stack: 11-ish layers, 3x MLP, seq 2048, SmearGate + BigramHash, deep-layer XSA, partial RoPE, EMA/SWA, and aggressive low-bit export. This candidate keeps that stack intact and only changes the **depth vs. KV-head allocation** by default.

## External research

- Shazeer, **Fast Transformer Decoding: One Write-Head is All You Need** (2019), `https://arxiv.org/abs/1911.02150`
  - Multi-Query Attention shows that sharing K/V across query heads is a strong efficiency lever with modest quality loss.
- Ainslie et al., **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (2023), `https://arxiv.org/abs/2305.13245`
  - Intermediate or shared KV structure can recover most of multi-head quality while keeping the efficiency gains of MQA-like designs.
- Lan et al., **ALBERT** (2019), `https://arxiv.org/abs/1909.11942`
  - Reinforces the broader principle that removing redundant transformer parameters can be a good trade when model size is fixed.
- DeepSeek-AI, **DeepSeek-V2** (2024), `https://arxiv.org/abs/2405.04434`
  - A more modern example that KV compression remains an active frontier for efficient transformer design.

## What changed vs. the chosen base implementation

This directory starts from the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` script and keeps its training/export machinery, including:

- parameter banking + Parallel Muon,
- LeakyReLU(0.5)^2 MLP,
- SmearGate + BigramHash,
- deep-layer XSA,
- partial RoPE,
- VE support,
- mixed low-bit export and score-first TTT support.

Intentional changes in this candidate:

- default `NUM_LAYERS` changed from `11` to `12`
- default `NUM_KV_HEADS` changed from `4` to `1`
- default `VE_LAYERS` moved from `9,10` to `10,11` so value embeddings still target the deepest two layers
- inherited `BIGRAM_VOCAB_SIZE=2048` is left unchanged so the default script isolates the depth/KV-head trade
- added a **FlashAttention import fallback** that uses `torch.nn.functional.scaled_dot_product_attention` when FlashAttention is unavailable, which makes CPU-side import/smoke validation possible in environments that already have `torch` but not FlashAttention

## How to run / evaluate

Main candidate run:

```bash
NUM_LAYERS=12 NUM_KV_HEADS=1 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=10,11 \
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

Useful follow-up sweeps:

```bash
NUM_LAYERS=12 NUM_KV_HEADS=2 ...
NUM_LAYERS=11 NUM_KV_HEADS=1 ...
```

Those two ablations separate the effect of **MQA itself** from the effect of **reinvesting the saved parameters into an extra layer**.

## Validation run for this candidate

Commands executed successfully in this workflow:

```bash
python -m compileall candidates/202603261753_12l-mqa-stack/train_gpt.py
python - <<'PY'
def params(num_layers, num_kv_heads, model_dim=512, num_heads=8, mlp_mult=3,
           vocab_size=1024, bigram_vocab_size=2048, bigram_dim=128,
           ve_dim=128, ve_layers=2):
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    mlp_dim = int(mlp_mult * model_dim)
    total = vocab_size * model_dim
    total += bigram_vocab_size * bigram_dim + bigram_dim * model_dim + 1
    total += model_dim
    total += min(num_layers // 2, num_layers - num_layers // 2) * model_dim
    total += (2 * num_layers) * model_dim * model_dim
    total += (2 * num_layers) * kv_dim * model_dim
    total += num_layers * mlp_dim * model_dim
    total += num_layers * model_dim * mlp_dim
    total += num_layers * (4 * model_dim + num_heads)
    ve_target_dim = kv_dim
    total += vocab_size * ve_dim + ve_dim * ve_target_dim + 1 + ve_layers
    return total

base = params(11, 4, bigram_vocab_size=2048)
candidate = params(12, 1, bigram_vocab_size=2048)
print(base, candidate, candidate - base)
PY
```

Attempted but blocked runtime smoke test:

```bash
python - <<'PY'
import importlib.util
import torch
from pathlib import Path
path = Path('candidates/202603261753_12l-mqa-stack/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

This container's Python environment does not have `torch` installed, so a real import/forward smoke test was **not feasible here without networked dependency installation**.

Observed outcomes:

- `compileall`: passed
- parameter-budget sanity check: the `12L / 1KV / BIGRAM=2048` configuration is slightly **smaller** than the corresponding `11L / 4KV / BIGRAM=2048` baseline under a direct formula count (`-22,008` parameters)
- CPU runtime smoke test: **not run**, because local Python lacked `torch`

## Main risks / tradeoffs

- **Training-from-scratch MQA risk**: 1 KV head may lose too much quality relative to 4 KV heads even if parameter count is roughly preserved.
- **Wallclock risk**: the extra layer may consume most of the saved attention cost, so 12L/1KV could end up neutral or slightly slower in practice.
- **Interaction risk**: XSA, VE, and score-first TTT all survived the KV-head change in shape logic here, but their quality impact with MQA specifically is still untested.
- **Artifact risk**: the hypothesis is approximately parameter-neutral, not guaranteed byte-neutral after quantization/compression; `1KV` and `2KV` should both be checked.
