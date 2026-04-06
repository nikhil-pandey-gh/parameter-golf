# Annealed training-only MTP on the LeakyReLU² + legal TTT stack

## Hypothesis

The strongest next low-risk idea is to turn on **training-only multi-token prediction (MTP)** for the current best stack, then **anneal that auxiliary loss away during warmdown** so the model finishes aligned to the true next-token objective used for quantized evaluation.

The bet is simple:

1. Early in training, one future-token head improves sample efficiency by giving the trunk denser supervision.
2. Because the extra head is excluded from export, the artifact budget stays effectively unchanged.
3. Late in training, decaying the MTP loss weight should reduce mismatch between the auxiliary objective and the final BPB metric.

## Why this is promising here

The repository history shows that most large wins are now already stacked: mixed quantization, wider MLPs, deeper 11-layer U-Net layouts, XSA, EMA, partial RoPE, GPTQ-lite, and legal TTT. The remaining promising space is ideas that improve learning **without permanently costing artifact bytes**.

This candidate fits that gap:

- the recent record stack already contains dormant MTP support,
- every reviewed record log still kept `MTP_NUM_HEADS=0`,
- and the exporter already strips `mtp_heads` before serialization.

That makes MTP one of the few ideas that can plausibly improve training efficiency without forcing a larger artifact or broad infrastructure changes.

## Prior experiments that influenced this candidate

There were no prior `candidates/` directories to build from, so this candidate is based on record evidence only.

Most relevant records:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` — best absolute score; chosen as the direct base.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — strongest non-TTT 11-layer stack and confirmation that the current 11L family is the right trunk.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` — evidence that cheap, zero- or near-zero-parameter training tweaks still move BPB.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` — useful negative result showing naive layer recurrence is a bad fit under the fixed 10-minute wall-clock cap.

## External research that informed it

1. **Gloeckle et al., “Better & Faster Large Language Models via Multi-Token Prediction”** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737)).  
   Main takeaway: auxiliary prediction of multiple future tokens improves sample efficiency and downstream behavior while keeping inference-time overhead optional.

2. **Zelikman et al., “Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking”** ([arXiv:2403.09629](https://arxiv.org/abs/2403.09629)).  
   Main takeaway: predicting farther-ahead text disproportionately helps difficult tokens. The full latent-thought machinery is too heavy here, but it reinforces the value of future-token supervision.

3. **Nakanishi et al., “Scalable-Softmax”** ([arXiv:2501.19399](https://arxiv.org/abs/2501.19399)).  
   Considered as another compact-model direction because it improves focus over long contexts, but it requires a broader attention change than this repository likely wants for a minimal next candidate.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate adds only a small twist on top of that script:

1. **Enable one training-only MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Anneal MTP during warmdown**
   - new env: `MTP_ANNEAL_SCALE` (default `0.25`)
   - when the LR scale drops below that threshold, the MTP loss weight decays linearly toward zero.

3. **Keep MTP excluded from the exported artifact**
   - the existing export path still drops `mtp_heads`, so the serialized model remains the no-MTP inference model.

4. **Make dynamic training toggles compile-safe**
   - the candidate uses runtime tensors for the annealed MTP coefficient
   - it also replaces the copied late-QAT class flag with per-module tensor toggles so warmdown activation is not relying on a compile-time Python boolean

5. **Add a FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, attention falls back to `torch.nn.functional.scaled_dot_product_attention`
   - this is primarily to make a minimal local CPU smoke test feasible; the intended fast path remains FlashAttention on GPU.

## How to run

From the candidate directory:

```bash
cd candidates/202604062047_annealed-mtp

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 MTP_ANNEAL_SCALE=0.25 \
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

Note: this script already keeps EMA active internally; unlike some earlier record folders, there is no separate `EMA_ENABLED` toggle.

For a minimal local smoke import on CPU, use the validation command below instead of full training.

## Expected risks and tradeoffs

- **Throughput risk:** even one MTP head adds another vocab projection during training, so the step count may drop enough to erase the gain.
- **Objective mismatch risk:** MTP is not the leaderboard objective, which is why this candidate decays the auxiliary loss late.
- **Compilation risk:** dynamic training-side toggles can be brittle under `torch.compile`; this candidate uses a runtime tensor weight for the MTP coefficient instead of a compile-time Python flag.
- **Small-win risk:** this may be a marginal gain rather than a leaderboard jump, because the rest of the stack is already highly optimized.

## Validation

Commands run:

```bash
python -m compileall candidates/202604062047_annealed-mtp/train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path("candidates/202604062047_annealed-mtp/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_annealed_mtp", path)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

model = mod.GPT(
    vocab_size=64,
    num_layers=2,
    model_dim=32,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.01,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    mtp_num_heads=1,
    mtp_loss_weight=0.15,
).float()
tokens = torch.randint(0, 64, (2, 16))
loss = model(tokens, tokens)
print(f"cpu_smoke_loss={loss.item():.4f}")
PY
```

Outcomes:

- `compileall`: **passed**
- CPU smoke forward: **not runnable in this workflow environment because the Python environment does not have `torch`, `numpy`, or `sentencepiece` installed** (`importlib.util.find_spec(...) -> None` for all three). The smoke snippet above is the intended minimal check once those dependencies are available.

## Suggested next experiments if this works

1. Try `MTP_NUM_HEADS=2` only if the step-time hit is small.
2. Sweep `MTP_LOSS_WEIGHT` in `{0.10, 0.15, 0.20}`.
3. Sweep `MTP_ANNEAL_SCALE` in `{0.20, 0.25, 0.33}`.
4. If the best no-TTT score improves, retest with legal TTT enabled on top.
