# Forward MTP Curriculum on the LeakyReLU² + Legal TTT Stack

## Hypothesis

The current best stack in this repository still trains with plain next-token prediction even though the codebase already contains dormant multi-token prediction (MTP) support. This candidate turns MTP into a real training signal and stages it with a **forward curriculum**: stay pure next-token early, add one auxiliary future-token head mid-run, and only unlock the full 2-head MTP objective late in the 10-minute budget.

The goal is to improve **sample efficiency** in the fixed-wallclock regime without paying any artifact-size cost, because the auxiliary MTP heads are still excluded from export and evaluation.

## Why this is promising here

This repository's winning trend is clear: once the quantization gap was mostly solved, the remaining frontier moved toward techniques that extract more quality from the same 600-second budget.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed the 11-layer EMA/GPTQ-lite line had already converged on a strong training-time-efficient architecture.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` then pushed that line further with LeakyReLU², parameter banking, and legal score-first TTT to `1.1194` mean BPB.
- The same family of scripts already carries MTP code paths and export-time exclusion of `mtp_heads`, but recent record READMEs never enabled MTP, and the parameter-banked script had stopped wiring the MTP heads into any optimizer.

That makes MTP curriculum a strong fit: it is local to `train_gpt.py`, artifact-neutral, and directly targeted at the repo's main bottleneck of **quality per unit training time**.

## Prior repository work that influenced this candidate

1. **Best base to fork:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - Best reported mean score in the repository (`1.1194` post-TTT BPB).
   - Already contains the current winning architectural stack plus optional TTT.
2. **Immediate predecessor stack:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - Confirms EMA + GPTQ-lite + partial RoPE/LN-scale/XSA/VE is still the strongest training-side base before TTT.
3. **Longer trend across records**
   - 11 layers, seq 2048, compression-aware training, EMA/SWA, SmearGate/BigramHash, partial RoPE, and evaluation-aware methods consistently helped.
   - Weight sharing / recurrence was explicitly reported as a dead end in short wallclock runs, which makes MTP a better next step than another recurrent-depth experiment.

## External research that informed it

1. **Better & Faster Large Language Models via Multi-token Prediction** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737))
   - Argues that predicting multiple future tokens improves sample efficiency with no training-time infrastructure change beyond extra heads.
   - Also reports that MTP helps induction-head style behavior, which is attractive for the compact, context-sensitive models in this repo.
2. **Pre-Training Curriculum for Multi-Token Prediction in Language Models** ([arXiv:2505.22757](https://arxiv.org/abs/2505.22757))
   - The most important paper for this candidate.
   - Shows that **small language models struggle with naive MTP**, and that a **forward curriculum** from NTP to MTP is the better way to get the benefit in SLM regimes.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes six focused changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Add a wallclock-aware forward MTP curriculum**
   - `0` auxiliary heads before `MTP_START_FRAC=0.25`
   - `1` head through the middle of training
   - full `2` heads by `MTP_FULL_FRAC=0.75`
3. **Fix optimizer wiring for MTP heads**
   - The parameter-banked base still defined `mtp_heads`, but did not add them to any optimizer param group.
   - This candidate explicitly optimizes them with the scalar AdamW path.
4. **Keep export artifact-neutral**
   - `mtp_heads` are still excluded from the exported state dict and the quantized evaluation model is rebuilt with `mtp_num_heads=0`.
5. **Add a FlashAttention fallback**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA.
   - This makes CPU import/smoke validation possible even though the full `main()` training entrypoint still expects CUDA plus the real dataset shards.
6. **Bake in the strongest recent base defaults**
   - `BIGRAM_VOCAB_SIZE=1536`
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`

## How to run

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Remove MTP entirely, keep the same TTT-capable base
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep MTP, but remove the curriculum
MTP_CURRICULUM=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Inspect the training-side effect without the longer TTT eval
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key candidate-specific knobs:

| Variable | Default | Purpose |
|---|---:|---|
| `MTP_NUM_HEADS` | `2` | number of auxiliary future-token heads |
| `MTP_LOSS_WEIGHT` | `0.15` | overall auxiliary loss scale |
| `MTP_CURRICULUM` | `1` | enable staged NTP -> MTP training |
| `MTP_START_FRAC` | `0.25` | fraction of wallclock/iteration progress before MTP begins |
| `MTP_FULL_FRAC` | `0.75` | fraction by which all MTP heads are active |

## Evaluation notes

- Standard post-quant metric still comes from the exported NTP model (`final_int6_roundtrip*` log lines).
- With `TTT_ENABLED=1`, the script also runs the same legal score-first TTT evaluation used by the current best record and prints `legal_ttt*` log lines.
- Because the export path removes `mtp_heads`, this candidate should be compared on the same artifact-budget footing as the current record stack.

## Validation run for this candidate

### Commands

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604050728_forward-mtp-curriculum/train_gpt.py

/tmp/gh-aw/agent/pg-venv/bin/python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path("candidates/202604050728_forward-mtp-curriculum/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.GPT(
    vocab_size=128,
    num_layers=2,
    model_dim=64,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.005,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
    mtp_num_heads=2,
    mtp_loss_weight=0.15,
    bigram_vocab_size=0,
    bigram_dim=128,
    xsa_last_n=0,
    rope_dims=16,
    ln_scale=True,
    dtg=False,
    ve_enabled=False,
    ve_dim=128,
    ve_layers="9,10",
    gated_attention=False,
    value_residual=False,
)
model.train()
model.active_mtp_heads = 2
x = torch.randint(0, 128, (2, 16), dtype=torch.int64)
y = torch.randint(0, 128, (2, 16), dtype=torch.int64)
loss = model(x, y)
loss.backward()
assert all(p.grad is not None for p in model.mtp_heads.parameters())
model.zero_grad(set_to_none=True)
model.active_mtp_heads = 0
assert torch.isfinite(model(x, y))
print("candidate smoke ok")
PY
```

### Outcomes

- `compileall` completed successfully for the existing Python entrypoints plus this candidate.
- The CPU smoke test imported the candidate without FlashAttention, exercised the SDPA fallback, ran forward/backward with `active_mtp_heads=2`, and confirmed the MTP heads now receive gradients.
- The same smoke test then switched to `active_mtp_heads=0` and produced a finite next-token-only loss.

No full end-to-end training launch was run in this workflow because the repository's normal path expects the real FineWeb shards plus the CUDA/H100 training environment.

## Main risks / tradeoffs

1. **Training overhead vs step count**
   - Even with a curriculum, MTP adds compute.
   - If step-time inflation is too large, the extra sample-efficiency signal may be offset by fewer total optimizer steps.
2. **Curriculum thresholds may need tuning**
   - Small models can be sensitive to when MTP starts.
   - If the auxiliary task still destabilizes early training, push `MTP_START_FRAC` later or reduce `MTP_LOSS_WEIGHT`.
3. **Compile-stage transitions**
   - Changing `active_mtp_heads` can trigger a small number of graph recompiles during the run.
4. **Interaction with TTT is unproven**
   - The hypothesis is that better pre-TTT weights should help both the standard exported model and the later TTT pass, but that should be measured explicitly.
