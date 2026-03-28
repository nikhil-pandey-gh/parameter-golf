# Forward-Curriculum MTP on the LeakyReLU² + Legal TTT + Parallel Muon stack

## Hypothesis

Enable the already-wired training-only multi-token prediction (MTP) heads on top of the current best stack, but do it with a **forward curriculum** tailored to small models.

The repo's strongest lineage already gets most of its gains from efficient training, evaluation, and quantization tricks rather than from larger artifacts. MTP is attractive here because the extra heads are **excluded from export**, so the candidate can spend more training compute to improve sample efficiency without paying model bytes at evaluation time.

## Why this is promising for this repository

Three repository facts point in the same direction:

1. The current best result is the 2026-03-23 `LeakyReLU_LegalTTT_ParallelMuon` stack, which already contains dormant MTP support but keeps `mtp_num_heads=0` in all published runs.
2. The strongest non-TTT branch (`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`) shows the repo is already very close to squeezing out improvements from quantization and averaging alone.
3. Recent records keep stacking training-time efficiency ideas that do **not** increase exported bytes, so training-only auxiliary heads fit the challenge well.

## Prior records or candidates that influenced this

There were **no prior `candidates/` directories** in the repository when this candidate was created.

The main implementation lineage is:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`

Those runs established the current winning recipe: 11 layers, partial RoPE, XSA in deep layers, EMA / SWA-style averaging, aggressive quantization, and legal score-first TTT.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737). The paper argues that predicting multiple future tokens with independent auxiliary heads improves sample efficiency while preserving a shared trunk.
- DeepSeek-AI et al., **"DeepSeek-V3 Technical Report"** (arXiv:2412.19437). DeepSeek-V3 explicitly includes an MTP training objective in a modern large-model recipe, which is strong evidence that MTP can stack with other optimization techniques.
- Ansar Aynetdinov and Alan Akbik, **"Pre-Training Curriculum for Multi-Token Prediction in Language Models"** (arXiv:2505.22757). This is the key small-model motivation: they report that smaller language models benefit much more from a **forward curriculum** than from turning full MTP on immediately.
- Guoliang Zhao et al., **"Self-Distillation for Multi-Token Prediction"** (arXiv:2603.23911). I did not implement self-distillation here, but it reinforces that MTP is still an active and improving line of work rather than a dead end.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes four surgical changes:

1. **Turns MTP on by default** with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`.
2. **Adds a forward curriculum** for the auxiliary heads. The first MTP head ramps in after a warmup period, and the second head starts later. This follows the small-model curriculum lesson from arXiv:2505.22757.
3. **Fixes the optimizer wiring for MTP heads** in this Parallel-Muon branch. The MTP heads were already excluded from export, but in this branch they were not assigned to any optimizer group because all published runs used `mtp_num_heads=0`. This candidate trains them with scheduled AdamW optimizers that only start stepping once each head's curriculum weight becomes positive.
4. **Adds an explicit FlashAttention-3 -> SDPA fallback** for non-CUDA / non-FA3 smoke tests. The intended H100 path is unchanged; the fallback only exists so the script can be imported and sanity-checked in lighter environments.

## How to run or evaluate it

From the candidate directory. This candidate resolves its default dataset and tokenizer paths relative to the repository root, so the command below works without adding `DATA_PATH` / `TOKENIZER_PATH` overrides:

```bash
cd candidates/202603280946_forward-mtp

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 \
MTP_WARMUP_STEPS=1000 MTP_RAMP_STEPS=1600 MTP_HEAD_STAGGER_STEPS=800 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to isolate the effect of the candidate idea before spending time on TTT, the simplest ablation is to keep the same command but set `TTT_ENABLED=0` and compare pre-TTT roundtrip BPB against the base branch.

## Expected risks and tradeoffs

- **Extra training compute:** even though the auxiliary heads are not exported, they still add two extra vocab projections during training and can reduce steps-per-600s.
- **Small-model fragility:** this is exactly why the curriculum exists. If the warmup or ramp is too aggressive, the candidate could underperform the `mtp_num_heads=0` baseline.
- **Curriculum transition overhead:** the implementation keeps early training truly NTP-only by activating MTP heads in stages, which means `torch.compile` may need to retrace when the active head count changes.
- **No direct inference gain in this repo path:** unlike full speculative decoding pipelines, this candidate uses MTP only as a training objective. The win has to come from better trunk representations, not from faster evaluation.
- **TTT interaction uncertainty:** MTP may improve the base model before TTT, but the incremental gain could partially overlap with what legal TTT already recovers.

## Validation

The following validation commands were run for this candidate:

```bash
python -m compileall candidates/202603280946_forward-mtp/train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202603280946_forward-mtp/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_forward_mtp', path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

model = module.GPT(
    vocab_size=1024,
    num_layers=4,
    model_dim=128,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2.0,
    tie_embeddings=True,
    tied_embed_init_std=0.005,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
    mtp_num_heads=2,
    mtp_loss_weight=0.15,
    bigram_vocab_size=0,
    xsa_last_n=0,
    rope_dims=16,
    ln_scale=False,
    dtg=False,
    ve_enabled=False,
    ve_dim=64,
    ve_layers='',
    gated_attention=False,
    value_residual=False,
)
with torch.no_grad():
    model.mtp_head_weights.copy_(torch.tensor([0.5, 0.0], dtype=torch.float32))
input_ids = torch.randint(0, 1024, (2, 32))
target_ids = torch.randint(0, 1024, (2, 32))
loss = model(input_ids, target_ids)
print(f'cpu_smoke_loss={loss.item():.6f}')
PY
```

Results:

- `python -m compileall candidates/202603280946_forward-mtp/train_gpt.py` — **passed** on this workflow runner.
- The CPU smoke import / forward command was **attempted but blocked by the environment**, not by candidate code: this runner does not have `torch` installed, so Python exited with `ModuleNotFoundError: No module named 'torch'` before the smoke test could import the module. The candidate keeps the explicit FlashAttention-3 -> SDPA fallback so the same smoke command can run in a PyTorch-equipped environment without requiring FA3.
