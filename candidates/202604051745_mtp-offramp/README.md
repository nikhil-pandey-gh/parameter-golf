# MTP Off-ramp on the LeakyReLU² + Legal TTT Stack

## Hypothesis

Training-only multi-token prediction (MTP) heads should improve sample efficiency inside the same 600s training budget by forcing the shared trunk to model slightly longer horizons, while a late warmdown off-ramp should hand the last stretch of optimization back to the main next-token objective before EMA, quantization, and TTT evaluation.

The key repository-specific attraction is that the strongest current scripts already exclude `mtp_heads.*` from export, so this auxiliary objective can add training signal with **zero artifact-byte cost**.

## Why this is promising for this repository

- The frontier has already mined most of the obvious no-byte gains: sliding eval, EMA/SWA, partial RoPE, LN scale, XSA, VE, LeakyReLU(0.5)^2, and GPTQ-lite-style quantization.
- Prior records repeatedly show that extra artifact bytes are expensive, while training-time-only machinery is much easier to justify under the 16MB cap.
- The strongest recent scripts implement MTP plumbing and export stripping, but every logged record I reviewed still ran with `mtp_num_heads:0`, so this is a real gap rather than a repeated idea.
- Earlier dead ends in the repo were mostly about **compute-heavy** changes (recurrence, always-on QAT, slower activations). MTP is a lighter intervention than those and keeps evaluation unchanged.

## Prior record influence

- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`** supplied the chosen base stack: LeakyReLU(0.5)^2, parameter banking + parallel Muon, legal score-first TTT, and the current best overall recipe.
- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`** established the 11-layer XSA4 + partial RoPE + LN scale + VE128 + EMA/GPTQ-lite core that the latest record builds on.
- **`records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`** and related quantization-focused runs reinforced the importance of byte-neutral or export-aware ideas over brute-force architectural growth.
- No prior `candidates/` directory existed in this repository when this candidate was created.

## External research that informed it

| Work | Why it matters here |
|---|---|
| Fabian Gloeckle et al., [*Better & Faster Large Language Models via Multi-token Prediction*](https://arxiv.org/abs/2404.19737) | Introduces MTP as an auxiliary objective on a shared trunk and reports better sample efficiency with no inference-cost requirement to get the training benefit. |
| DeepSeek-AI et al., [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437) | A strong modern recipe that explicitly keeps an MTP objective in the training stack for stronger performance, which is a good signal that the objective is not just for decoding speedups. |
| Guoliang Zhao et al., [*Self-Distillation for Multi-Token Prediction*](https://arxiv.org/abs/2603.23911) | Emphasizes the need to preserve main-head quality while using MTP. That directly motivated the challenge-specific warmdown off-ramp here, even though this candidate does not implement their full self-distillation method. |

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

1. **Enabled training-only MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Added an MTP warmdown off-ramp**
   - New env: `MTP_OFFRAMP_SCALE` (default `0.2`)
   - The auxiliary MTP weight now linearly decays to zero once the main LR multiplier falls below that scale.
   - This keeps early-horizon supervision during the fast-learning phase, then removes it late so the main head can specialize before export/eval.
3. **Made candidate-local execution sane**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve by walking upward to find the repository root (`.git` or `data/` + `README.md`), so the script still works if this recipe is copied into another nested experiment folder later.
4. **Added explicit logging**
   - Logs now print `mtp_offramp_scale` and the current `mtp_scale` during training.

The rest of the stack is intentionally unchanged so the MTP effect stays easy to isolate: LeakyReLU², XSA, partial RoPE, VE, legal TTT, parallel Muon, EMA/SWA behavior, and the int6+lzma export path all stay intact.

## How to run / evaluate

From this directory:

```bash
cd candidates/202604051745_mtp-offramp
SEED=1337 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_OFFRAMP_SCALE=0.20 \
TTT_ENABLED=1 \
TTT_LR=0.002 \
TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 \
TTT_MOMENTUM=0.9 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a cleaner ablation of the training-side idea alone, run the same command with `TTT_ENABLED=0`.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202604051745_mtp-offramp/train_gpt.py
python - <<'PY'
import importlib.util
print(importlib.util.find_spec("torch"))
PY
```

Outcomes:

- `python -m compileall ...` **succeeded**.
- The workflow container reported `None` for `importlib.util.find_spec("torch")`, so a real runtime smoke test was **not feasible in this environment**.
- A proper CPU/GPU startup smoke requires the repository's normal training dependencies, especially PyTorch (and for the full fast path, CUDA/FlashAttention support).

## Main risks and tradeoffs

- **Training-time overhead:** extra MTP heads increase forward/backward work and optimizer state. If step throughput drops too much, sample-efficiency gains can be erased.
- **Small-model uncertainty:** the strongest public MTP evidence is from larger models; this 16MB regime may benefit less.
- **Off-ramp tuning risk:** `MTP_OFFRAMP_SCALE=0.2` is a reasoned starting point, not a proven optimum. Sweeping `0.1-0.3` and `1-3` heads is the obvious next tuning pass.
- **No direct eval-time gain:** since MTP heads are stripped from export, this idea only helps if it improves the shared trunk enough to survive quantization and TTT.
