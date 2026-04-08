# Reverse-Curriculum MTP on the LeakyReLU² + Legal TTT stack

## Hypothesis

A tiny language model can benefit from **training-only multi-token prediction (MTP)** without paying any artifact-size penalty, but small models are more likely to overfit the auxiliary objective if it stays on for the whole run. The bet here is:

1. use a small MTP objective early, when sample-efficiency gains matter most;
2. **turn it off for the final stretch** so the trunk finishes on the true next-token objective;
3. export only the main model, not the auxiliary heads.

In short: **MTP early, pure next-token late**.

## Why this is promising for this repository

- The record stack has already mined a lot of value out of evaluation tricks, quantization, lexical side channels, and optimizer/export tuning.
- The repo history still leaves **training objectives** relatively underexplored compared with quantization and eval.
- This idea is cheap in artifact bytes because the extra MTP heads are **training-only** and are dropped before export.
- The repo's own negative result on naive depth recurrence suggests a safer next bet is an objective change that keeps the main architecture and runtime structure mostly intact.

## Prior experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the direct base: 11-layer banked stack, LeakyReLU(0.5)^2, legal score-first TTT, EMA, partial RoPE, XSA, BigramHash, and parallel Muon.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` reinforced that the best current runs already heavily optimize quantization/export, so the next candidate should look elsewhere for gains.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` documented a `torch.compile` constant-folding failure for runtime feature flags; that pushed this candidate toward **separate compiled loss paths** instead of trying to toggle MTP inside one compiled graph.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` recorded a clear negative result for naive recurrence, which made shared-block depth reuse a weaker immediate bet for this repo's 10-minute budget.
- No prior `candidates/` directory existed when this candidate was created.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737). Main takeaway used here: predicting multiple future tokens can improve sample efficiency and strengthen useful algorithmic/induction-like behavior in the shared trunk.
- Ansar Aynetdinov and Alan Akbik, **"Pre-Training Curriculum for Multi-Token Prediction in Language Models"** (arXiv:2505.22757). Main takeaway used here: **small language models benefit more from curriculum MTP than from a fixed always-on MTP objective**, and reverse-curriculum training can improve final next-token quality.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default** with two auxiliary heads.
2. Add **horizon decay** (`MTP_HEAD_DECAY=0.5`) so the nearest future token gets the largest weight.
3. Add a **reverse curriculum** (`MTP_DISABLE_FRAC=0.60`): MTP is active for the first 60% of training progress, then the script switches to pure next-token loss for the final 40%.
4. Use **two compiled loss paths**:
   - one compiled `forward()` path with MTP enabled,
   - one compiled `forward_main_loss()` path without MTP.
   
   This avoids relying on mutable runtime flags inside a single compiled graph.
5. **Wire MTP head weights into AdamW**. The banked PR #549-style script carried MTP hooks, but they were not wired into an optimizer in this variant; this candidate fixes that so the auxiliary heads actually train.
6. Keep exports artifact-safe by **excluding `mtp_heads` from the saved/exported state dict**. The quantized eval model is still instantiated with `mtp_num_heads=0`.
7. Set defaults closer to the current best public stack by defaulting to:
   - `BIGRAM_VOCAB_SIZE=1536`
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`

## Why this idea over the other researched options

| Option | Why it was considered | Why it was not chosen here |
|---|---|---|
| Shared-block / recurrent depth | Strong fit for the 16MB budget in theory | The repo already has a negative recurrence result under the short wallclock budget, so it looked riskier for the next immediate candidate |
| More QAT / export tuning | Strong literature and already proven here | The records have already pushed this axis hard; likely smaller incremental gain |
| Tokenizer or embedding redesign | Potentially large upside | Higher infrastructure and validation burden than a surgical objective change |
| **Reverse-curriculum MTP** | Training-only, artifact-cheap, literature-backed for small models | **Chosen** because it attacks an underexplored surface while preserving the current winning stack |

## How to run

From the repository root:

```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 \
  candidates/202604080517_reverse-mtp-curriculum/train_gpt.py
```

Useful overrides:

```bash
SEED=1337 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_HEAD_DECAY=0.5 \
MTP_DISABLE_FRAC=0.60 \
torchrun --standalone --nproc_per_node=8 \
  candidates/202604080517_reverse-mtp-curriculum/train_gpt.py
```

The script still emits the same main outputs as the base stack:

- standard validation loss / BPB,
- post-EMA quantized roundtrip metrics,
- sliding-window metrics,
- legal TTT metrics when enabled.

## Main expected risks and tradeoffs

- **Training cost risk:** even training-only MTP heads add extra matmuls and CE work; the gain must outweigh any lost training steps.
- **Curriculum threshold risk:** `MTP_DISABLE_FRAC=0.60` is a strong but still heuristic default.
- **Small-model sensitivity:** the 2025 curriculum paper exists precisely because smaller models can struggle with always-on MTP.
- **No free lunch from export:** the auxiliary heads are excluded from export, so any win has to come from a better trunk, not from keeping extra prediction heads at eval time.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202604080517_reverse-mtp-curriculum/train_gpt.py
python -m compileall train_gpt.py train_gpt_mlx.py data
```

Outcomes:

- `candidates/202604080517_reverse-mtp-curriculum/train_gpt.py`: **passed** syntax compilation.
- Root baseline syntax check (`train_gpt.py`, `train_gpt_mlx.py`, `data/`): **passed**.
- Minimal CPU smoke test: **not run**. This environment does not have the PyTorch runtime installed, and the script is CUDA-only; installing the full runtime stack would have exceeded the intended lightweight validation scope.
