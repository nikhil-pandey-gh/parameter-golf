# Train-only MTP on the LeakyReLU2 + Legal TTT stack

## Hypothesis

Add a small number of **training-only multi-token prediction (MTP) heads** to the current best stack so the shared trunk learns richer short-horizon structure within the same 10-minute budget, then **prune those auxiliary heads before export** so the final artifact stays on the same 16MB footing as the existing records.

## Why this is promising for this repository

The repo trend is already very saturated on:

- eval-only gains (stride-64 sliding eval, legal TTT),
- compression tuning (int6/int8 mixes, GPTQ-lite clip search, late-QAT variants),
- lightweight architectural biasing (XSA, partial RoPE, LN scale, SmearGate, BigramHash, VE).

That makes **training-side sample efficiency** the most attractive remaining lever. MTP fits especially well here because:

1. it reuses the current trunk instead of adding a new architecture,
2. the repo already had export-time pruning support for `mtp_heads`,
3. the added parameters only exist during training, so the artifact budget impact is near-zero after export,
4. the strongest recent stacks already have enough compression headroom discipline that a training-only auxiliary objective is more attractive than another quantization sweep.

I also rejected a rotation-assisted PTQ follow-up after research: ideas like QuaRot/SpinQuant are interesting, but this codebase's residual/skip/control-tensor structure would make function-preserving rotation folding much more invasive than the training-only MTP path.

## Prior repository work that influenced this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - strongest overall score,
  - best current training/eval stack,
  - already prunes `mtp_heads` from the exported artifact.
- **Important implementation reference:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - clean non-TTT base,
  - still routed `mtp_heads` into the matrix optimizer path before the later parallel-Muon refactor.
- **Structural background:** the 2026-03-20 through 2026-03-23 records established the main winning recipe: 11L/512d, MLP 3x, XSA on deep layers, partial RoPE, LN scale, BigramHash, SmearGate, EMA/SWA, and compression-aware export.

There were **no prior `candidates/` directories** in the repository at review time, so this candidate is the first entry in that namespace.

## External research that informed it

- **Better & Faster Large Language Models via Multi-token Prediction** (arXiv:2404.19737) argues that predicting multiple future tokens with independent auxiliary heads improves sample efficiency and downstream quality while sharing the same backbone.
- **Medusa** (arXiv:2401.10774) shows that extra decoding heads can be trained on top of a shared backbone and are useful enough to justify head-specific supervision.
- **DeepSeek-V3 Technical Report** (arXiv:2412.19437) explicitly reports using a multi-token prediction objective for stronger performance in a modern large-scale system.

Those papers point in the same direction: **extra future-token heads can improve the trunk even when the heads themselves are not part of the final deployed model**.

## What changed vs the chosen base implementation

Starting from the **codebase** in `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes three focused changes:

1. **Enable MTP by default** with `MTP_NUM_HEADS=2`.
2. **Actually optimize the MTP heads** under the parallel-Muon matrix path.
   - In the 2026-03-23 code, `mtp_heads` existed and were used in the loss, but they were not routed into any optimizer after the parameter-banking refactor.
   - This candidate restores a real optimizer path for them.
3. **Add a safe attention fallback** to PyTorch SDPA when `flash_attn_interface` is unavailable, which makes import-level smoke checks feasible in torch-enabled environments without changing the GPU fast path.

The export behavior stays aligned with the base record:

- `mtp_heads` are **excluded from `export_sd`**,
- the quantized eval model is instantiated with `mtp_num_heads=0`,
- so the candidate still evaluates/export as a normal single-head model.

## How to run or evaluate it

From the candidate directory, point the script back at the repo-root tokenizer and dataset paths:

```bash
cd candidates/202604051012_train-only-mtp
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

That command is the recommended **pure training-side MTP ablation**. The script still contains the 2026-03-23 legal-TTT path, so a second combined run is:

```bash
cd candidates/202604051012_train-only-mtp
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 \
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 TTT_LR=0.002 TTT_EPOCHS=3 \
TTT_CHUNK_TOKENS=32768 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful comparison sweeps (with the same `DATA_PATH=...` and `TOKENIZER_PATH=...` prefix as above):

```bash
# Same candidate code path, but disable MTP
MTP_NUM_HEADS=0 TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Heavier auxiliary supervision sweep
MTP_NUM_HEADS=4 MTP_LOSS_WEIGHT=0.2 TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main risks and tradeoffs

- **Step-time overhead:** extra vocab heads add logits and CE work, so the gain from better supervision must outweigh any loss in completed steps.
- **Short-horizon bias:** MTP might overemphasize very local prediction and partly overlap with BigramHash/SmearGate.
- **Optimizer interaction:** parallel Muon now updates the auxiliary heads too, which is reasonable but not yet empirically tuned here.
- **TTT interaction uncertainty:** legal TTT may amplify or wash out the training-side gain, so both `TTT_ENABLED=0` and `TTT_ENABLED=1` are worth checking.

## Validation

Commands run and outcomes for this candidate:

- `python -m compileall candidates/202604051012_train-only-mtp/train_gpt.py` -> **passed**
- `python - <<'PY' import importlib.util; print({'torch': importlib.util.find_spec("torch"), 'sentencepiece': importlib.util.find_spec("sentencepiece")}) PY` -> `{'torch': None, 'sentencepiece': None}`
- Tiny import/forward smoke test -> **not feasible in this workflow environment because Python does not have the required runtime deps (`torch`, `sentencepiece`) installed**
