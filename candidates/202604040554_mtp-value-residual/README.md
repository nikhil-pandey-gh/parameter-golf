# Candidate: MTP + Value Residual on the LeakyReLU² Parallel-Muon stack

## Hypothesis

The current best record stack already squeezes a lot out of evaluation, quantization, and small late-layer attention tweaks. The next cheap lever is a **training-only lookahead objective**: enable a single multi-token-prediction (MTP) head so the shared trunk learns short-horizon future structure, then **drop that head at export time** so the artifact budget is unchanged. I paired that with the script's dormant **value-residual** path because it is almost free in parameters and matches the repo's recent trend toward small, deep attention/value-path interventions instead of wider blocks.

## Why this is promising here

This candidate starts from the strongest local base, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, which already combines LeakyReLU(0.5)^2, parameter banking + Parallel Muon, XSA on the deepest 4 layers, partial RoPE, shared value embeddings, EMA/SWA, mixed int6 export, and legal score-first TTT.

The records review suggests three things:

1. The big easy wins from sliding eval, fp16 embeddings, and plain quantization tuning have mostly already been harvested.
2. The strongest later improvements came from **cheap representational changes** layered onto the same 11-layer backbone.
3. Training-only extras are still underexplored in the winning lineage, even though this codebase already has an export-safe MTP path.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` — chosen base implementation and evaluation recipe.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` — reinforced the value of tiny post-base refinements on the same 11L family.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` and `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md` — showed that late-layer attention tweaks and near-free positional changes continue to matter.

## External research that informed it

- **Multi-Token Prediction via Self-Distillation** (`arXiv:2602.06019`) argues that online distillation can turn a next-token LM into a standalone multi-token predictor without auxiliary verifier infrastructure.
- **Self-Distillation for Multi-Token Prediction** (`arXiv:2603.23911`) reports that even 1-head MTP can materially improve MTP-head acceptance while largely preserving the main head, suggesting a low-cost auxiliary objective is worth trying before widening the model.
- **Exclusive Self Attention** (`arXiv:2603.09078`) and **Diff Transformer** (`arXiv:2410.05258`) both support the broader repo pattern that better context selection and noise suppression in attention are fruitful even when the change is localized and cheap.

## What changed vs. the chosen base

Compared with the 2026-03-23 record script, this candidate:

1. Enables **1-head MTP by default** with `MTP_NUM_HEADS=1` and `MTP_LOSS_WEIGHT=0.15`.
2. Enables the dormant **value-residual** path by default with `VALUE_RESIDUAL=1`.
3. Uses `BIGRAM_VOCAB_SIZE=1536` by default to match the chosen base run command rather than the copied script's broader 2048-bucket default.
4. Keeps the intended strong evaluation path on by default with `TTT_ENABLED=1` and `TTT_FREEZE_BLOCKS=0`.
5. Auto-discovers the repository root by walking up to the nearest ancestor containing `data/`, so the default dataset/tokenizer paths work from this candidate folder without depending on the current working directory.
6. Adds a **FlashAttention3 -> PyTorch SDPA fallback** when FA3 is unavailable or the CUDA device is pre-Hopper. Benchmark runs should still prefer the original Hopper/FA3 path.

The MTP heads are already excluded from export in the base script, so this keeps the main candidate idea training-only rather than artifact-expanding.

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

That uses the baked-in candidate defaults and expects the normal repository dataset layout under `data/`, which the script locates automatically from its own file path.

If your dataset or tokenizer lives elsewhere, override the paths explicitly:

```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a faster diagnostic that skips legal TTT during final evaluation:

```bash
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604040554_mtp-value-residual/train_gpt.py
python - <<'PY'
try:
    import torch
    print("torch_available")
except Exception as e:
    print(type(e).__name__ + ": " + str(e))
PY
```

Observed outcome:

- `compileall` succeeded.
- A minimal CPU smoke run was **not feasible here** because the local environment does not currently have PyTorch installed (`ModuleNotFoundError: No module named 'torch'`).

## Main risks / tradeoffs

- **MTP adds training compute.** If the auxiliary head costs too many steps in the 600s window, it can erase its own benefit.
- **Value residual may over-anchor deep layers** to early-layer value structure and reduce abstraction if the mixing is too strong.
- **This is intentionally low-retune.** I changed defaults for the new idea and candidate ergonomics, but did not run a fresh hyperparameter sweep around MTP loss weight, value-residual strength, or TTT interactions.
