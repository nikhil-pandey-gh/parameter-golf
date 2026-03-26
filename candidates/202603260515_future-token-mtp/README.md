# 202603260515 Future-Token MTP

## Hypothesis

A **single training-only multi-token prediction (MTP) head** can improve the hidden-state quality of the current best 11-layer stack without increasing artifact size, because the auxiliary head is excluded from export. The bet is that light future-token supervision will improve the pre-TTT model and possibly make the legal score-first TTT stage more effective, while preserving the existing compression-aware recipe.

## Why this is promising for this repository

The repository's winning trend is clear:

- better **compression-aware training/export** funded the move from 9L to 11L,
- cheap architectural math tricks like **XSA, Partial RoPE, LN scale, EMA, GPTQ-lite** kept helping,
- the best current score came from stacking those ideas on top of **LeakyReLU² + legal TTT + Parallel Muon**.

A future-token auxiliary loss fits those same constraints:

- it adds **no exported parameters** because `mtp_heads` are already stripped before serialization,
- it reuses infrastructure already present in the stronger 11L codepaths,
- it is a **training-only** change, so it doesn't compete with the 16 MB artifact budget,
- it targets representation quality rather than more export tricks, which is a relatively unexhausted axis in this repo.

## Prior records that influenced this candidate

Primary donor base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Most relevant earlier precursors:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
- root `train_gpt.py`

Important repo-review observations that drove this choice:

- recent strong codepaths already carried **MTP scaffolding**, but the public runs/logs kept `MTP_NUM_HEADS=0`, so the idea was never actually exercised in a record run,
- the copied donor implementation also did **not attach `mtp_heads` to any optimizer group**, so simply toggling `MTP_NUM_HEADS>0` would have left the auxiliary head effectively dead.

## External research that informed it

This candidate is grounded in recent work arguing that supervising the model on future information can improve or enrich autoregressive representations:

- **Multi-Token Prediction via Self-Distillation** (`arXiv:2602.06019`) proposes converting autoregressive models into standalone multi-token predictors via a simple online distillation objective.
- **Self-Distillation for Multi-Token Prediction** (`arXiv:2603.23911`) reports that lightweight self-distillation can improve MTP head quality while preserving the main head.
- **Thinking into the Future: Latent Lookahead Training for Transformers** (`arXiv:2603.20219`) shows that supervising future predictions in latent space can outperform autoregressive baselines on planning-style tasks.
- **Beyond Multi-Token Prediction: Pretraining LLMs with Future Summaries** (`arXiv:2510.14751`) argues that future-target auxiliary objectives can improve reasoning/planning beyond plain next-token training.
- **Understanding and Enhancing the Planning Capability of Language Models via Multi-Token Prediction** (`arXiv:2509.23186`) analyzes how MTP helps models capture multi-step structure.

This implementation is intentionally simpler than those papers: it uses the repo's existing lightweight extra-head formulation instead of adding a new distillation or summary-training stack.

## What changed vs the chosen base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. **Turns on a real 1-head MTP default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`

2. **Fixes the auxiliary-head training bug**
   - the donor code defined `mtp_heads` and added their loss, but never added those parameters to any optimizer,
   - this candidate adds a dedicated optimizer group for the MTP heads via `MTP_HEAD_LR` (defaulting to `HEAD_LR`).

3. **Keeps MTP training-only and export-free**
   - `mtp_heads` are still excluded from `final_model.pt` and the compressed int6 artifact.

4. **Makes the script runnable from the candidate directory itself**
   - default `DATA_PATH` / `TOKENIZER_PATH` are now derived from the script location instead of assuming repo-root CWD,
   - `TRAIN_FILES` / `VAL_FILES` overrides were added for synthetic or custom smoke data.

5. **Adds lightweight CPU smoke-test support**
   - optional `ALLOW_CPU=1 DEVICE=cpu COMPILE=0` path,
   - FlashAttention falls back to PyTorch SDPA,
   - fused optimizers and `torch.compile` are disabled in CPU mode,
   - Parallel Muon falls back to fp32 buffers on CPU.

6. **Fixes a RoPE cache edge case**
   - cached rotary tensors created under `torch.inference_mode()` are cloned before later training use, avoiding inference-tensor/autograd conflicts during eval-then-train flows.

## How to run / evaluate

### Full GPU run (candidate defaults + legal TTT at eval)

```bash
cd candidates/202603260515_future-token-mtp
TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This candidate intentionally keeps the donor script's non-MTP defaults unchanged. If you want to match the donor record's documented launch command more closely, try the ablation `BIGRAM_VOCAB_SIZE=1536`.

Useful overrides for ablations:

```bash
MTP_NUM_HEADS=0      # recover donor behavior (no future-token aux)
MTP_LOSS_WEIGHT=0.05 # weaker aux pressure
MTP_HEAD_LR=0.004    # slower auxiliary-head learning
TTT_ENABLED=0        # pre-TTT-only evaluation
```

### Minimal smoke run with synthetic data

This runner snapshot did **not** contain the real SentencePiece model or FineWeb shards under `data/`, so the smoke test below uses temporary synthetic shards plus a temporary local SentencePiece model:

```bash
cd candidates/202603260515_future-token-mtp
ALLOW_CPU=1 DEVICE=cpu COMPILE=0 \
TOKENIZER_PATH=/tmp/gh-aw/agent/smoke-data/smoke_sp.model \
TRAIN_FILES=/tmp/gh-aw/agent/smoke-data/fineweb_train_0.bin \
VAL_FILES=/tmp/gh-aw/agent/smoke-data/fineweb_val_0.bin \
VOCAB_SIZE=64 NUM_LAYERS=2 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
TRAIN_SEQ_LEN=32 EVAL_SEQ_LEN=32 TRAIN_BATCH_TOKENS=128 VAL_BATCH_SIZE=128 \
ITERATIONS=1 WARMDOWN_ITERS=0 WARMUP_STEPS=0 VAL_LOSS_EVERY=1 TRAIN_LOG_EVERY=1 \
BIGRAM_VOCAB_SIZE=0 XSA_LAST_N=0 VE_ENABLED=0 TTT_ENABLED=0 EVAL_STRIDE=0 \
python train_gpt.py
```

## Risks / tradeoffs

- **Step-time risk:** even a single future-token head adds extra softmax/loss work, which may reduce total steps in the 10-minute budget.
- **Optimization risk:** the auxiliary objective could regularize too aggressively and hurt main-head BPB instead of helping.
- **TTT interaction risk:** future-token supervision may help legal TTT adapt better, but it could also make the model less eager to specialize chunk-by-chunk.
- **Validation gap risk:** the smoke run confirms startup/training/export/roundtrip correctness, but not leaderboard-representative quality or throughput.

## Validation run in this workflow

### Syntax check

Command:

```bash
python -m compileall candidates/202603260515_future-token-mtp/train_gpt.py
```

Outcome: **passed**.

### Synthetic CPU smoke test

Command family:

- create a temporary 64-token SentencePiece model under `/tmp/gh-aw/agent/smoke-data/`,
- create temporary train/val binary shards with the repo's expected header format,
- run `train_gpt.py` for 1 tiny CPU step.

Observed outcome:

- script started successfully,
- evaluation ran,
- 1 training step completed,
- EMA application completed,
- export excluded the MTP params (`export_excluding_mtp_params:4096`),
- int6+lzma export + roundtrip evaluation completed.

Key smoke outputs:

- `step:1/1 train_loss:4.5813`
- `step:1/1 val_loss:3.4711 val_bpb:2.9457`
- `final_int6_roundtrip_exact val_loss:4.15782464 val_bpb:3.52851353`
- `Total submission size int6+lzma: 192865 bytes`

These numbers are only smoke-test sanity checks on synthetic data, not meaningful leaderboard estimates.
