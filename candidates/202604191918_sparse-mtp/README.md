# Sparse MTP on the March 23 SOTA Stack

## Hypothesis

A **sparse multi-token prediction (MTP)** auxiliary loss should improve sample efficiency on the current best stack without changing exported artifact bytes or evaluation latency, because the extra heads are training-only and are still excluded from export.

Concretely, this candidate asks the model to predict **one additional future token** (`t+2`) on **every 4th position** during training. The sparse schedule is the key twist: dense MTP is attractive in the literature, but this repository is wallclock-limited, so the auxiliary loss needs to be cheap enough not to erase its own gain by slowing training too much.

## Why this is promising here

Repository review showed that the strongest lineage has already extracted most of the obvious wins from sliding evaluation, quantization-aware export, EMA, XSA, Partial RoPE, LN scale, BigramHash, SmearGate, and LeakyReLU^2. It also showed two useful facts:

1. The top record lineage already contains dormant MTP hooks, but the submitted runs never enabled them (`mtp_num_heads:0` in the logs).
2. In the March 23 parameter-banked branch, the MTP heads were defined and logged but not wired into any optimizer, so simply flipping `MTP_NUM_HEADS=1` there would not actually train them.

That makes MTP a good next candidate for this repository: it is **novel locally**, **cheap in artifact bytes**, and **small enough to try in a self-contained folder**.

## Prior records that influenced this candidate

- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`**  
  Current best stack. This candidate copies that script as the base because it already carries LeakyReLU^2, legal score-first TTT, parameter banking, Parallel Muon, BigramHash(1536), XSA4, Partial RoPE, LN scale, VE128, EMA, and the strongest known lzma export path.

- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`**  
  Best non-TTT training/export stack. It helped confirm that the repo's strongest pure-training lineage already had MTP code paths, and that training-only auxiliary heads can be exported away cleanly.

- **`records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`**  
  Important reminder that some training-path ideas can silently no-op under `torch.compile`, so this candidate keeps the MTP delta explicit and auditable.

- **No prior `candidates/` directory existed** at workflow start, so this is the first candidate iteration in that folder.

## External research that informed it

- **Better & Faster Large Language Models via Multi-Token Prediction** (Gloeckle et al., 2024, arXiv:2404.19737) argues that predicting multiple future tokens from a shared trunk improves sample efficiency and downstream quality.
- **DeepSeek-V3 Technical Report** (DeepSeek-AI, 2024/2025, arXiv:2412.19437) explicitly calls out a multi-token prediction training objective as part of a high-performing modern LM recipe.

I also surveyed other compact-model directions, especially **ALBERT-style cross-layer sharing** (Lan et al., 2019, arXiv:1909.11942), **Universal Transformer-style recurrence** (Dehghani et al., 2018, arXiv:1807.03819), and **more aggressive KV sharing / MQA** (Shazeer, 2019, arXiv:1911.02150). Those remain interesting, but I did **not** choose them for this candidate because the repository already reports negative results for naive recurrence and because MTP is a lower-risk way to buy training efficiency without changing the proven architecture or export path.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`

2. **Add sparse MTP**
   - New hyperparameter: `MTP_STRIDE` (default `4`)
   - The auxiliary future-token loss is only computed on every 4th position instead of every token.

3. **Fix MTP optimizer wiring in the parameter-banked branch**
   - The March 23 script comments said MTP heads should go through Adam, but their weights were not actually added to any optimizer.
   - This candidate adds the MTP head weights to the replicated AdamW parameter set so they receive gradients, updates, and cross-rank all-reduce.

4. **Keep export behavior unchanged**
   - MTP heads are still excluded from the exported checkpoint, so the candidate pays training-time compute for the auxiliary heads but does **not** pay artifact bytes for them.

Everything else stays on the March 23 stack.

## How to run / evaluate

Run from this candidate directory:

```bash
cd candidates/202604191918_sparse-mtp

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.1 MTP_STRIDE=4 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate defaults already enable the sparse MTP settings; the explicit flags above are included so the intended configuration is obvious in logs and ablations.
This base script always maintains an EMA shadow model internally at decay `0.997`.

## Main expected risks / tradeoffs

- **Training-time overhead**: even sparse MTP adds extra vocab projections and cross-entropy work, so it may reduce steps enough to erase the theoretical gain.
- **Aux-loss strength**: `MTP_LOSS_WEIGHT=0.1` and `MTP_STRIDE=4` are deliberately conservative; denser or stronger MTP may help more, or may simply be too expensive.
- **Optimizer choice**: the MTP heads now go through AdamW in the parameter-banked branch. That matches the March 23 comment, but Muon-on-MTP-heads is another reasonable ablation.
- **TTT interaction**: the auxiliary objective changes the pretrained model before legal score-first TTT, so the post-TTT gain could move in either direction.

## Validation

Commands run in this workflow:

```bash
PYTHONPYCACHEPREFIX=/tmp/gh-aw/agent/pycache \
python -m compileall candidates/202604191918_sparse-mtp/train_gpt.py
```

Outcome:

- **Succeeded**.

Attempted additional smoke check:

```bash
python - <<'PY'
# import candidate script with a local flash-attn stub and run a tiny CPU forward
PY
```

Outcome:

- **Blocked on this runner** because `torch` is not installed in the workflow environment, so a real import/forward smoke test was not feasible without installing heavyweight dependencies.
