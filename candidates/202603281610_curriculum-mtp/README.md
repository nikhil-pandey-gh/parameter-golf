# Curriculum MTP on top of the 2026-03-23 SOTA stack

## Hypothesis

Training-only multi-token prediction (MTP) heads can improve sample efficiency for this tiny 11-layer model without increasing the final artifact size, because the auxiliary heads are dropped before export. The twist is to use a **forward curriculum** instead of fixed-strength MTP from step 0: small language models are more brittle under MTP, so gradually turning on farther-future heads should make the auxiliary task helpful instead of disruptive.

## Why this is promising for this repository

The current record stack is already very strong on compression and evaluation: LeakyReLU(0.5)^2, XSA on deep layers, Partial RoPE, EMA, GPTQ-lite-style int6 export, and legal score-first TTT. That means one of the best remaining opportunities is a **training-side efficiency gain that does not cost export bytes**.

This repository already contains the key plumbing for that experiment in the 2026-03-23 record code: MTP heads exist, and they are explicitly excluded from the exported artifact. However, no shipped record actually enabled them. That makes MTP a good fit here: high leverage, low infrastructure, and genuinely new relative to the current `records/` tree.

## Which records influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Chosen base implementation. It is the current best overall stack and already includes the training-only MTP/export split.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Useful pre-TTT reference for the mature 11-layer XSA + Partial RoPE + GPTQ-lite export stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Important cautionary note: mutable runtime flags can be constant-folded away by `torch.compile`, so new training controls should be passed more explicitly.

There were no prior `candidates/` directories in the repository at the time this candidate was created.

## External research that informed it

- Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** (2024)  
  https://arxiv.org/abs/2404.19737
  - Motivates MTP as an auxiliary objective that can improve sample efficiency.
- Ansar Aynetdinov and Alan Akbik, **Pre-Training Curriculum for Multi-Token Prediction in Language Models** (2025)  
  https://arxiv.org/abs/2505.22757
  - Most relevant for this repo: they specifically report that **small language models struggle with raw MTP**, and that a **forward curriculum** helps small models benefit from the objective.

I also considered more invasive ideas from the literature, especially ALBERT-style sharing/factorized embeddings and learned-scale QAT. I did not choose them here because they require broader architectural churn, while this repo already has a clean training-only MTP path ready to extend.

## What changed versus the chosen base implementation

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes four focused changes:

1. **Turn MTP on by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Add a forward MTP curriculum**
   - New env var: `MTP_CURRICULUM_STEPS` (default `2000`)
   - The two auxiliary heads are ramped in sequentially across the first 2000 training steps instead of applying full-strength MTP immediately.

3. **Pass curriculum weights explicitly through the compiled forward**
   - The curriculum is fed as a runtime tensor argument to the model, rather than toggling a mutable flag that `torch.compile` could freeze.

4. **Make the script runnable from the candidate directory**
   - Default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root computed from `__file__`, so `cd candidates/202603281610_curriculum-mtp && torchrun ... train_gpt.py` works without rewriting paths.

The rest of the stack is intentionally inherited: parameter banking + Parallel Muon, LeakyReLU(0.5)^2 MLP, deep-layer XSA, Partial RoPE, EMA, mixed int6 export, and optional legal TTT.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603281610_curriculum-mtp

RUN_ID=curriculum_mtp \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
MTP_CURRICULUM_STEPS=2000 \
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

Notes:

- If your FineWeb shards/tokenizer live in the repository-standard locations under `data/`, you do not need to set `DATA_PATH` or `TOKENIZER_PATH`.
- To compare against fixed-strength MTP, set `MTP_CURRICULUM_STEPS=0`.
- To measure the pure training-side effect without inherited TTT, set `TTT_ENABLED=0`.

## Main expected risks and tradeoffs

- **Training-time overhead:** the auxiliary heads add compute. If the MTP gain is smaller than the lost step count under the 600s cap, this could regress.
- **Schedule sensitivity:** a 2000-step curriculum is a reasonable first guess, not a proven optimum. The best setting may depend on actual throughput on 8xH100.
- **Interaction with TTT:** MTP is only used during pretraining. The exported/evaluated model drops those heads, so the gain must survive the export boundary and still help the TTT-enabled final score.
- **Tiny-model brittleness:** the curriculum paper is encouraging, but this repo’s models are extremely small and highly compression-constrained, so the win could be narrow.

## Validation

I ran the lowest-cost checks available in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603281610_curriculum-mtp/train_gpt.py
python -m compileall candidates/202603281610_curriculum-mtp/train_gpt.py
```

Outcome:

- Both compile steps passed.
- A follow-up import/smoke probe was **not feasible** in this runner:
  - the Python environment is missing repo dependencies from `requirements.txt` (the probe failed immediately on `ModuleNotFoundError: No module named 'numpy'`), and
  - the downloaded dataset/tokenizer artifacts under `data/datasets/` and `data/tokenizers/` are not present in this workspace snapshot.

So this candidate is syntax-validated here, but still needs a real dependency-installed GPU run for end-to-end verification and score measurement.
