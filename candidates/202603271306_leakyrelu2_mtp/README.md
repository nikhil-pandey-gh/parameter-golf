# Candidate: LeakyReLU(0.5)^2 + MTP on the clean 11L GPTQ-lite stack

## Hypothesis

Take the strongest clean non-TTT stack in the repo, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, and make two low-infrastructure changes:

1. swap the MLP activation from `ReLU^2` to `LeakyReLU(0.5)^2`, which the current best record already ablates as a real win, and
2. actually turn on the dormant multi-token prediction (MTP) auxiliary heads that already exist in the codebase, so training gets denser supervision without paying for those heads in the final artifact.

The expected outcome is a better 10-minute quality/complexity tradeoff: `LeakyReLU(0.5)^2` improves gradient flow at essentially zero systems cost, while short-horizon MTP should improve sample efficiency under the fixed wall-clock budget.

## Why this is promising for this repository

This repository's best results come from stacking cheap, cumulative improvements on top of the same strong 11-layer recipe rather than from broad infrastructure rewrites. The 2026-03-22 run is the best "clean" base because it already has 11 layers, XSA, partial RoPE, LN scale, EMA, GPTQ-lite int6 export, shared value embeddings, and sliding-window evaluation, but it does **not** yet port the `LeakyReLU(0.5)^2` change that helped the 2026-03-23 record.

The repo review also showed that MTP support is already wired into several top scripts, but none of the record READMEs describe a real MTP result. That makes MTP a practical underexplored direction here: it is already implemented, it is easy to enable, and the export path already excludes `mtp_heads`, so the final artifact stays comparable to the base model.

## Which records influenced it

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the base because it is the strongest non-TTT, non-parameter-banked stack in the repo.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the `LeakyReLU(0.5)^2` activation change; its README reports a meaningful ablation win from this one-line modification.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` and `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - provide the architectural ancestry for the 11-layer XSA / partial-RoPE line that this candidate keeps intact.

## Which external research informed it

- **Multi-Token Prediction via Self-Distillation** ([arXiv:2602.06019](https://arxiv.org/abs/2602.06019))
  - argues that multi-token prediction can improve efficiency without needing separate verifier/speculator infrastructure.
- **Understanding and Enhancing the Planning Capability of Language Models via Multi-Token Prediction** ([arXiv:2509.23186](https://arxiv.org/abs/2509.23186))
  - gives additional evidence that MTP helps models internalize multi-step structure instead of only the immediate next token.

Those papers are not a direct recipe for this challenge, but they strengthen the case for enabling a short-horizon auxiliary prediction loss in a fixed-time training regime where token efficiency matters.

## What changed versus the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

- default `MTP_NUM_HEADS` changed from `0` to `2`
- default `MTP_LOSS_WEIGHT` changed from `0.2` to `0.15`
- MLP activation changed from:

```python
x = torch.relu(self.fc(x))
return self.proj(x.square())
```

to:

```python
x = F.leaky_relu(self.fc(x), negative_slope=0.5)
return self.proj(x.square())
```

- default dataset/tokenizer paths are now computed relative to the repository root so the script can be run directly from this candidate directory without needing to override `DATA_PATH` and `TOKENIZER_PATH`.

Everything else is intentionally left the same as the base record.

## How to run

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults expect the cached challenge dataset and tokenizer at:

- `../../data/datasets/fineweb10B_sp1024/`
- `../../data/tokenizers/fineweb_1024_bpe.model`

Useful overrides:

```bash
RUN_ID=leakyrelu2_mtp \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

I ran:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603271306_leakyrelu2_mtp/train_gpt.py
```

Outcome: all listed files compiled successfully.

Planned lightweight smoke check:

```bash
python candidates/202603271306_leakyrelu2_mtp/train_gpt.py
```

but a real CPU-only smoke run is **not feasible in this workspace as-is** because:

- the script is CUDA-only (`main()` raises if CUDA is unavailable),
- the environment used for this task does not currently have the challenge runtime dependencies installed, and
- the cached dataset/tokenizer shards are not present under the repository `data/` tree in this checkout.

So validation here is limited to syntax-level compilation and code review.

## Main expected risks / tradeoffs

- MTP adds training-time compute, so too many auxiliary heads or too much weight could lower step count enough to erase the gain.
- The best repo evidence for `LeakyReLU(0.5)^2` comes from a stronger TTT/Parallel-Muon stack; the transfer to the clean 2026-03-22 base is plausible but still unverified.
- Because the export still drops `mtp_heads`, the final model is trained with a richer objective than the one evaluated at inference/export time. That mismatch may help or hurt depending on the exact regime.
