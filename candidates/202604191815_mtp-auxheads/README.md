# MTP auxiliary heads on the 2026-03-23 banked recipe

## Hypothesis

The strongest unexplored low-infrastructure idea in this repo is to turn on **training-only multi-token prediction (MTP)** for the current 11-layer frontier recipe. Predicting the next 4 tokens with auxiliary heads should improve sample efficiency and planning-style representations during the same 10-minute training budget, while adding **zero exported artifact bytes** because the MTP heads are stripped before quantization/export.

## Why this is promising here

- The repo frontier already converged on a strong 11-layer, 512-dim, 3x-MLP, seq2048 stack with XSA, partial RoPE, EMA/SWA, legal TTT, and compression-aware quantization.
- The newest banked script already contains complete MTP forward/export plumbing, but every shipped run keeps `mtp_num_heads:0`, and the 2026-03-23 banked optimizer path drops the MTP heads from all optimizer groups.
- This challenge uses a tiny `VOCAB_SIZE=1024`, so auxiliary MTP heads are unusually cheap compared with normal LLM settings: extra training compute is modest, but the representation-learning upside can still be meaningful.

## Prior experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` established the pre-TTT 11-layer XSA/EMA/partial-RoPE/GPTQ-lite recipe.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` added LeakyReLU^2, legal score-first TTT, and parameter-bank / parallel-Muon execution, but still logged `mtp_num_heads:0`.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/` and `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` show that the codebase has carried dormant MTP support for several generations without actually exercising it.

## External research

- **Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737): multi-token auxiliary heads improve sample efficiency and downstream capability while keeping inference overhead optional.
- **Chertkov et al., "Faster Language Models with Better Multi-Token Prediction Using Tensor Decomposition"** (arXiv:2410.17765): reinforces the idea that MTP remains effective even when efficiency is a concern.
- **Huang et al., "How Transformers Learn to Plan via Multi-Token Prediction"** (arXiv:2604.11912): gives a recent mechanistic argument that MTP encourages cleaner planning-style circuits than pure next-token prediction.

## What changed vs the base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Set `MTP_NUM_HEADS=4` by default so the candidate actually trains with MTP.
2. Fixed the banked optimizer split so `mtp_heads` are actually optimized on the replicated non-bank AdamW path when enabled.
3. Resolved the default dataset/tokenizer paths relative to the repository root so the script can be launched directly from this candidate directory.
4. Kept the existing export behavior that excludes `mtp_heads`, preserving the artifact budget.

## How to run

From the repository root:

```bash
cd candidates/202604191815_mtp-auxheads
python train_gpt.py
```

To stack this candidate with the same legal TTT path used by the current frontier:

```bash
cd candidates/202604191815_mtp-auxheads
TTT_ENABLED=1 python train_gpt.py
```

The script keeps the base record's environment-variable surface, so further sweeps can still tune `MTP_NUM_HEADS`, `MTP_LOSS_WEIGHT`, `TTT_*`, or the existing optimizer/quantization knobs.

## Risks and tradeoffs

- Extra auxiliary heads add training compute; if the extra loss does not pay for itself in fewer BPB/nat per step, total progress in 600s could regress.
- The best MTP setting for this small-vocab regime may be fewer than 4 heads or a lower loss weight.
- This candidate fixes the missing optimizer wiring in the banked script, but it does not yet add the newer tensor-decomposition MTP variant from the 2025 follow-up paper.
- Legal TTT remains optional here; if MTP improves the trunk enough, the best combined result may come from re-enabling TTT during full GPU runs.

## Validation

Lightweight validation run from the repository root:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604191815_mtp-auxheads/train_gpt.py
```

Outcome:

- `python -m compileall ...` succeeded for the root scripts, `data/`, and `candidates/202604191815_mtp-auxheads/train_gpt.py`.
- A stricter CPU smoke test was planned with a local FlashAttention stub, but this runner does not currently have the repo's `torch` dependency installed (`ModuleNotFoundError: No module named 'torch'`), so an import/forward smoke test was not feasible without installing heavyweight training dependencies.
