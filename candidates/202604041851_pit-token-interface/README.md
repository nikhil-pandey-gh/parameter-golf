# PIT-lite Token Interface on the 11L LeakyReLU² + Legal TTT stack

## Hypothesis

The current best stack already captures most of the obvious wins in this repo: deeper 11-layer U-Net structure, efficient XSA, EMA-backed export, partial RoPE, GPTQ-lite quantization, and legal score-first TTT. A remaining weak spot is the **directly tied token interface** itself. Recent work on **Pseudo-Inverse Tying (PIT)** argues that compact language models benefit when embedding and unembedding stay coupled through a stable shared token memory rather than drifting apart during training.

This candidate tests a minimal adaptation of that idea: keep the best existing training/eval stack, but replace the plain tied embedding with a **PIT-lite diagonal token interface** that:

1. uses an orthogonally initialized shared token memory,
2. applies a learned diagonal hidden-space transform on the input/output token path, and
3. adds a small orthogonality regularizer so the shared memory stays close to a pseudo-inverse-friendly basis.

## Why this is promising for this repository

Repository history strongly favors **small, low-overhead refinements on top of a strong 11-layer stack** over broad architectural resets. Partial RoPE, LN scale, EMA, GPTQ-lite, LeakyReLU(0.5)^2, and legal TTT all improved the same base by roughly 0.001-0.003 BPB each while preserving the core recipe.

This PIT-lite change follows the same pattern:

- it does **not** disturb the parameter-banked attention/MLP core,
- it adds only a tiny number of extra control parameters,
- it keeps the existing quantization/export path intact, and
- it targets a component that every run relies on: the tied token table.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the direct base. This candidate keeps its 11-layer banked stack, LeakyReLU^2 MLP, XSA, VE, legal TTT, and Parallel Muon machinery unchanged.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` reinforced the pattern that export-aware, low-cost changes on top of the mature 11-layer stack keep paying off.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` showed that zero/near-zero-parameter tweaks to the representation path can matter at this scale.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` is useful as a negative control: more aggressive recurrence or slower architectural shifts were fragile, so this candidate stays surgical instead.

There were **no prior folders under `candidates/`** at implementation time, so this is the first candidate iteration in that namespace.

## External research informing the idea

- **Gu et al., “Rethinking Weight Tying: Pseudo-Inverse Tying for Stable LM Training and Updates” (arXiv:2602.04556, 2026).** Motivates replacing naive tied embeddings with a shared-memory interface that keeps embedding and unembedding coupled through a pseudo-inverse-consistent transform.
- **Janson et al., “Stabilizing Native Low-Rank LLM Pretraining” (arXiv:2602.12429, 2026).** Motivates orthogonalized/shared token factors and explicit regularization to keep compact parameterizations stable during training.
- **Lan et al., “ALBERT: A Lite BERT for Self-supervised Learning of Language Representations” (arXiv:1909.11942, 2019).** Older but still relevant prior art for focusing parameter-efficiency work on the token interface instead of only scaling depth/width.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Replaced the plain tied `nn.Embedding` path with `PseudoInverseTiedEmbedding`.
2. Added a learned diagonal `pit_log_scale` so the input embedding path uses `exp(-scale)` and the output projection path uses `exp(scale)`.
3. Initialized the shared token memory orthogonally instead of with the base run's Gaussian tied-embedding init.
4. Added `PIT_ORTHO_REG` as a small orthogonality penalty on the shared token memory during training.
5. Resolved the default dataset/tokenizer paths relative to the repository root so the script can be launched directly from this candidate directory.
6. Kept EMA as the documented default averaging path and wired optional SWA application correctly if someone explicitly enables it.
7. Added a FlashAttention import fallback to PyTorch SDPA so the module can still import and run tiny CPU-only model smoke tests outside the full challenge environment.

Everything else stays aligned with the current best record's architecture and evaluation recipe.

## How to run / evaluate

Training command, run from this candidate directory:

The script resolves `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root by default, so no extra path overrides are needed for a standard checkout. EMA is always on in this script, and late fake-quant is controlled by `LATE_QAT_THRESHOLD`.

```bash
PIT_ENABLED=1 PIT_ORTHO_REG=1e-4 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Executed from the **repo root**:

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604041851_pit-token-interface/train_gpt.py` -> passed
- `python -m compileall candidates/202604041851_pit-token-interface/train_gpt.py` -> passed

Executed from the **candidate directory** in an isolated temporary venv after installing `numpy sentencepiece torch`:

- `python -m compileall train_gpt.py` -> passed
- `python - <<'PY' ...` import/path check -> passed (`default_data_path=/home/runner/work/parameter-golf/parameter-golf/data/datasets/fineweb10B_sp1024`, `default_tokenizer_path=/home/runner/work/parameter-golf/parameter-golf/data/tokenizers/fineweb_1024_bpe.model`, `swa_enabled_default=False`)
- `python - <<'PY' ...` tiny CPU smoke test that imports `train_gpt.py`, instantiates a small `GPT(...)`, and runs both `forward` and `forward_logits` -> passed (`loss=6.4838`, `logits_shape=(2, 16, 64)`, `pit_enabled=True`)

## Main expected risks / tradeoffs

- The diagonal PIT-lite approximation is much cheaper than full PIT, but it may also be too weak to deliver the full benefit described in the paper.
- Orthogonal token memory may improve stability but could slightly worsen quantization behavior relative to the empirically tuned direct tied table.
- The orthogonality penalty may help early training but interfere with late TTT adaptation if it is too strong.
- Because this change only touches the token interface, the upside is likely incremental rather than dramatic; the goal is a safe additive improvement on top of the current best stack, not a wholesale replacement.
