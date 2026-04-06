# Zero-byte MTP on the 2026-03-23 stack

## Hypothesis

The current public record line has already captured most of the obvious gains from better quantization, EMA/SWA, partial RoPE, XSA, and evaluation tricks. A strong next move is to improve **training-time sample efficiency** without paying permanent artifact bytes.

This candidate enables **1-head multi-token prediction (MTP)** by default on top of the strongest in-repo training script. The extra head is used only during training and is explicitly excluded from the exported checkpoint, so the main tradeoff is a small amount of extra training compute in exchange for richer supervision.

## Why this is promising here

1. The best public stacks are already close to the 16MB artifact limit, so export-free improvements are especially attractive.
2. The 2026-03-23 record and the 2026-03-22 pre-TTT stack already contain dormant MTP support, but no public record in this repository appears to have actually turned it into the main hypothesis.
3. Recent MTP research argues that predicting multiple future tokens improves sample efficiency and reasoning development under the same trunk, which is exactly the kind of tradeoff this challenge rewards.

## Prior repository evidence that influenced this candidate

- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`**: strongest in-tree stack, with LeakyReLU(0.5)^2, parameter banks, Parallel Muon, legal score-first TTT, partial RoPE, VE, and deep XSA.
- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`**: strongest pre-TTT training recipe and the clearest evidence that the repo is already highly optimized on the quantization side.
- **`records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`** and nearby 03-20 records: showed that the winning backbone is stable, so a new training objective is more novel than yet another small architecture retune.

## External research that informed it

- **Better & Faster Large Language Models via Multi-token Prediction** (`arXiv:2404.19737`): reports that auxiliary multi-token heads improve sample efficiency and downstream capability while reusing a shared trunk.
- **Faster Language Models with Better Multi-Token Prediction Using Tensor Decomposition** (`arXiv:2410.17765`): reinforces that low-overhead MTP can improve efficiency without sacrificing quality.
- **Self-Distillation for Multi-Token Prediction** (`arXiv:2603.23911`): suggests MTP heads benefit from careful training; that informs the conservative choice here to start with a single lightweight auxiliary head before adding more machinery.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Turn MTP on by default** with:
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`
2. Keep the existing **export-free behavior**: `mtp_heads` are still excluded from the saved/exported model artifact.
3. Add a **FlashAttention fallback** to PyTorch SDPA when FlashAttention is unavailable or when running on CPU.
4. Add a **`COMPILE_MODEL` toggle** and CPU-safe runtime handling so the candidate can be smoke-tested off-GPU without changing the default CUDA path.

The intent is to keep the candidate focused: same strong base stack, same quantization/export path, same legal TTT option, but a new training objective branch.

## How to run or evaluate it

From this directory:

```bash
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
SWA_ENABLED=1 SWA_EVERY=50 MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a local non-GPU smoke run, set `COMPILE_MODEL=0` and use a tiny dataset/tokenizer plus reduced dimensions.

## Main risks and tradeoffs

- **Step-time regression**: MTP is not free during training, even if it is free at export. A gain in sample efficiency only matters if it is not cancelled out by fewer steps in 600 seconds.
- **Small-model uncertainty**: the main MTP papers show strong results, but most public results are on larger models than this repository’s regime.
- **Objective interference**: MTP can steal capacity from the main next-token objective if the head count or weight is too aggressive.
- **Stack interaction risk**: MTP is being layered onto a highly tuned recipe with EMA, quantization, Parallel Muon, and optional TTT; those interactions are not yet ablated here.

## Validation

Commands run:

1. `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604061449_zero-byte-mtp/train_gpt.py`
   - **Outcome:** passed.
2. Minimal CPU smoke on a synthetic SentencePiece tokenizer plus temporary FineWeb-format toy shards, with reduced dimensions and `COMPILE_MODEL=0`.
   - **Outcome:** completed 1 training step, EMA application, int6 export, roundtrip eval, and sliding-window eval without crashing.
   - Final synthetic smoke metric: `final_int6_sliding_window_exact val_bpb: 3.85263992` (not meaningful for quality, only for startup verification).

## Suggested next experiments

1. Sweep `MTP_NUM_HEADS` in `{1, 2}` and `MTP_LOSS_WEIGHT` in `{0.10, 0.15, 0.20}` on the 8xH100 path.
2. Check whether MTP still helps once legal TTT is enabled, or whether it mainly improves the pre-TTT checkpoint.
3. If 1-head MTP helps, test a lightweight distillation variant inspired by `arXiv:2603.23911` rather than immediately adding more heads.
