# MTP-Distilled Auxiliary Head on the PR #549 Stack

## Hypothesis

Add a **training-only** multi-token prediction head to the current best stack, then stabilize it with **stop-gradient self-distillation from the main head**. The extra future-token supervision should improve sample efficiency inside the 600s training budget, while the exported artifact size stays unchanged because the MTP head is excluded from export.

## Why this is promising here

The record history in this repo repeatedly rewards ideas that improve **training signal density** or **evaluation efficiency** without paying permanent artifact cost: seq2048, sliding-window eval, EMA, GPTQ-lite, XSA, Partial RoPE, and legal TTT all fit that pattern. The strongest recent scripts already include dormant MTP support in code, but no published record actually turns it on, which makes MTP a high-upside path that is both **research-backed** and **easy to adapt to this codebase**.

I specifically chose this over more invasive ideas like latent lookahead or layer recurrence because prior repo evidence says recurrence/looping needs more than the 10-minute budget, and the latest latent-lookahead papers would require much broader architectural surgery than this repository has historically tolerated.

## Influences from prior records and candidates

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Pure training/export reference:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- **Architectural lineage:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- **Repo trend evidence:** sliding-window eval, quantization-sensitive tied embeddings, 11L/2048-token training, EMA/SWA, and GPTQ-lite all recur as wins across the March 19-23 record chain
- **Prior candidates:** there was no existing `candidates/` directory in this repo when this candidate was created

## External research that informed it

- **Gloeckle et al., 2024 — _Better & Faster Large Language Models via Multi-Token Prediction_ (`arXiv:2404.19737`)**  
  Motivates MTP as an auxiliary objective that improves sample efficiency by asking the shared trunk to predict multiple future tokens.
- **Zhao et al., 2026 — _Self-Distillation for Multi-Token Prediction_ (`arXiv:2603.23911`)**  
  Motivates the exact twist used here: keep the main head untouched and improve auxiliary MTP heads with **gradient-detached KL distillation** from the main-head distribution.
- **Kirchenbauer et al., 2026 — _Multi-Token Prediction via Self-Distillation_ (`arXiv:2602.06019`)**  
  Reinforces the idea that future-token heads benefit from teacher-aligned supervision rather than plain token CE alone.
- **Noci et al., 2026 — _Thinking into the Future: Latent Lookahead Training for Transformers_ (`arXiv:2603.20219`)**  
  Useful as a contrast class: interesting, but too invasive relative to this repo’s fast-iteration record scripts and prior negative evidence on depth recurrence.

## What changed versus the chosen base

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes only the MTP path active and better behaved:

1. **Enable one auxiliary MTP head by default** (`MTP_NUM_HEADS=1`) instead of leaving MTP disabled.
2. **Add full-vocab stop-gradient KL distillation** from the main head to the MTP head during training. Because this challenge uses a 1024-token SentencePiece vocabulary, full-vocab KL is cheap enough that the Top-N truncation from MTP-D is unnecessary here.
3. **Initialize the MTP head from the tied embedding / main unembedding weights** instead of zero-initializing it, so the auxiliary head starts near a useful distribution instead of learning from scratch.
4. **Keep export behavior unchanged**: `mtp_heads.*` are still filtered out before serialization and quantization, so the MTP parameters do not count against the final artifact.

## How to run

From this candidate directory:

```bash
RUN_ID=mtp_distill_head \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 \
MTP_DISTILL_WEIGHT=0.10 MTP_DISTILL_TEMPERATURE=1.5 MTP_INIT_FROM_MAIN=1 \
SWA_ENABLED=1 SWA_EVERY=50 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For a cleaner training-only ablation, rerun the same command with `TTT_ENABLED=0` and compare the pre-TTT diagnostic / int6 roundtrip numbers.

## How to evaluate

The script keeps the base stack’s evaluation flow:

1. standard full-val BPB,
2. EMA-applied diagnostic eval,
3. int6+lzma roundtrip eval,
4. stride-64 sliding eval,
5. optional legal score-first TTT if `TTT_ENABLED=1`.

## Main expected risks and tradeoffs

- **Training-time overhead:** even one extra full-vocab head adds compute. If the step-time regression is larger than the sample-efficiency gain, this will lose despite being artifact-free.
- **Teacher mismatch:** the distillation target comes from the future-position main head, which has access to more context than the MTP student. That soft target may still help, but too much KL weight could over-regularize the trunk.
- **Hyperparameter sensitivity:** the most important knobs are `MTP_NUM_HEADS`, `MTP_LOSS_WEIGHT`, `MTP_DISTILL_WEIGHT`, and whether distillation helps more with TTT on or off.
- **No GPU confirmation here:** this candidate is code-complete and compiles, but it still needs an actual H100 run to know whether the added supervision pays for its training-time cost.

## Validation

Commands run from the repository root:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604020825_mtp-distill-head/train_gpt.py
```

Outcomes:

- `compileall` succeeded for the repository baseline files listed above
- `compileall` succeeded for this candidate script
- I did **not** run a CPU smoke train because this stack hard-requires CUDA at runtime and imports `flash_attn_interface`, so there is no safe CPU-only startup path in the existing record implementation
