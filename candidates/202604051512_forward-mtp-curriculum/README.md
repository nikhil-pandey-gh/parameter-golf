# Forward-Curriculum MTP on the 03-23 LeakyReLU2 + Legal TTT Stack

## Hypothesis

The strongest next pure-training addition for this repository is **multi-token prediction (MTP) with a small-model-friendly forward curriculum**: start the 11-layer winning stack as a standard next-token model, then gradually phase in auxiliary future-token heads over the training budget. The auxiliary heads are **dropped before export**, so the candidate pays extra training-time compute but **no artifact-size cost**.

## Why this is promising for this repository

Repository review points to two clear ladders:

1. **Evaluation/context improvements**: doc isolation -> sliding-window evaluation -> legal score-first TTT.
2. **Compression-funded model improvements**: fp16-sensitive tensors -> mixed int6/int8 quantization -> bigger MLPs / more layers -> SmearGate + BigramHash -> XSA + EMA + Partial RoPE + GPTQ-lite.

The current best overall record (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`) already looks close to locally optimized on architecture, evaluation, and export. The cleanest underexplored knob is therefore a **train-time-only objective improvement** that can ride on top of that stack without changing the exported artifact.

This candidate specifically targets that gap:

- the repo already carries dormant MTP plumbing in late record code, but no documented record turns it on;
- the best pure-training/export run (`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`) shows the repository can still win through small, targeted training/export changes;
- the non-record survey suggests more invasive recurrence ideas are riskier here, while a training-only head objective is much cheaper to try.

No prior `candidates/` directory existed when this candidate was chosen.

## Prior records that influenced this candidate

- **2026-03-23 LeakyReLU2 + Legal TTT + Parallel Muon**: chosen as the direct code base because it is the strongest overall stack and already knows how to exclude `mtp_heads` from export.
- **2026-03-22 11L EMA + GPTQ-lite + warmdown3500**: strongest pure-training/export record; reinforced the idea that the next gain should preserve the 11L/XSA/Partial-RoPE family and attack training quality or quantization gap rather than rewrite the model.
- **2026-03-20 / 2026-03-21 11L XSA + EMA + Partial RoPE line**: established the stable late-layer recipe this candidate keeps intact.
- **2026-03-19 non-record SwiGLU / recurrence survey**: useful negative evidence against spending this iteration on layer recurrence or broader block sharing.

## External research that informed it

- **Fabian Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_ (arXiv:2404.19737)**: argues that predicting multiple future tokens from a shared trunk improves sample efficiency and can improve downstream capability.
- **Ansar Aynetdinov and Alan Akbik, _Pre-Training Curriculum for Multi-Token Prediction in Language Models_ (arXiv:2505.22757)**: the most relevant source for this repo's regime; it reports that **small language models struggle with naive MTP**, while a **forward curriculum** makes MTP helpful again.

I also considered recurrence / sharing ideas such as **Intra-Layer Recurrence in Transformers for Language Modeling** (arXiv:2505.01855), but did not choose them for this iteration because they are more invasive and the repository's own non-record evidence around recurrence is weak.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enabled MTP by default** with `MTP_NUM_HEADS=2`.
2. Added a **forward curriculum**:
   - `MTP_CURRICULUM_START=0.10`
   - `MTP_CURRICULUM_END=0.60`
   - head 1 ramps in first, then head 2 ramps in, using the training wallclock budget when available.
3. Added an explicit **optimizer path for MTP heads** in the Parallel Muon variant so the auxiliary heads actually train when enabled.
4. Kept the existing **export-time MTP stripping** so auxiliary heads do not count toward the final artifact.
5. Made the script **runnable from the candidate directory** by resolving default dataset and tokenizer paths relative to the repository root inferred from `__file__`.
6. Added lightweight logging of the active MTP head weights during training (`mtp_w:[...]`) for debugging and ablation sanity.

## How to run

From the repository root:

```bash
cd candidates/202604051512_forward-mtp-curriculum
RUN_ID=forward_mtp_candidate \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.2 MTP_LR=0.025 \
MTP_CURRICULUM_START=0.10 MTP_CURRICULUM_END=0.60 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The candidate defaults already enable the MTP curriculum; set `MTP_NUM_HEADS=0` for a direct ablation.
- If you want to isolate the training-side effect without the evaluation-side TTT stack, set `TTT_ENABLED=0`.
- Default `DATA_PATH` and `TOKENIZER_PATH` now work when launching from this candidate directory, assuming the repository's standard `data/` layout is present.
- EMA remains the active averaging path in this script; the inherited SWA bookkeeping is intentionally not part of the recommended command.

## Validation run in this workflow

Executed:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604051512_forward-mtp-curriculum/train_gpt.py
```

Outcome: **passed**.

Attempted an import-only smoke test for the candidate module, but this workflow environment is missing required runtime packages (`numpy` failed first, and the full runtime also expects the CUDA/FlashAttention stack), so a deeper CPU-only startup test was **not feasible here without adding infrastructure**.

## Main risks and tradeoffs

- **Training-time overhead**: even though the MTP heads are dropped before export, they still add compute during training; the candidate is betting that better sample efficiency beats fewer optimizer steps.
- **Small-model sensitivity**: the curriculum is meant to address this, but it remains possible that two future-token heads are still too aggressive for this parameter budget.
- **Interaction with TTT**: if MTP mostly improves already-strong next-token representations, its incremental gain after legal TTT may be smaller than its pre-TTT gain.
- **Compile behavior**: the curriculum is implemented via runtime-updated tensor weights to avoid the class-attribute constant-folding issue that previously broke late-QAT-style toggles in this repo.
