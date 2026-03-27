# Candidate: 1-head MTP on the LeakyReLU2 + Legal TTT stack

## Hypothesis

The current record stack already looks close to locally optimal on architecture, evaluation, and compression. The cleanest untried lever is to improve **sample efficiency during the same 600-second training budget** by turning on a **single training-only multi-token prediction (MTP) head**.

The idea is that one extra future-token objective should sharpen the shared trunk's representations without changing the exported model. If it works, we get a better main next-token model for nearly zero artifact cost.

## Why this looks promising for this repository

Repository history suggests that the biggest recent gains came from:

- better evaluation protocol (`SlidingWindowEval`, legal TTT),
- better compression / quantization (`GPTQ-lite`, int6), and
- small, targeted architectural changes on top of the strongest 11-layer stack (XSA, Partial RoPE, LN scale, LeakyReLU2).

What has **not** been tried in any record or prior candidate is a non-zero MTP setting, even though the strongest recent scripts already contain MTP support. That makes MTP unusually attractive here: it is differentiated, low-risk to implement, and directly aligned with the challenge's fixed-time / fixed-artifact constraints.

## Prior repository work that informed this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current stack,
  - adds LeakyReLU(0.5)^2,
  - uses legal score-first TTT,
  - already includes parameter banking + Parallel Muon,
  - already has dormant MTP support that was never enabled.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - shows the importance of post-training quantization quality and EMA-based weight smoothing.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms that small, parameter-free attention changes can still move BPB on the mature 11-layer stack.

- There were **no existing `candidates/` experiments** in the repository when this candidate was created.

## External research that informed it

- **Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"** (`arXiv:2404.19737`)
  - reports that MTP improves sample efficiency when trained as an auxiliary objective on top of a shared trunk.

- **DeepSeek-V3 Technical Report** (`arXiv:2412.19437`)
  - includes a multi-token prediction training objective in a state-of-the-art production model, which is strong evidence that the idea scales beyond toy settings.

- **Mehra et al., "On multi-token prediction for efficient LLM inference"** (`arXiv:2502.09419`)
  - finds that MTP heads are harder to retrofit onto a frozen next-token model than to learn jointly with the backbone. That supports trying MTP here during end-to-end training rather than as a post-hoc add-on.

- **Zhao et al., "Self-Distillation for Multi-Token Prediction"** (`arXiv:2603.23911`)
  - reinforces the recent trend toward making MTP practical with minimal extra training cost while preserving main-head quality. This motivated the conservative choice here: just **1 extra head** and a **modest loss weight**.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- enable **1 MTP head by default** with `MTP_NUM_HEADS=1`,
- reduce the auxiliary weight to `MTP_LOSS_WEIGHT=0.15` to stay conservative,
- set defaults closer to the published top-record recipe by defaulting to `BIGRAM_VOCAB_SIZE=1536`, `TTT_ENABLED=1`, and `TTT_FREEZE_BLOCKS=0`,
- make dataset and tokenizer defaults **repo-root relative**, so the script can be run directly from this candidate directory,
- document in code that MTP heads are **training-only** and are excluded from export.

## How to run / evaluate

Prerequisite: populate the standard repository dataset/tokenizer paths first, for example from the repository root:

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

Then, from this candidate directory:

```bash
cd candidates/202603271919_one-step-mtp
SEED=1337 RUN_ID=one_step_mtp_seed1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15
BIGRAM_VOCAB_SIZE=1536
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0
```

Because the script now resolves `DATA_PATH` and `TOKENIZER_PATH` from the repository root by default, the command above works from this directory once those standard repository paths have been populated.

## Expected risks / tradeoffs

- MTP increases training compute slightly, so the extra objective has to improve sample efficiency enough to pay for fewer effective trunk updates.
- Recent MTP literature also warns that preserving main next-token quality is not automatic, especially if the auxiliary loss is too strong.
- Legal TTT dominates final eval quality on the current frontier, so some gains may only show up after a real 8xH100 run.
- This candidate intentionally does **not** add newer MTP variants like distillation or register tokens; it is a minimal first pass on the already-implemented auxiliary-head path.

## Validation

- `python3 -m compileall candidates/202603271919_one-step-mtp/train_gpt.py`
  - **Passed**

- Targeted code review of the candidate diff
  - initially found one documentation issue around the data-path prerequisite in this README,
  - fixed in the run instructions above,
  - no remaining code-level issues surfaced.

- Tiny CPU-only import / forward smoke test with a temporary `flash_attn_interface` stub
  - **Not feasible in this container** because the available local Python environment does not have `torch` installed, so `import torch` fails before the model can be instantiated.
