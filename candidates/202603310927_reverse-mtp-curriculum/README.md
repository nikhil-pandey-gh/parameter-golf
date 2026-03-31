# Reverse-Curriculum MTP on the March 23 LeakyReLU2 + Legal TTT stack

## Hypothesis

The strongest local stack already has a dormant multi-token prediction (MTP) implementation that is never enabled, and those MTP heads are already excluded from the exported artifact. My hypothesis is that this repository can get a better train-time efficiency / eval-objective tradeoff by turning that path on as a **curriculum auxiliary loss**: ramp MTP up early enough to help representation learning, then fade it back toward pure next-token prediction before the end of training so EMA, quantization, and TTT are handed weights optimized for the actual scoring objective.

In short: use MTP as a training-only scaffold, not as a permanent objective.

## Why this looks promising for this repository

The records show a very consistent story:

- the big wins came from stacking low-byte, low-code changes onto the same 11-layer family: XSA, EMA, partial RoPE, GPTQ-lite, LeakyReLU(0.5)^2, and legal TTT;
- export quality is already highly optimized, so a good next move should improve learning without paying artifact bytes;
- several strong record scripts already contain MTP auxiliary heads, but leave them disabled and explicitly strip them from export.

That makes MTP unusually attractive here: it is already compatible with the repo's export path, it does not require broad infrastructure changes, and it attacks sample efficiency rather than just squeezing a few more bytes out of quantization.

## Prior records that influenced this candidate

The main local influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best overall stack; this candidate copies that script as the base;
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest clean non-TTT branch and a reminder that export-aware polish still matters;
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirmed that zero-parameter training/eval refinements on the mature 11L stack can still move the needle;
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - README explicitly shows `MTP_NUM_HEADS=0`, which helped confirm that the path exists but is not being used.

## External research that informed it

Primary sources considered:

- Fabian Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction* (arXiv:2404.19737)
  - argues that predicting multiple future tokens with independent heads on top of a shared trunk improves sample efficiency;
- Ansar Aynetdinov and Alan Akbik, *Pre-Training Curriculum for Multi-Token Prediction in Language Models* (arXiv:2505.22757)
  - especially relevant here because it focuses on smaller language models and shows that curriculum matters; notably, reverse curricula can improve next-token quality even when they give up self-speculative decoding benefits;
- Zhenzhong Lan et al., *ALBERT* (arXiv:1909.11942) and Mostafa Dehghani et al., *Universal Transformers* (arXiv:1807.03819)
  - both were considered because parameter sharing / recurrent depth are natural fits for a parameter-budget challenge, but they require a much more invasive architectural fork than this round seemed to justify.

I also looked at recent recurrent-depth work such as *SCORE: Replacing Layer Stacking with Contractive Recurrent Depth* (arXiv:2603.10544), but the repo history already suggests recurrence is a risky compute tradeoff under this 10-minute budget.

## What changed versus the chosen base implementation

Base: `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

- enable the existing MTP path by default with `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15`;
- add a lightweight MTP curriculum with three knobs:
  - `MTP_WARMUP_FRAC` (default `0.08`),
  - `MTP_DECAY_START_FRAC` (default `0.65`),
  - `MTP_FINAL_LOSS_WEIGHT` (default `0.0`);
- move MTP weighting onto a runtime buffer so the compiled model can read the current auxiliary weight each step instead of baking a fixed Python float into the trace;
- log the active MTP weight during training.
- resolve the default dataset and tokenizer paths from the repository root so the script can be launched directly from the candidate directory without extra path overrides.

Importantly, the candidate **does not** change the export rule that strips `mtp_heads` from the artifact before quantization, so the new idea is still training-only from the submission-size perspective.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603310927_reverse-mtp-curriculum

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_WARMUP_FRAC=0.08 \
MTP_DECAY_START_FRAC=0.65 MTP_FINAL_LOSS_WEIGHT=0.0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The only intentional new knobs are the four `MTP_*` settings. Everything else is the familiar March 23 stack, and this candidate does not rely on the inherited late-QAT path for its core hypothesis.

## Validation

From the repository root, I ran the lowest-cost validation that fits this repository and environment:

```bash
python -m compileall candidates/202603310927_reverse-mtp-curriculum/train_gpt.py
```

Outcome:

- passed.

I also attempted a no-training import smoke via `importlib`, but this runner does not currently have the repository's Python runtime dependencies installed (`numpy`, and likewise `torch` / `sentencepiece` are absent here). Because this candidate inherits the CUDA + `flash_attn_interface` training stack from the March 23 record, a real CPU-only start check is not feasible in this environment without installing the full training runtime and adding non-trivial CPU fallbacks.

## Main risks and tradeoffs

- MTP adds train-time compute, so even if it improves sample efficiency, it may reduce steps completed under the fixed 600s wallclock cap.
- The tiny-model literature is mixed: MTP helps, but small models appear to benefit much more from curricula than from naive fixed-weight objectives.
- This implementation decays the MTP loss weight to zero, but the compiled training graph still contains the MTP head projections whenever `MTP_NUM_HEADS > 0`; the objective alignment improves late in training, but the auxiliary compute cost is still present.
- TTT interactions are uncertain: MTP may improve the pre-TTT base model, but it could also reshape the weights in a way that changes how much legal TTT helps.

## Suggested next experiments

- sweep `MTP_NUM_HEADS` in `{1, 2}` and `MTP_LOSS_WEIGHT` in `{0.08, 0.12, 0.15}`;
- compare this reverse curriculum against a simpler forward-only schedule (`MTP_FINAL_LOSS_WEIGHT = MTP_LOSS_WEIGHT`) and a sharper reverse schedule (`MTP_DECAY_START_FRAC` around `0.5`);
- if the training-time hit is too large, port the same curriculum onto the March 22 non-TTT GPTQ-lite branch to isolate the MTP effect from the heavier evaluation stack.
