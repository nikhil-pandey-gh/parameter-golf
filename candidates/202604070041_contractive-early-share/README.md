# Contractive Early-Shared Encoder

## Hypothesis

The clean 11-step `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack can be made more parameter-efficient by **sharing only the earliest encoder blocks**, then applying the repeated passes with a **contractive residual update** instead of a full-strength extra block application.

Concretely, this candidate keeps the base run's **11 logical block applications**, but reduces them to **9 unique blocks**:

- encoder logical schedule: `[0, 0, 1, 1, 2]`
- decoder logical schedule: `[3, 4, 5, 6, 7, 8]`

Only the second pass of each repeated early block is contractive:

```python
x_next = block(x)
x = torch.lerp(x, x_next, dt)
```

This is meant to preserve the base stack's compute budget while testing whether early recurrent refinement is the missing version of recurrence that fits this repository.

## Why this is promising here

Recent repo history says two things at once:

1. **The 11-layer pre-TTT stack is very strong**: EMA, GPTQ-lite, partial RoPE, LN scale, XSA-last-4, SmearGate, BigramHash, VE, and warmdown tuning already reach `1.1233` without score-first TTT.
2. **Full-model looping looked too expensive / ineffective** in earlier experiments.

This candidate targets the gap between those observations:

- keep the best clean pre-TTT stack,
- only recur in the **earliest encoder stages**,
- keep total logical depth at **11** rather than adding more compute than the strong baseline,
- use a **contractive** update on the repeated pass to avoid the instability / over-iteration failure mode.

## Prior repository evidence that influenced this candidate

### Chosen base implementation

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest clean non-TTT base
  - already bundles the repo's most repeatable training/export wins

### Positive trends carried forward

- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - partial RoPE + LN scale mattered
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - deep-layer XSA + EMA mattered
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - LeakyReLU(0.5)^2 looked like a real pre-TTT gain, so I ported it without the heavier TTT / parameter-banking stack

### Dead ends / caution signals this candidate explicitly responds to

- `2026-03-18_FP16Embed_WD3600`
  - notes that full depth recurrence was promising in principle but not viable under the 10-minute step budget
- `2026-03-19_SlidingWindowEval`
  - contains unused looped-architecture code paths, confirming recurrence existed in the repo but was not part of the winning path

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

Primary sources:

1. **ALBERT** — <https://arxiv.org/abs/1909.11942>  
   Cross-layer sharing can preserve quality while reducing parameters and memory.
2. **Universal Transformer** — <https://arxiv.org/abs/1807.03819>  
   Recurrent depth can give a better inductive bias than purely feed-forward stacking.
3. **Intra-Layer Recurrence in Transformers for Language Modeling** — <https://arxiv.org/abs/2505.01855>  
   Finds that recurrence is most effective when allocated to **earlier layers**, which directly motivated repeating only the earliest encoder blocks here.
4. **SCORE: Replacing Layer Stacking with Contractive Recurrent Depth** — <https://arxiv.org/abs/2603.10544>  
   Motivated the contractive `x <- lerp(x, block(x), dt)` refinement step instead of a raw second full-strength pass.
5. **Universal YOCO for Efficient Depth Scaling** — <https://arxiv.org/abs/2604.01220>  
   Supports the idea that recursion should be confined to the shallow / cheaper part of the network rather than applied everywhere.

## What changed vs the chosen base

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Early shared encoder schedule**
   - from 11 unique blocks
   - to 9 unique blocks with 11 logical applications via `[0, 0, 1, 1, 2 | 3, 4, 5, 6, 7, 8]`
2. **Contractive recurrent pass**
   - only the repeated encoder passes use learned `sigmoid(dt)` interpolation
3. **LeakyReLU(0.5)^2 MLP**
   - ports the low-risk activation win from the 2026-03-23 record into the clean pre-TTT stack
4. **CPU-safe attention fallback**
   - if `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA so local import/smoke runs are easier once `torch` is installed

Intentionally unchanged from the base stack:

- GPTQ-lite int6 export
- EMA + tight SWA
- partial RoPE
- LN scale
- XSA on the last 4 logical layers
- SmearGate + BigramHash
- shared VE on the final logical layers
- seq2048 / batch786432 / warmdown3500 family

## How to run

From this directory:

```bash
NUM_LAYERS=9 \
ENCODER_UNIQUE_LAYERS=3 \
SHARED_ENCODER_LAYERS=2 \
SHARED_ENCODER_REPEATS=2 \
CONTRACTIVE_INIT=0.75 \
LEAKY_RELU_SLOPE=0.5 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate / inspect

- The script keeps the base export/eval flow: train -> EMA/SWA selection -> mixed int6 export -> exact roundtrip eval.
- The log now reports:
  - `unique_layers`
  - `effective_layers`
  - encoder schedule
  - which logical layers have XSA

## Main expected risks and tradeoffs

1. **Too much sharing too early** may remove genuinely useful unique encoder capacity; if so, the next sweep should try `NUM_LAYERS=10` or `SHARED_ENCODER_LAYERS=1`.
2. **Contractive recurrence may be too conservative**, especially with `CONTRACTIVE_INIT=0.75`; if underfitting shows up, try a larger initial `dt`.
3. **LeakyReLU^2 and shared recurrence interact**, so if results move in the wrong direction the first ablation should be recurrence-on/off while keeping LeakyReLU^2 fixed.
4. The candidate likely leaves **artifact headroom**, which can fund a future sweep on `BIGRAM_VOCAB_SIZE`, VE placement, or width.

## Validation run for this candidate

Commands run in this workspace:

```bash
python -m compileall candidates/202604070041_contractive-early-share/train_gpt.py
```

Outcome:

- **Passed**

Attempted lightweight smoke path:

- I did **not** run a CPU forward-pass smoke test in this workspace because the default Python environment here does not include the runtime dependencies needed to import the script (`torch` is missing). I therefore limited validation to syntax compilation only. The script now includes a non-FlashAttention fallback specifically so a small CPU import/forward smoke test is easier in a proper local environment with `torch` installed.
