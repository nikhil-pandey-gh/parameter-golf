# Sandwich-Shared Core + Bigger Bigram

## Hypothesis

The repo's best non-TTT stack is already very strong at training-time-efficient compression-aware modeling, but it still pays full artifact cost for every transformer block. A better tradeoff may be to **share only the early/mid heavy block weights**, keep **small layer-specific adapters** per depth, and leave the **late XSA/value-embedding tail unique**. That should preserve most of the 11-layer compute path while spending fewer unique bytes on repeated middle-layer matrices.

The saved artifact budget is then reinvested in a **larger BigramHash table (2048 -> 4096)**, which is one of the few late-stack levers that still showed gains in the records.

## Why this is promising here

Repository review suggested three relevant facts:

1. The strongest non-TTT core model is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, which already combines 11 layers, partial RoPE, XSA on the deepest layers, VE, EMA, GPTQ-lite, and int6 export well.
2. The current best overall record, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, found that **LeakyReLU(0.5)^2** and a modest **bigger BigramHash** both still helped on top of the frontier stack.
3. A non-record 1x5090 sweep found that **naive layer recurrence x2 was much worse**, so the right follow-up is not blunt recurrence, but a more careful **sandwich-style sharing** that keeps the deepest specialized layers unique and preserves per-layer control parameters.

## Prior runs that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- **Activation carry-over:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
- **Bigram-size signal:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
- **Negative result to avoid repeating:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` (`Layer recurrence x2`)

There were **no prior `candidates/` experiments** in the repo when this candidate was created.

## External research that informed it

- **Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers** (Reid et al., 2021, arXiv:2101.00234) argues that **sandwich-style sharing** works better than naive cross-layer sharing for generative transformers.
- **Basis Sharing: Cross-Layer Parameter Sharing for Large Language Model Compression** (Wang et al., 2024, arXiv:2410.03765) shows that structured cross-layer sharing is especially attractive under **high compression ratios**.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (Nguyen and Lin, 2025, arXiv:2505.01855) reports that **earlier-layer recurrence is the best place to spend repeated computation**.
- **ALBERT** (Lan et al., 2019, arXiv:1909.11942) is the classic reminder that cross-layer sharing can materially reduce parameter cost while retaining depth.

The implementation here adapts those ideas to this repo's constraints instead of copying any paper literally.

## What changed vs. the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. Replaces per-layer full blocks with a **sandwich-shared stack**:
   - default `SANDWICH_PREFIX_LAYERS=1`
   - default `SANDWICH_SHARED_BLOCKS=3`
   - default `SANDWICH_SUFFIX_LAYERS=4`
   - effective default core map: `0,1,2,3,1,2,3,4,5,6,7`
2. Keeps **layer-specific adapters** for:
   - RMSNorm placement,
   - `q_gain`,
   - `attn_scale`,
   - `mlp_scale`,
   - `resid_mix`,
   - optional DTG gate.
3. Keeps the **late specialized tail unique**, so the last four XSA layers are not forced into the shared middle.
4. Switches the MLP to **LeakyReLU(0.5)^2**.
5. Increases the default **BigramHash vocabulary from 2048 to 4096**.
6. Makes the default dataset/tokenizer paths resolve from the **candidate directory**, so `python train_gpt.py` works when launched from this folder.

## How to run

From this directory:

```bash
cd candidates/202604050803_sandwich-shared-core
RUN_ID=sandwich_shared_core \
SANDWICH_PREFIX_LAYERS=1 \
SANDWICH_SHARED_BLOCKS=3 \
SANDWICH_SUFFIX_LAYERS=4 \
BIGRAM_VOCAB_SIZE=4096 \
LEAKY_RELU_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults already match that configuration, so the env vars above are mainly there to make the intended recipe explicit.

## Main risks / tradeoffs

- **Sharing may over-couple middle layers.** Even with per-layer adapters, the shared cores may reduce specialization too much.
- **Compute is not reduced proportionally.** The model still executes an 11-layer path; the main benefit is artifact/optimizer-state efficiency, not a free speedup.
- **Quantization may move differently under sharing.** Reused cores could quantize better because they are smoother, or worse because they absorb more mixed roles.
- **Bigger bigram tables may saturate.** The records show some benefit from larger tables, but the exact sweet spot on this shared stack is unknown.

## Validation recorded for this candidate

1. `python -m compileall candidates/202604050803_sandwich-shared-core/train_gpt.py`  
   - **Passed**
2. Import-and-construct CPU smoke attempt using a tiny `GPT(...)` instantiation  
   - **Blocked in this environment** before model construction because the repo's Python runtime dependencies are not installed locally (`ModuleNotFoundError: numpy` on import).  
   - Since the script also expects the challenge CUDA stack, no stronger local runtime smoke test was feasible here without additional environment setup.

## Suggested next experiments

1. Sweep the sandwich layout:
   - `(prefix, shared, suffix) = (1,2,4), (1,3,4), (2,2,4), (2,3,3)`
2. Re-test `BIGRAM_VOCAB_SIZE` at `3072`, `4096`, and `6144`.
3. Compare this shared-core stack both **with** and **without** legal TTT once the pre-TTT quality is known.
