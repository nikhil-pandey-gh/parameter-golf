# Candidate: MTP auxiliary heads on the 11L GPTQ-lite / EMA stack

## Hypothesis

Add **train-only multi-token prediction (MTP) heads** to the strongest pre-TTT 11-layer lineage so the shared trunk learns a richer predictive signal per token without paying extra artifact bytes at export time. Pair that with the now-proven **LeakyReLU(0.5)^2** MLP from the latest record, which should help the shared trunk use the auxiliary supervision more effectively.

## Why this is promising for this repository

- The repository is **artifact-limited**, not just parameter-limited. MTP heads are useful here because they improve training supervision but are explicitly **dropped from the exported checkpoint**, so the final 16 MB budget is unchanged.
- The current best record already shows that a small activation change can move BPB materially, and the safest reusable base is the `2026-03-22` 11L GPTQ-lite / EMA stack.
- The candidate codebase already had dormant MTP support, but no record README or command actually enabled it. This makes MTP a strong next step that is **new for this repo's recorded experiments** while still staying close to a proven implementation.

## Prior repo work that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - safest non-TTT 11L base: GPTQ-lite, EMA, warmdown 3500, SmearGate, BigramHash, XSA4, Partial RoPE, LN scale.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms Partial RoPE + LN scale were real gains and that compile-time toggle bugs can invalidate late-training tricks.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - shows **LeakyReLU(0.5)^2** was worth about `-0.0021` BPB on a closely related stack.

## External research

1. **Gloeckle et al., “Better & Faster Large Language Models via Multi-token Prediction”** (arXiv:2404.19737)
   - argues that predicting multiple future tokens with separate heads on a shared trunk improves sample efficiency, with especially strong gains on generative tasks.
2. **DeepSeek-V3 Technical Report** (arXiv:2412.19437)
   - reports a multi-token prediction training objective as part of a strong modern recipe, reinforcing that MTP is not just an inference trick.
3. **Medusa** (arXiv:2401.10774)
   - inference-focused, but still useful evidence that extra prediction heads on a shared backbone are practical and can preserve main-model quality when trained carefully.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. **MTP enabled by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
   - the MTP heads are still excluded from the exported artifact, matching the existing export discipline in the base script.
2. **LeakyReLU(0.5)^2 MLP**
   - backported from the latest record via `LEAKY_RELU_SLOPE=0.5`.
3. **FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA so the script can still be imported or smoke-tested in lighter environments.
4. **Optional CPU smoke mode**
   - `SMOKE_TEST=1` runs a tiny forward/backward plus export-roundtrip check without dataset or GPU requirements.

## How to run

From this candidate directory:

```bash
RUN_ID=mtp_aux_heads \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
LEAKY_RELU_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults already enable the candidate idea, so the explicit `MTP_*` and `LEAKY_RELU_SLOPE` flags above are mainly for readability.

For a lightweight local smoke check:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Main risks and tradeoffs

- **Training overhead:** MTP adds extra output heads and losses. If the added supervision does not beat the lost steps under the 600 s cap, the candidate regresses.
- **Scale sensitivity:** the strongest published MTP gains are often reported on larger models; tiny 16 MB models may need only 1-2 heads and a smaller loss weight.
- **Objective mismatch:** evaluation is still next-token BPB. If the auxiliary task dominates too late into training, it could slightly hurt the main objective.

## Validation

Commands and outcomes for this candidate:

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604192314_mtp-aux-heads/train_gpt.py`
  - **passed**
- `SMOKE_TEST=1 python candidates/202604192314_mtp-aux-heads/train_gpt.py`
  - **passed** with `smoke_test:ok loss:5.5749 logits_shape:(2, 32, 128) mtp_heads:2`
