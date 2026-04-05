# Zero-Byte MTP on the Current Best Stack

## Hypothesis

A small **multi-token prediction (MTP)** auxiliary loss should improve sample efficiency on the strongest current Parameter Golf backbone without consuming submission bytes, because the extra head is used only during training and is dropped before export.

In this repo, that trade is unusually attractive: the leaderboard is already near the point of diminishing returns on evaluation tricks and post-training quantization, while the best public stack is also very close to the 16 MB artifact cap. MTP spends a bit more training compute, not artifact budget.

## Why this is promising here

The record history suggests three things:

1. **Big wins from evaluation and quantization have mostly been harvested.** Sliding-window eval, int6 export, GPTQ-lite, EMA/SWA, and Partial RoPE/XSA refinements all helped, but recent gains are increasingly sub-millibpb.
2. **The strongest runs are artifact-constrained.** The current best stack already leans on int6+lzma and careful compression-aware training, so adding bytes is expensive.
3. **Training-only improvements are underexplored.** The strongest scripts already contain dormant `MTP_*` hooks and explicitly exclude `mtp_heads` from export, but no reviewed record actually turns them on.

That makes MTP a good next bet: it is novel in this repo, grounded in primary-source literature, and naturally aligned with the challenge's byte budget.

## Prior repository runs that influenced this candidate

- `records/track_10min_16mb/2026-03-17_NaiveBaseline/` established the base 9L/512d tied-embedding setup and artifact accounting.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/` showed that evaluation protocol was a huge early unlock, so a new candidate should avoid spending its novelty budget there.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`, `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`, and `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` showed the winning pattern: 11 layers, 3x MLP, compression-aware training, zero-parameter deep-layer attention tweaks, and tiny post-training quantization gains.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the direct base here. It is the strongest public stack, uses BigramHash(1536), legal score-first TTT, and already contains a training-only MTP path that exports zero bytes.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` is a useful negative-result reminder that wallclock matters more than architectural novelty if the idea costs too many steps.

## External research

- **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737)** argues that predicting multiple future tokens with independent heads improves sample efficiency and helps induction-like behaviors emerge earlier.
- **DeepSeek-AI, "DeepSeek-V3 Technical Report" (arXiv:2412.19437)** explicitly cites a multi-token prediction training objective as one contributor to stronger model performance.

This candidate deliberately uses the lightweight version that best matches this repo's constraints: one extra future-token head on top of the existing trunk, exported away after training.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enable MTP by default** with `MTP_NUM_HEADS=1`. This adds a single auxiliary head that predicts one token beyond the standard next-token target.
2. **Keep the extra head zero-byte at export.** The script retains the base behavior that strips `mtp_heads` from the exported state dict before compression.
3. **Align defaults to the current best public stack** by defaulting `BIGRAM_VOCAB_SIZE=1536`, `TTT_ENABLED=1`, and `TTT_FREEZE_BLOCKS=0`.
4. **Add a safe SDPA fallback** when `flash_attn_interface` is unavailable, so the model can at least be imported and smoke-tested on CPU in environments that have the normal Python deps installed but not FlashAttention 3.

## How to run

From this candidate directory:

```bash
RUN_ID=zero_byte_mtp \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

- Disable the idea entirely: `MTP_NUM_HEADS=0`
- Compare TTT-on vs TTT-off: `TTT_ENABLED=0`
- Sweep auxiliary strength: `MTP_LOSS_WEIGHT=0.1` or `0.2`

## Expected risks and tradeoffs

- **Training-time overhead:** even one extra vocab head costs FLOPs, so the model may take fewer steps inside the 600s wallclock.
- **Quantization transfer risk:** a better pre-quant trunk does not guarantee a better post-quant `val_bpb`.
- **Objective mismatch risk:** this is the simple independent-head MTP variant, not the heavier sequential MTP modules used in some newer systems papers.
- **TTT interaction risk:** MTP could help the base model but wash out after legal TTT, or vice versa.

## Validation

Commands run in this repository:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604052312_zero-byte-mtp/train_gpt.py
```

Outcome:

- `compileall` completed successfully.

Attempted but not completed here:

```bash
python - <<'PY'
import torch
import sentencepiece
PY
```

- A direct CPU smoke import was **not feasible in this container** because the available Python environment does not have the runtime deps (`torch`, `sentencepiece`) installed.
- The candidate script still adds a CPU-safe SDPA fallback so that forward-pass smoke tests are possible in a provisioned environment without FlashAttention 3.
