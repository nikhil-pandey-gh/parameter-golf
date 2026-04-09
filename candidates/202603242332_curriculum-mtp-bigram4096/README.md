# Curriculum MTP + Bigram4096 on the 2026-03-22 winner

## Hypothesis

The strongest next step is to keep the current 11-layer EMA + GPTQ-lite winner almost unchanged, then spend the remaining training budget on a **training-only multi-token prediction (MTP) auxiliary** rather than another costly architectural change.

The candidate enables 2 auxiliary future-token heads during training, but ramps their contribution in with a **forward curriculum** so the model starts as plain next-token prediction and only gradually pays the harder MTP objective. To give the backbone a slightly richer cheap lexical prior, it also restores the BigramHash table from 2048 buckets to **4096** buckets.

## Why this is promising for this repository

This repository's best recent gains already come from reusing the same backbone and stacking cheap, composable improvements:

- better evaluation context (`SlidingWindowEval`, then every top run after it),
- smoother averaging (`EMA` replacing `SWA`),
- sharper post-training quantization (`GPTQ-lite`),
- stronger but still cheap token-pair priors (`SmearGate` + `BigramHash`),
- and zero-byte architectural tweaks like Partial RoPE and LN scaling.

MTP fits that pattern unusually well:

- it is **training-only**, because the script already excludes `mtp_heads` from export,
- it directly targets **sample efficiency**, which matters under the fixed 600-second wallclock,
- and it needs only a small code change on top of the current best implementation.

The repo review also surfaced that the current winning script already contains dormant `MTP_NUM_HEADS` / `MTP_LOSS_WEIGHT` hooks, making this one of the highest-upside ideas that does **not** require broad new infrastructure.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Most relevant lineage:

- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` established the 11-layer EMA + XSA family.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` showed Partial RoPE + LN scaling were real wins.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` added GPTQ-lite clip search, warmdown 3500, and the current value-embedding setup.
- `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` showed that **larger BigramHash tables** still helped after the big architectural wins, with measurable gains from `4096 -> 8192 -> 10240`.

## External research that informed it

- **Better & Faster Large Language Models via Multi-token Prediction** (Gloeckle et al., 2024, arXiv:2404.19737) argues that adding multiple future-token heads on top of a shared trunk improves sample efficiency while keeping the core language-model objective intact.
- **Pre-Training Curriculum for Multi-Token Prediction in Language Models** (Aynetdinov & Akbik, 2025, arXiv:2505.22757) is especially relevant here because it reports that **small language models struggle with raw MTP**, and that a **forward curriculum** helps smaller models benefit from MTP more reliably.

That combination matches this challenge well: the artifact budget is strict, the training budget is short, and the model is small enough that the curriculum detail may matter more than it would in larger LLMs.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four focused changes:

1. **Enable MTP by default** with `MTP_NUM_HEADS=2`.
2. **Add a forward MTP curriculum** with `MTP_WARMUP_STEPS=1500`, keeping the auxiliary branch off at zero weight and then ramping it to `MTP_LOSS_WEIGHT=0.10` over the early part of training.
3. **Increase `BIGRAM_VOCAB_SIZE` from 2048 to 4096** to reinvest a bit more capacity into the already-proven token-pair prior.
4. Add a **FlashAttention capability/runtime fallback** to PyTorch SDPA, while leaving non-flash SDPA backends enabled so the model code can be imported and locally smoke-tested on non-Hopper / non-FA3 setups when `torch` is available.

Importantly, the candidate leaves the base 11-layer stack intact:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- XSA on the last 4 layers
- Partial RoPE (`16/64`) + LN scale
- EMA + warmdown3500
- VE on layers `9,10`
- GPTQ-lite export quantization

## How to run or evaluate it

Defaults in `train_gpt.py` are already set for the candidate, so the simplest launch is:

```bash
cd candidates/202603242332_curriculum-mtp-bigram4096
RUN_ID=curriculum_mtp_bigram4096 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs to sweep first if this underperforms:

```bash
MTP_NUM_HEADS=1|2|4
MTP_LOSS_WEIGHT=0.05|0.10|0.15
MTP_WARMUP_STEPS=1000|1500|2500
BIGRAM_VOCAB_SIZE=4096|8192
```

## Validation run in this workflow

### Passed

```bash
python -m compileall candidates/202603242332_curriculum-mtp-bigram4096/train_gpt.py
```

Outcome: **passed**.

### Attempted but blocked by environment

Attempted a tiny CPU import/forward smoke test that instantiated `GPT(...)` with a toy shape and exercised the new MTP path. That was blocked because this workflow environment does not have `torch` installed for either `python` or `python3`, so the script cannot be imported here even though it compiles successfully.

The exact limitation observed was:

```text
ModuleNotFoundError: No module named 'torch'
```

Because of that, I could not run a real CPU forward pass in this runner. The added SDPA fallback is still useful for future local smoke checks in environments that do have PyTorch but not FlashAttention.

## Main expected risks or tradeoffs

- **Throughput risk:** even lightweight MTP heads still add some training compute. If step throughput drops too much, the extra supervision may not pay for itself within 600 seconds.
- **Small-model sensitivity:** the 2025 curriculum paper is encouraging, but it also highlights that small models can struggle with MTP. The ramp helps, but this may still need tuning.
- **Bigram table tradeoff:** `4096` is a conservative bump, but it still spends some bytes and may have diminishing returns on the stronger 11-layer backbone.
- **Artifact-size uncertainty:** this should still fit under 16 MB based on the 2026-03-22 winner's margin and the modest BigramHash increase, but only a real GPU export run can confirm the final compressed bytes.

## Suggested next experiments

If this candidate is close but not clearly better, the next experiments I would run are:

1. keep the MTP curriculum and sweep `MTP_NUM_HEADS` / `MTP_LOSS_WEIGHT` first,
2. compare `BIGRAM_VOCAB_SIZE=4096` against `8192`,
3. if MTP helps but overhead is noticeable, try `MTP_NUM_HEADS=1` with a slightly larger weight,
4. and only then combine this with dormant low-byte hooks from the winner like `DTG_ENABLED=1` or a broader `VE_LAYERS` sweep.
