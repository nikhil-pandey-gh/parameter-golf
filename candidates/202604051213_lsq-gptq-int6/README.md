# 202604051213_lsq-gptq-int6

## Hypothesis

Warm-started **learned per-row int6 step sizes** during the late-QAT phase should shrink the train-to-export quantization gap better than the current fixed fake-quant path, while staying close to the strongest non-TTT 11-layer stack already proven in this repository.

This candidate also folds in two cheap repo-proven tweaks from the current SOTA lineage:

1. **LeakyReLU(0.5)^2** in the MLP instead of ReLU^2.
2. **BigramHash 3072** instead of 2048, while keeping the rest of the mature 11L GPTQ-lite/XSA/EMA/partial-RoPE recipe intact.

## Why this looks promising here

Repository history says the remaining gains are mostly about **export quality**, not just raw pre-quant loss:

- the 4-hour non-record run still landed at a poor post-quant score, which shows compression/export is the real bottleneck,
- GPTQ-lite already bought a measurable win on the current 11-layer line,
- earlier late-QAT wiring existed, but one strong record explicitly notes it was neutralized by `torch.compile` constant-folding rather than by the idea itself.

So the best next low-infrastructure bet is to keep the strongest 11-layer architecture and make the quantizer itself more trainable.

## Prior repository experiments that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen base implementation: mature 11L XSA/EMA/partial-RoPE/LN-scale/VE/GPTQ-lite stack.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - imported the LeakyReLU(0.5)^2 MLP activation and the larger 3072-bucket bigram table idea.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - important negative result: its README documents that the late-QAT path was dead due to compile-time constant folding, so this candidate avoids that exact failure mode.

There were **no prior `candidates/` directories** in the repo when this candidate was created.

## External research that informed it

- **EfficientQAT** — <https://arxiv.org/abs/2407.11062>  
  Motivated splitting standard training from a quant-parameter-focused phase and treating quantizer parameters as first-class trainables.
- **LSQ: Learned Step Size Quantization** — <https://arxiv.org/abs/1902.08153>  
  Motivated learning the quantizer step size directly instead of fixing it from row maxima.
- **PACT** — <https://arxiv.org/abs/1805.06085>  
  Motivated the broader “learn the clipping/quantizer, don’t hand-code it” framing.
- **QAT scaling law / outlier analysis** — <https://arxiv.org/abs/2505.14302>  
  Reinforced that quantization error can remain the main bottleneck late in training and that weight-side quantization quality matters.

## What changed versus the chosen base

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. replaces the dead class-flag late-QAT path with **learned per-row step sizes** stored on each `CastedLinear`,
2. reinitializes those quantizer scales from the existing GPTQ-lite percentile search right when late-QAT begins,
3. ramps late-QAT strength numerically instead of switching a compile-time boolean,
4. reuses the learned scales during final int6 export instead of discarding them and re-running pure static search,
5. switches the MLP activation to **LeakyReLU(0.5)^2**,
6. raises `BIGRAM_VOCAB_SIZE` from **2048** to **3072** by default,
7. adds a **FlashAttention fallback to PyTorch SDPA** so the script is less brittle outside the exact Hopper environment,
8. resolves dataset/tokenizer defaults relative to the **repository root**, so the script can be launched from this candidate directory directly.

## How to run

From the repository root:

```bash
cd candidates/202604051213_lsq-gptq-int6
RUN_ID=lsq_gptq_int6 \
LEARNED_QAT_ENABLED=1 \
LATE_QAT_THRESHOLD=0.15 \
MLP_NEGATIVE_SLOPE=0.5 \
BIGRAM_VOCAB_SIZE=3072 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The default `DATA_PATH` and `TOKENIZER_PATH` resolve back to the repo-root `data/` directory automatically, so no extra path overrides are needed if the standard challenge cache is already present.

## Evaluation notes

- The script keeps the base stack’s int6 roundtrip export and sliding-window evaluation.
- If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA instead of failing at import time.
- Export excludes the learned QAT scale parameters themselves; only the learned int6 scales are reused for quantizing the real weights.

## Main risks and tradeoffs

- The learned fake-quant path adds extra training compute; if the overhead is too high, step count may fall enough to erase the quantization win.
- Because the learned scales are always present as trainable parameters, optimizer noise or poor late initialization could hurt otherwise good weights.
- The larger bigram table helps if extra short-range lexical capacity matters, but it also consumes some artifact headroom.
- The candidate is still weight-only int6 focused; if the main bottleneck has shifted to a small set of especially sensitive tensors, a selective fp16/int8 passthrough search could beat it.

## Validation run in this workflow

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604051213_lsq-gptq-int6/train_gpt.py` | Passed |
| Minimal CPU smoke test | Not run: this workflow environment did not have `torch`, `numpy`, or `sentencepiece` installed, and it also lacked cached FineWeb shards / GPU access needed for a realistic startup check |
