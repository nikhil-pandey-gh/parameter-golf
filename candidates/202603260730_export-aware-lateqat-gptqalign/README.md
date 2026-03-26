# Candidate: Export-aware Late QAT aligned to GPTQ-lite

## Hypothesis

The clean 11-layer GPTQ-lite/EMA stack still leaves some train-to-export mismatch on the table. If late fake-quantization uses the **same per-row clip-search family as export**, and if the script avoids the known `torch.compile` dead-code issue around late QAT, the final quantized artifact should track training more closely and improve post-export validation bits-per-byte without changing the architecture.

## Why this is promising for this repository

The repository's strongest record progression is dominated by **quantization-aware export improvements** rather than large architectural overhauls:

- fp16 embedding passthrough sharply reduced quantization damage;
- int6 / mixed-precision export unlocked bigger 10L and 11L models;
- GPTQ-lite clip search, EMA, and warmdown tuning improved the strongest clean 11-layer base;
- one prior 11-layer record explicitly notes that its late-QAT switch was neutralized by `torch.compile` constant folding.

That makes export-aware late QAT a good fit for this challenge: it directly targets the **final compressed artifact**, which is what the leaderboard actually measures, and it does so with a small, surgical change on top of an already strong recipe.

## Which records influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen as the base implementation. It is the strongest clean training/export stack before the more complex legal-TTT and parameter-banking additions.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Important cautionary precedent: its README documents that late QAT did not actually activate because of `torch.compile` behavior. This candidate is explicitly designed to address that failure mode.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Reinforced the lesson that recent wins are incremental and stack-clean: tiny, well-targeted changes matter when the base is already close to the frontier.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - Reinforced that export-sensitive tensors and quantization gaps are decisive in this repo.

There were **no prior `candidates/` directories** in the repository at the time this candidate was created, so the fork decision was based entirely on the record history.

## Which external research informed it

- **LSQ** — Esser et al., *Learned Step Size Quantization* (arXiv:1902.08153)
  - Useful background for why training should adapt to the quantizer actually used at export rather than a looser proxy.
- **SmoothQuant** — Xiao et al. (arXiv:2211.10438)
  - Reinforces the value of making inference-time quantization constraints visible during training or pre-export transformation.
- **AWQ** — Lin et al. (arXiv:2306.00978)
  - Supports the broader idea that weight-aware calibration choices matter materially for post-training quantization quality.
- **QuaRot** — Ashkboos et al. (arXiv:2404.00456)
  - Motivates export-aware outlier handling and quantization-friendly parameter geometry.
- **SpinQuant** — Liu et al. (arXiv:2405.16406)
  - Another example of targeted quantization-oriented weight transformation improving final low-bit deployment quality.
- **KurTail** — arXiv:2503.01483
  - Recent evidence that tail behavior and clip choice still matter in modern low-bit settings.

These papers do not imply a one-to-one implementation here. Instead, they motivate the narrower bet that **aligning the training-time fake quantizer with the final export quantizer** is a high-leverage change for this specific repository.

## What changed vs the chosen base implementation

Starting from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes the following changes:

1. Adds new knobs:
   - `QAT_REFRESH_EVERY`
   - `QAT_DISABLE_COMPILE`
   - `QAT_CANDIDATE_PCTS`

2. Extends `CastedLinear` with cached per-row `qat_scale` refresh logic so late fake-quant uses the same percentile-search family as GPTQ-lite export.

3. Adds `refresh_qat_scales(...)` to update cached QAT scales across the model.

4. Changes late QAT activation so that when it turns on:
   - scales are refreshed immediately;
   - the model can switch from compiled mode to eager mode to avoid dead-code elimination of the QAT branch;
   - scales refresh periodically during the late-QAT region.

5. Changes export quantization so cached QAT scales are evaluated as an **additional export candidate** alongside GPTQ-lite's normal percentile search.

6. Aligns the fake-quant clamp from `[-32, 31]` to `[-31, 31]` so training better matches final int6 export.

## How to run

From the repository root:

```bash
cd candidates/202603260730_export-aware-lateqat-gptqalign
RUN_ID=exportaware_qat_seed1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
QAT_REFRESH_EVERY=64 QAT_DISABLE_COMPILE=1 \
QAT_CANDIDATE_PCTS=0.999,0.9995,0.9999,0.99999,1.0 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

The script keeps the same overall evaluation flow as the 03-22 base:

- train under the 600-second wallclock cap;
- apply EMA before export;
- export a quantized int6 artifact;
- reload and evaluate the round-tripped artifact in-script;
- report sliding-window metrics when `EVAL_STRIDE` is enabled.

The key difference is that export can now reuse the cached late-QAT scales as one extra candidate when choosing the final quantizer.

## Main expected risks and tradeoffs

- **Wallclock risk**: switching from compiled to eager mode during late QAT may reduce step throughput enough to erase any quantization win.
- **Refresh cadence risk**: if `QAT_REFRESH_EVERY` is too large, cached scales may go stale; if it is too small, refresh overhead may dominate.
- **Diminishing-return risk**: GPTQ-lite export may already be close to optimal, so the achievable gain could be very small.
- **Stability risk**: making training more tightly match export can help the quantized artifact while slightly harming raw fp32/bf16 training loss.

## Validation

Commands run locally for this candidate:

```bash
python -m compileall candidates/202603260730_export-aware-lateqat-gptqalign/train_gpt.py
python -m py_compile candidates/202603260730_export-aware-lateqat-gptqalign/train_gpt.py
```

Observed outcomes in this environment:

- Both syntax checks **passed**.
- A deeper runtime smoke test was **not feasible on this runner** because the available `python` / `python3` interpreters do not have `torch`, `numpy`, or `sentencepiece` installed, so the training module cannot be imported here for an actual forward pass. That limitation is environmental rather than a syntax issue in the candidate script.
