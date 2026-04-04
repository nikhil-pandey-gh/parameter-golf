# 202604042114_lsq-lite-qat

## Hypothesis

The current 11-layer GPTQ-lite stack is already strong on modeling quality, but it still spends measurable BPB on the final int6 export step. A **trace-safe late LSQ-style QAT** should reduce that gap by letting the model learn per-row int6 step sizes during the last slice of training instead of relying only on fixed row-max STE or post-hoc clip search.

## Why this looks promising here

- Recent repo wins increasingly come from **better quantization/export**, not just more architecture changes: fp16 embed passthrough, mixed int5/int6, GPTQ-lite, EMA, and late-stage schedule tuning all helped (`records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`, `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md`, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`).
- The 2026-03-21 record explicitly documented that its late-QAT flag never activated because `torch.compile` constant-folded the class-level toggle (`records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`).
- LSQ and OmniQuant both argue that **quantizer parameters should be learned**, not fixed by hand, especially at low bitwidths:
  - LSQ: https://arxiv.org/abs/1902.08153
  - OmniQuant: https://arxiv.org/abs/2308.13137
- AWQ and FlatQuant reinforce the same broader lesson from the PTQ side: low-bit success is driven by treating quantization as a first-class optimization target rather than a passive export afterthought:
  - AWQ: https://arxiv.org/abs/2306.00978
  - FlatQuant: https://arxiv.org/abs/2410.09426

## Prior local work that shaped this candidate

- **Chosen base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- **Direct cautionary predecessor:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` for the `torch.compile` late-QAT bug note.
- **No earlier candidates influenced this** because `candidates/` did not exist before this run.

## What changed vs the base implementation

1. Added **LSQ-lite late QAT** to block attention/MLP `CastedLinear` layers via a learnable per-row `qat_log_step`.
2. Replaced the old class-level QAT flag with a **per-module tensor mix ramp** (`qat_mix`) so the fake-quant path stays visible to `torch.compile`.
3. Initialized late-QAT step sizes from each row's 99.99th percentile magnitude, then optimized them with the existing scalar optimizer group.
4. Kept the existing GPTQ-lite-style export, but now let export reuse the learned QAT scales as extra candidates for the final int6 row scales **only if late QAT actually activated**.
5. Excluded QAT-only parameters from the exported checkpoint so the artifact budget still reflects the real inference model.

## How to run

From this directory:

```bash
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
QAT_ENABLED=1 LATE_QAT_THRESHOLD=0.15 QAT_MIX_POWER=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This inherits the same dataset/tokenizer expectations as the record base:

- `DATA_PATH=./data/datasets/fineweb10B_sp1024/`
- `TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model`

## Expected tradeoffs / risks

- The extra fake-quant math may cost some throughput during warmdown; if the slowdown is too large, the net win can disappear.
- Learned step sizes may overfit to the late-training regime and still lose to pure GPTQ-lite at export on some seeds.
- This only QATs the large block matrices; embeddings and other small/control tensors still follow the existing mixed-precision export rules.

## Validation run here

Executed in this repository:

```bash
python -m compileall candidates/202604042114_lsq-lite-qat/train_gpt.py
```

Outcome: success.

A minimal CPU smoke run was **not feasible** in this environment because this candidate intentionally follows the challenge CUDA path and expects pre-tokenized FineWeb shards plus the repository's GPU evaluation stack.
