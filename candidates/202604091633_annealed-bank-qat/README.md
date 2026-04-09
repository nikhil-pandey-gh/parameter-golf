# Annealed Bank-QAT

This candidate keeps the current best repo stack and adds one new training-time ingredient: an annealed STE fake-quant blend over the **banked core weights** (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`) during the late warmdown.

## Hypothesis

The strongest current stack still pays a post-training int6 quantization penalty on the big banked tensors that dominate the compressed artifact. A lightweight late-QAT regularizer targeted at those tensors should make the final trunk more export-friendly without adding persistent artifact bytes.

## Why this is promising here

- The current best record is already banked and export-size constrained, so the biggest low-bit sensitivity sits exactly in the tensors this candidate now regularizes.
- The 2026-03-21 Partial RoPE + LN Scale record explicitly notes that its late-QAT flag was dead-code-eliminated by `torch.compile`, so the repository has already identified the idea as promising but not actually exercised it.
- The 2026-03-22 GPTQ-lite + EMA record still won by shaving quantization error at export time, which suggests compression-aware training is still a live frontier.

## Prior experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest current stack in this repo
  - already uses parameter banking + Parallel Muon, so it is the right place to target bank-weight quantization directly
- **Quantization motivation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - showed small but real wins from better post-training quantization
- **Bug report / dead end to fix:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - documents that the prior late-QAT path never activated under `torch.compile`
- **Prior candidates:** there were no existing `candidates/` runs in this repository when this candidate was created.

## External research that informed it

- **LSQ / Learned Step Size Quantization** — Esser et al., arXiv:1902.08153  
  Training-time low-bit regularization can recover much of the accuracy lost at inference-time quantization; that motivated adding a real fake-quant path instead of relying only on export-time clipping.
- **AWQ / Activation-aware Weight Quantization** — Lin et al., arXiv:2306.00978  
  Quantization error is concentrated in a small set of important weight channels; that motivated targeting the main banked weight path rather than only small auxiliary matrices.

## What changed versus the chosen base

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

1. adds repo-root-resolved dataset/tokenizer defaults so the script can be run directly from the candidate directory,
2. adds `BANK_QAT_ENABLED`, `BANK_QAT_START_SCALE`, and `BANK_QAT_BITS`,
3. applies per-row STE fake quantization to the four bank tensors during training,
4. anneals the fake-quant blend from `0 -> 1` as the warmdown LR scale falls below `BANK_QAT_START_SCALE`,
5. keeps the fake-quant state out of the saved artifact and leaves export-time GPTQ-lite / lzma behavior unchanged.

The design is intentionally minimal: no new files, no new dependencies, and no change to the final serialized parameter set beyond whatever training pressure the late fake-quant path induces.

## How to run

From this candidate directory:

```bash
cd candidates/202604091633_annealed-bank-qat
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
BANK_QAT_ENABLED=1 BANK_QAT_START_SCALE=0.20 BANK_QAT_BITS=6 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script now resolves `data/` relative to the repository root, so `DATA_PATH` and `TOKENIZER_PATH` do not need to be overridden if you run it from this folder in the standard repo layout.
EMA is inherited from the base stack and remains hard-coded at `0.997` unless you change the script.

## Expected risks and tradeoffs

- The training signal uses simple per-row int6 fake quantization, while export still uses GPTQ-lite clip search; the train/export mismatch could limit gains.
- Bank fake-quant is applied every training forward, so there is some extra step-time cost even before the blend ramps up.
- The best stack already includes TTT, EMA, SWA, and banking; interactions between late fake-quant pressure and those moving parts may be non-linear.
- If `BANK_QAT_START_SCALE` is too large, the candidate may trade away too many clean full-precision steps; if it is too small, the quantization gap may barely move.

## Validation

Commands run locally for this candidate:

```bash
python -m compileall candidates/202604091633_annealed-bank-qat/train_gpt.py
```

Outcome:

- `python -m compileall` **succeeded**.
- A CPU-only startup smoke test was **not feasible** in this runner because the local Python environment does not have `torch` installed, and the repository cache required for startup is also absent here (`data/tokenizers/fineweb_1024_bpe.model` missing, `fineweb_train_*.bin` shards: 0, `fineweb_val_*.bin` shards: 0).
