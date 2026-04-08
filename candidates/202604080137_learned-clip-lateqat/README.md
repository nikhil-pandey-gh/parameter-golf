# Learned-Clip Late QAT

## Hypothesis

The repo's strongest no-TTT quantization stack already combines EMA, GPTQ-lite clip search, XSA, Partial RoPE, and LN scale, but its late QAT path is still a fixed row-max fake quantizer toggled after `torch.compile`. Replacing that with LSQ/OmniQuant-style **learned per-row clipping** and recompiling once QAT turns on should reduce the train-to-int6 mismatch on the actual deployment weights.

## Why this is promising here

- The records show that the best improvements after the naive baseline came from deeper 11-layer stacks, XSA, EMA, Partial RoPE, and better low-bit handling rather than from simple longer-context runs or basic LR tuning.
- Older long-context-only, seq-length-only, mixed-precision-only, and LoRA-TTT runs improved the baseline, but they were overtaken once the repo leaned harder into quantization-aware training and evaluation-aware tricks.
- I chose `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` as the code base because every large weight still flows through `CastedLinear`, which means a learned-clip QAT change reaches the full model with minimal surgery. That is cleaner than the parameter-banked 2026-03-23 stack, where most of the weight mass bypasses the old QAT hook.

## Prior repo influences

- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`: established EMA + XSA on the 11-layer stack as a winning base.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: showed Partial RoPE + LN scaling were real gains and also documented how late-QAT plumbing can silently fail under compile.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: the direct implementation base here; it already centered the remaining headroom around quantization quality.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: current frontier evidence that the architecture/eval stack is already strong, so a new candidate should target the compression-aware training path rather than repeat older context-only ideas.
- There were **no prior `candidates/` folders** in this repository when this candidate was created.

## External research that informed this candidate

- **Learned Step Size Quantization** (Esser et al., 2019, arXiv:1902.08153) argues that learning quantizer configuration during training can recover low-bit accuracy with only small code changes.
- **OmniQuant** (Shao et al., 2023/2024, arXiv:2308.13137) shows that learnable weight clipping is especially useful in aggressive low-bit regimes, which maps well to this repo's int6 artifact target.
- **QuaRot** (Ashkboos et al., 2024, arXiv:2404.00456) and **SpinQuant** (Liu et al., 2024/2025, arXiv:2405.16406) both reinforce the same theme: outliers are a central low-bit bottleneck. I am taking the smallest adaptation that fits this codebase and time budget: learn clipping directly instead of adding rotation machinery.

## What changed vs. the chosen base implementation

1. Every **int6-exported** `CastedLinear` now owns a trainable `qat_clip_logits` vector, one scalar per output row.
2. The fake-quant branch uses a learned clip factor in `[QAT_CLIP_MIN_FACTOR, 1.0]` instead of always using raw row maxima.
3. The export-time GPTQ-lite search now tests the learned clip candidate in addition to the old percentile grid, so training and export share the same clipping hypothesis.
4. The learned clip tensors are used to shape export-time quantization but are dropped from the final serialized artifact, since they are not needed at inference.
5. When late QAT becomes active, the script recompiles the model once so `torch.compile` does not keep the false-only trace of the old branch, and it re-wraps the compiled module in DDP for multi-GPU runs.
6. The default data/tokenizer paths resolve from the repository root, so the script can be launched directly from this candidate directory.

## How to run

From the repository root:

```bash
cd candidates/202604080137_learned-clip-lateqat
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
QAT_CLIP_INIT=4.0 QAT_CLIP_MIN_FACTOR=0.5 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script defaults to the repo-root `data/datasets/fineweb10B_sp1024` and `data/tokenizers/fineweb_1024_bpe.model` paths when run from this directory. Override `DATA_PATH` and `TOKENIZER_PATH` only if your local layout differs.

## Validation

- `python -m compileall candidates/202604080137_learned-clip-lateqat/train_gpt.py` — passes.
- A minimal CPU runtime smoke test was **not** run here. This workspace did not contain the cached FineWeb shard/tokenizer artifacts needed to get through startup, and this candidate intentionally keeps the existing CUDA/FlashAttention 3 execution path rather than adding a second CPU fallback stack.

## Main risks / tradeoffs

- If late QAT turns on too early, the learned clipping factors could over-regularize the model and hurt the full-precision trajectory before the quantized win materializes.
- Recompiling at QAT activation costs wallclock near the end of training; the threshold may need retuning if that overhead meaningfully eats into the warmdown window.
- Even with the clip tensors dropped from export, the added code/recompile path is still a better fit for the 2026-03-22-style base than for the already-near-limit 2026-03-23 TTT stack.
