# Active Late-QAT + LeakyReLU²

## Hypothesis

The strongest training-only stack in this repo is already clustered around the same 11-layer, compression-aware backbone. The next high-probability win is to stack two **cheap, orthogonal** improvements on top of that backbone:

1. **Make late QAT actually turn on under `torch.compile`** instead of staying dead-code-eliminated.
2. **Swap ReLU² for LeakyReLU(0.5)²** in the MLP, which already showed a measurable gain on a nearby top-stack run.

The expectation is better post-quantization BPB without paying the large throughput penalties that hurt heavier ideas like SwiGLU or recurrent depth reuse.

## Why this is promising for this repository

Repository review showed a stable pattern:

- the leaderboard moved mostly through **artifact-aware quantization**,
- then by **stacking nearly-free architectural tweaks** on the same 11L backbone,
- while **throughput-heavy ideas** were usually net negative inside the 10-minute wallclock.

That makes this candidate a better fit than broader architectural changes:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` was the best training-only record here and already had the right backbone for a small follow-up;
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly notes that its late-QAT path never activated because `torch.compile` constant-folded the class flag;
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` reports a **-0.0021 BPB** gain from LeakyReLU(0.5)² on a closely related stack;
- the non-record `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` and the earlier fp16-embed run both suggest **SwiGLU and recurrence are too expensive per step** in this challenge regime.

There were **no prior `candidates/` directories** in this checkout, so this idea is new relative to both records and prior candidates reviewed here.

## Prior experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Activation evidence:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Dead-code late-QAT diagnosis:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **Rejected directions:** recurrence and SwiGLU notes in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` and `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`

## External research that informed it

- **Primer: Searching for Efficient Transformers for Language Modeling** ([arXiv:2109.08668](https://arxiv.org/abs/2109.08668)) motivated staying in the cheap **squared-ReLU family** instead of adding a slower MLP block.
- **Learned Step Size Quantization** ([arXiv:1902.08153](https://arxiv.org/abs/1902.08153)) reinforced the idea that **real low-bit-aware training** can materially shrink the quantization gap, so it is worth making the repo's late-QAT path actually execute.
- **GLU Variants Improve Transformer** ([arXiv:2002.05202](https://arxiv.org/abs/2002.05202)) was part of the broader search space, but repo evidence pushed this candidate away from GLU/SwiGLU because those gains were offset by step-time regressions here.
- I also reviewed parameter-sharing / recurrent-depth ideas (for example **Universal Transformer**, [arXiv:1807.03819](https://arxiv.org/abs/1807.03819), and **ALBERT**, [arXiv:1909.11942](https://arxiv.org/abs/1909.11942)) but did **not** choose them because this repo already has direct negative evidence for reused-depth under a hard 10-minute wallclock.

## What changed vs. the base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes only three substantive changes:

1. **Compile-safe late QAT**
   - removes the class-level `CastedLinear._qat_enabled` switch that `torch.compile` could constant-fold;
   - threads an explicit `qat_active: bool` through the compiled forward path;
   - flips that boolean once `lr_mul(...) < LATE_QAT_THRESHOLD`, which should cause a retrace into the QAT-enabled graph instead of silently doing nothing.
2. **LeakyReLU(0.5)² MLP**
   - replaces `relu(x)^2` with `leaky_relu(x, 0.5)^2`;
   - adds `LEAKY_RELU_SLOPE` as an env override, defaulting to `0.5`.
3. **Candidate-directory runnable defaults**
   - default `DATA_PATH` and `TOKENIZER_PATH` now resolve from the repo root via `Path(__file__).resolve().parents[2]`, so `train_gpt.py` can be launched from inside this candidate folder.

Everything else stays aligned with the proven 11L stack: XSA on the last 4 layers, partial RoPE, LN scale, VE128, EMA, tight SWA, GPTQ-lite int6 export, BigramHash, and SmearGate.

## How to run or evaluate

From this candidate directory:

```bash
cd candidates/202604071314_active-late-qat-leakyrelu
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs for ablation:

```bash
LEAKY_RELU_SLOPE=0.5
LATE_QAT_THRESHOLD=0.15
```

The script still emits the same important evaluation lines as the base implementation, including:

- `final_int6_roundtrip_exact`
- `final_int6_sliding_window_exact`
- `final_int6_sliding_window_s64_exact`

## Main risks / tradeoffs

- **Retrace cost when late QAT activates:** switching the compiled graph from `qat_active=False` to `True` should be much better than dead QAT, but it may still create a one-time compilation stall near warmdown.
- **Fake-quant mismatch:** the late-QAT path still uses a simple per-row int6 proxy during training, while export uses GPTQ-lite percentile search. The match is better than dead QAT, but still imperfect.
- **Activation drift:** LeakyReLU² helped on a nearby stack, but the exact gain may differ on the VE128 + GPTQ-lite configuration.
- **No local full runtime validation here:** this environment lacked the runtime stack needed for a faithful training launch.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604071314_active-late-qat-leakyrelu/train_gpt.py
```

Outcome:

- **Passed**.

Attempted lightweight CPU smoke:

- I tried importing and exercising the script with a stubbed FlashAttention path, but the local runner does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`), so a meaningful runtime smoke test was **not feasible in this environment**.
- A real runtime check still needs the repository's normal PyTorch + CUDA + FlashAttention setup.
