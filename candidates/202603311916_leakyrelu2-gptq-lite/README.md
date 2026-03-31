# LeakyReLU(0.5)^2 on the 11L EMA + GPTQ-lite base

## Hypothesis

The cleanest next improvement is to isolate the `LeakyReLU(0.5)^2` MLP activation on top of the strongest non-TTT stack already present in this repository: the 11-layer EMA + GPTQ-lite + warmdown3500 + late-QAT base from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`.

The hypothesis is that preserving a controlled negative-path gradient in the feed-forward blocks improves optimization of the same 11L / 3x-MLP / XSA / Partial-RoPE / EMA stack without adding parameters, code complexity, or evaluation-time cost. The goal is to capture part of the gain seen in the later `LeakyReLU^2` + TTT record while keeping the candidate focused on the clean training stack rather than confounding it with TTT or Parallel Muon.

## Why this looks promising in this repository

Repository history points to a narrow but high-confidence opportunity:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` is the strongest clean no-TTT training stack in the repo (`val_bpb: 1.1233`).
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` reports that swapping `relu^2` to `LeakyReLU(0.5)^2` was worth roughly `-0.0021` to `-0.003` BPB inside the later TTT-based stack.
- The non-record `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` found that heavier activation changes can improve quality but often cost too much throughput. `LeakyReLU^2` is much safer because it is almost a drop-in swap.

So the most efficient next candidate is to transplant that activation change into the best existing no-TTT base instead of betting on costlier architectural experiments like recurrence or larger sequence lengths.

## Prior records and candidates that influenced this candidate

There were no prior entries under `candidates/` when this candidate was created.

The main repository influences were:

- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/README.md` for the repo-wide lesson that evaluation and context handling matter enough that low-risk quality gains are worth isolating cleanly.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md` for the 11-layer EMA/XSA direction.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` for Partial RoPE and LN scaling.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` as the direct implementation base.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` for the activation ablation motivating the change.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` as a caution that more expensive activation ideas can lose on throughput under the 10-minute cap.

## External research that informed the choice

I reviewed several compact-model directions before picking this candidate:

- **Primer** (`arXiv:2109.08668`) found that two simple transformer changes consistently improved autoregressive language modeling efficiency, with one of the main wins coming from **squared ReLU activations**. That supports staying close to the repo's already-successful `relu^2` family instead of switching to a much heavier MLP design.
- **GLU Variants Improve Transformer** (`arXiv:2002.05202`) showed that activation design inside transformer feed-forward blocks materially affects quality, reinforcing the idea that a targeted activation swap can be worthwhile even without larger architectural changes.
- **Learned Step Size Quantization** (`arXiv:1902.08153`) and **SmoothQuant** (`arXiv:2211.10438`) make quantization-aware or quantization-friendly training/export look attractive in general, but this repository already has strong GPTQ-lite/QAT machinery. That made an activation-only change more attractive for this iteration because it isolates a different lever while preserving the best existing quantization stack.
- I also reviewed classic recurrent/parameter-sharing directions such as **Universal Transformer** (`arXiv:1807.03819`) and **ALBERT** (`arXiv:1909.11942`), but the repository's own prior notes and non-record experiments suggest depth recurrence is risky under the 10-minute wallclock because reduced step count can outweigh the parameter-efficiency benefits.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **MLP activation swap**

   From:

   ```python
   x = torch.relu(self.fc(x))
   return self.proj(x.square())
   ```

   To:

   ```python
   x = F.leaky_relu(self.fc(x), negative_slope=0.5)
   return self.proj(x.square())
   ```

2. **Run-from-candidate-directory path fix**

   The copied record script assumed repo-root-relative defaults for `DATA_PATH` and `TOKENIZER_PATH`. This candidate rewrites those defaults to be derived from `Path(__file__).resolve().parents[2]`, so `python train_gpt.py` works correctly when launched from inside this candidate directory without extra path overrides.

Everything else is intentionally held constant so the activation change can be evaluated cleanly:

- 11 layers, 512 width, 8 heads / 4 KV heads
- 3x MLP
- XSA on the last 4 layers
- Partial RoPE (16/64) + LN scaling
- BigramHash + SmearGate + VE128
- always-on EMA (hard-coded to `0.997` in the inherited base) + tight SWA
- GPTQ-lite export + zstd-22
- late-QAT threshold `0.15`
- seq_len/eval_seq_len `2048`, eval stride `64`

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603311916_leakyrelu2-gptq-lite

SEED=1337 \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
ROPE_DIMS=16 \
LN_SCALE=1 \
QAT_ENABLED=0 \
LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The default `DATA_PATH` and `TOKENIZER_PATH` now resolve back to the repository `data/` directory automatically, so they only need overrides if your dataset or tokenizer live elsewhere.

## Main expected risks or tradeoffs

- The gain may already be partially entangled with the later TTT stack that reported it, so the isolated no-TTT win could be smaller than the `-0.002` to `-0.003` repo hint.
- `LeakyReLU(0.5)^2` is intentionally low-risk, but the slope is still a new hyperparameter choice; `0.5` is inherited from the prior record rather than retuned for this base.
- This candidate does not tackle the remaining quantization gap directly. If the activation swap lands flat, the next iteration should likely target quantization-aware export rather than larger architectural changes.

## Validation

Commands run in this repository:

```bash
python -m compileall candidates/202603311916_leakyrelu2-gptq-lite/train_gpt.py
```

Outcome:

- `python -m compileall candidates/202603311916_leakyrelu2-gptq-lite/train_gpt.py` passed.
- The script-relative `REPO_ROOT = Path(__file__).resolve().parents[2]` check resolves to the repository root as intended, so the default dataset/tokenizer paths are now candidate-directory-safe.
- The current container does not include downloaded FineWeb shard/tokenizer caches under `data/datasets/` or `data/tokenizers/`, so a real runtime launch could not be exercised here.

CPU smoke-test note:

- A true runtime smoke test is not feasible in this environment because this candidate inherits the record script's CUDA + FlashAttention 3 execution path and does not include a CPU fallback. Running it meaningfully requires the challenge GPU environment.
