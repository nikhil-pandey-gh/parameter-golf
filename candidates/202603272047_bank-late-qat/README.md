# Bank-Aware Late QAT

## Hypothesis

The current strongest repo stack already wins by making post-training quantization unusually cheap, but its **banked attention/MLP weights** still train entirely in floating point and only see int6 constraints at export time.

This candidate tests whether a **late, compile-safe int6 STE path applied directly to the banked weights** can further reduce the quantization gap without paying the full training-time overhead of always-on QAT.

## Why this is promising for this repository

Three patterns stand out from the repo history:

1. Quantization quality, not just pre-quant loss, has repeatedly driven leaderboard gains.
2. The 2026-03-21 Partial-RoPE record explicitly documented that its intended late-QAT path never actually affected training because `torch.compile` constant-folded the toggle.
3. The 2026-03-23 LeakyReLU² + legal TTT record moved the main transformer weights into large **parameter banks**, which makes the old `CastedLinear` late-QAT path even less relevant to the bytes that dominate the artifact.

So the natural next step is not “more post-training clip search,” but **teaching the banked weights to survive int6 before export**.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Key supporting records:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`

There were **no prior `candidates/` folders** in the repository when this candidate was created.

## External research that informed it

- **BitNet b1.58** (`arXiv:2402.17764`): motivates low-bit-aware training rather than treating quantization as a purely post-training problem.
- **GPTQ** (`arXiv:2210.17323`): strong post-training quantization remains valuable, but it works best when the trained weight distribution is already quantization-friendly.
- **AWQ** (`arXiv:2306.00978`): quantization quality depends heavily on how weight/activation statistics line up, reinforcing the idea that training should help shape exportable weights.
- **Precision-aware scaling laws** (`arXiv:2411.04330`): argues that low-precision training and post-train quantization are coupled, not separable, and that precision-aware training can be compute-optimal.

## What changed versus the chosen base implementation

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps the same overall stack — LeakyReLU(0.5)^2, parameter banking, GPTQ-lite export, EMA, XSA, partial RoPE, VE, BigramHash, and optional legal TTT — but changes the late-QAT story in one focused way:

- adds `BANK_QAT_ENABLED` (default `1`)
- adds `fake_quantize_int6_ste()` for per-row symmetric int6 STE fake-quantization
- threads a `qat_active` flag through `GPT -> Block -> CausalSelfAttention / MLP`
- when the LR scale falls below `LATE_QAT_THRESHOLD`, the script now fake-quantizes the **banked** attention and MLP matrices during training
- logs `bank_qat:enabled ...` when this transition happens

Importantly, this is different from the older dormant late-QAT path because it targets the **actual banked matrices** that dominate both model capacity and compressed artifact size.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603272047_bank-late-qat

RUN_ID=bank_late_qat \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
BANK_QAT_ENABLED=1 LATE_QAT_THRESHOLD=0.15 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 VOCAB_SIZE=1024 \
python train_gpt.py
```

By default the script resolves `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so it can be launched directly from this candidate folder. This candidate inherits the base script's fixed EMA export path, so no extra EMA/SWA environment knobs are needed here. If you want to isolate the training-side effect before paying for TTT evaluation, rerun with `TTT_ENABLED=0` first.

## Validation

Validated in this container:

```bash
python -m compileall candidates/202603272047_bank-late-qat/train_gpt.py
python -m py_compile candidates/202603272047_bank-late-qat/train_gpt.py
```

Outcome:

- both syntax checks passed
- a runtime smoke test was **not feasible in this container** because the required runtime modules are missing here (`torch`, `sentencepiece`, and `flash_attn_interface` were all unavailable), so no honest import-time or CPU start-up check could be executed locally

## Main expected risks / tradeoffs

- **Training-time overhead**: fake-quantizing four bank families during the late phase may cost steps and eat some of the win.
- **Mismatch with GPTQ-lite export**: training uses a simple per-row symmetric int6 STE, while export still uses percentile-searched GPTQ-lite clipping; the mismatch could limit gains.
- **Late-only schedule sensitivity**: if `LATE_QAT_THRESHOLD` is too early, speed suffers; if it is too late, the bank distributions may not move enough.
- **Peripheral weights remain separate**: this candidate targets the big banked matrices first; smaller non-banked projections still rely mostly on the pre-existing export path.

## Suggested next experiments

1. Compare `TTT_ENABLED=0` runs first to measure whether the quantization gap itself shrinks.
2. Sweep `LATE_QAT_THRESHOLD` across `0.10`, `0.15`, and `0.20`.
3. Try bank-QAT on only the attention banks or only the MLP banks if the full late-phase overhead is too large.
4. If this helps, combine it with a future BOS-aware/doc-aware training or evaluation pass.
