# 11L GPTQ-lite + EMA + Value Residual

## Hypothesis

ResFormer-style value residual connections should help this repository's deep, thin 11-layer stack preserve token-local information deeper into the network with almost no extra parameter or compute cost. In this setting, the value path is a good place to add capacity because the current best training-only stack already uses GQA, partial RoPE, XSA, LN scaling, and shared value embeddings, so improving how value states propagate is a natural next step.

## Why this is promising here

- The current record trend is strongly in favor of **deep/thin 11-layer models** with aggressive compression-aware training rather than wider 9-layer baselines.
- The best pre-TTT training stack already converged on **GQA + partial RoPE + XSA + EMA + GPTQ-lite**, which leaves relatively little easy headroom in optimization alone.
- Value residuals target a gap that the repo has not explored in any documented record: preserving early token identity in later attention layers without materially changing artifact size.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`: chosen base implementation because it is the strongest training-only 11-layer stack without the added complexity of legal TTT and parameter banking.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`: confirms that partial RoPE and LN scaling were meaningful wins on top of the same family.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`: establishes XSA + EMA as the right backbone for the modern 11-layer line.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`: shows the next leaderboard gain came from eval-time adaptation and systems work, which is exactly why this candidate instead looks for a fresh training-side architecture improvement.

## External research

- **Value Residual Learning / ResFormer** ([arXiv:2410.17897](https://arxiv.org/abs/2410.17897)): introduces residual connections on the value states and reports equivalent validation loss with fewer parameters and less training data than a vanilla Transformer, with similar compute and memory.
- **MobileLLM** ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905)): argues that sub-billion models benefit disproportionately from architectural choices such as deep-thin layouts and GQA, which aligns with the direction already winning in this repository.

## What changed versus the base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes:

1. Added a **ResFormer-style value residual** path in attention. The first captured value tensor is mixed into later layers with a learned 2-way coefficient `vr_lambda`.
2. Added `VALUE_RESIDUAL` as an environment flag and enabled it by default in this candidate.
3. Marked `vr_lambda` as a control tensor so the int6 export path preserves it in fp32 like the other low-dimensional scaling parameters.
4. Switched default dataset and tokenizer paths to be **repo-relative**, so the script can be launched from this candidate directory directly.
5. Added a minimal FlashAttention fallback import path so local non-Hopper environments fail later and more clearly instead of at import time.

## How to run

From this candidate directory:

```bash
SEED=1337 \
VALUE_RESIDUAL=1 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected risks and tradeoffs

- Reusing the first layer's values may over-constrain later layers if the learned mixing coefficients do not quickly specialize.
- The extra residual could interact negatively with XSA, which already modifies the attention output geometry in the deepest layers.
- This candidate does not change the export format, so if the gain is mostly pre-quantization and not robust after int6 roundtrip, the final leaderboard metric may not move much.

## Validation

- `python3 -m compileall train_gpt.py`
- Minimal CPU smoke test was attempted only if existing local dependencies allowed it; see the command outcomes below.

### Outcomes

- `python3 -m compileall train_gpt.py` **passed**.
- A CPU smoke launch was **not feasible in this workflow environment** because the available Python runtime did not have the existing project dependencies installed (`torch`, `numpy`, and `sentencepiece` were missing, and FlashAttention was also unavailable). This candidate therefore only received syntax-level validation here.
