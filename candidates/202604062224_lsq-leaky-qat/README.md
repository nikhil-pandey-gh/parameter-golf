# LSQ warmdown QAT + LeakyReLU^2 on the 11L EMA/GPTQ-lite stack

## Hypothesis

The March 22 11-layer stack is already strong on training-side quality, but it still leaves measurable loss on the table when exporting to int6. A compile-safe, LSQ-style late QAT path should make the large linear weights land on more export-friendly step sizes during warmdown, while the LeakyReLU(0.5)^2 MLP from the current top record should improve feature learning without changing the parameter or FLOP budget of the 3x ReLU^2 MLP.

## Why this is promising here

- The best clean training-first base in this repo is the March 22 `11L_EMA_GPTQ-lite_warmdown3500_QAT015` script, which already stacks XSA, partial RoPE, LN scale, EMA, VE, and GPTQ-lite export without the extra evaluation-time complexity of TTT or parameter banking.
- The March 21 README explicitly notes that its late-QAT path was accidentally dead code under `torch.compile`, so there is still unexploited room in this exact part of the design space.
- The March 23 record shows that LeakyReLU(0.5)^2 is a real win on top of the same 11-layer family.
- External research points in the same direction: LSQ-style learned step sizes are a small-code-change way to improve low-bit robustness, PACT/QDrop motivate doing this late in training, and the recent QAT scaling-law work argues that quantization error matters more as token budgets grow.

## Prior records and candidates that influenced it

- **Base fork:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Activation change:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Late-QAT cautionary note:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- **No prior `candidates/` directory existed when this candidate was created.**

## External research that informed it

- **LSQ:** Esser et al., *Learned Step Size Quantization* — <https://arxiv.org/abs/1902.08153>
- **PACT:** Choi et al., *PACT: Parameterized Clipping Activation for Quantized Neural Networks* — <https://arxiv.org/abs/1805.06085>
- **QDrop:** Yao et al., *QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization* — <https://arxiv.org/abs/2203.05740>
- **QAT scaling law:** *Scaling Law for Quantization-Aware Training* — <https://arxiv.org/abs/2505.14302>
- **GLU-family MLP motivation:** Shazeer, *GLU Variants Improve Transformer* — <https://arxiv.org/abs/2002.05202>

## What changed versus the chosen base implementation

1. **LeakyReLU(0.5)^2 MLP**
   - Replaces the base script's ReLU^2 MLP activation with the March 23 record's LeakyReLU(0.5)^2 variant.
   - Keeps the same 3x hidden size, so this is a low-risk drop-in quality change.

2. **LSQ-style late QAT for `CastedLinear`**
   - Every `CastedLinear` now carries a learned per-row log step size (`qat_log_scale`).
   - Late QAT initializes these step sizes from a percentile clip (`QAT_INIT_PERCENTILE`, default `0.9995`) and then fake-quantizes weights with an LSQ-style step-size gradient during warmdown.
   - The QAT gate is now a runtime module state, and the script recompiles once when late QAT turns on so the fast pre-QAT graph does not pay fake-quant overhead from step 0.

3. **Export uses learned scales as an extra candidate**
   - The GPTQ-lite int6 exporter still tries its percentile clips, but it now also evaluates the learned LSQ step sizes as another candidate during final row-wise scale selection.

4. **FlashAttention fallback for local smoke checks**
   - If `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA so the module can still be imported and shape-checked locally.

## How to run

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides when reproducing the intended recipe:

```bash
SEED=1337 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
SWA_ENABLED=1 SWA_EVERY=50 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_WD=0.04 ADAM_WD=0.04 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
LATE_QAT_THRESHOLD=0.15 QAT_INIT_PERCENTILE=0.9995 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

The script keeps the base March 22 evaluation path:

- EMA weights are applied before export.
- The model is exported to mixed int6/int8 form and decompressed for roundtrip validation.
- Sliding-window eval uses `EVAL_STRIDE=64`.

The key output lines to compare against prior records are:

- `DIAGNOSTIC post_ema ...`
- `final_int6_roundtrip_exact ...`
- `final_int6_sliding_window_exact ...`

## Validation run locally for this candidate

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604062224_lsq-leaky-qat/train_gpt.py` | Passed |
| `python -m compileall train_gpt.py train_gpt_mlx.py data` | Passed |
| CPU import/forward smoke via `importlib` | Could not run in this runner because `torch` is not installed here even though it is listed in `requirements.txt` |

## Main expected risks and tradeoffs

- The LSQ path adds extra scalar state and some per-step fake-quant overhead.
- This candidate only implements the weight-side part of the LSQ/PACT/QDrop family; activation clipping and quant-drop are still follow-up experiments.
- It has not yet been timed on the target 8xH100 setup, so the real question is whether the extra warmdown QAT work pays for itself in bpb without costing too many steps.
