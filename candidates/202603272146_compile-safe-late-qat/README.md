# Compile-Safe Late QAT on the 11L EMA + GPTQ-lite Base

## Hypothesis

The strongest training-only stack in this repository already has most of the right ingredients: 11 layers, 3x MLP, XSA on deep layers, Partial RoPE, LN scaling, EMA, and GPTQ-lite export. The most attractive next win is not another large architectural jump, but making the existing late int6 QAT path actually participate in optimization.

The hypothesis is that a short, explicit eager-mode QAT tail can reduce the final int6 export gap enough to improve post-quant `val_bpb`, while keeping most of the run on the fast compiled path.

## Why this looks promising for this repository

The repository review pointed to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` as the best pure training stack. It also showed that `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` explicitly identified a failure mode: the late-QAT branch was dead-code-eliminated under `torch.compile`, so the intended QAT phase never actually ran.

That makes compile-safe late QAT unusually attractive here: it is tightly matched to the repo's current bottleneck, it is compatible with the existing code structure, and it directly targets the compression-aware objective that has driven most of the leaderboard gains.

## Prior records or candidates that influenced this

There were no prior experiments under `candidates/` when this candidate was created.

The main repository influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen base implementation
  - strongest training-only stack found during review
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - documented that late QAT was effectively a no-op under `torch.compile`
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - reinforced that compression-aware training continues to matter even after architecture improvements
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative evidence that true recurrent-depth loops can lose badly under a fixed wall-clock budget, which pushed this candidate toward a staged training fix instead of extra depth reuse

## External research that informed it

This candidate is mostly motivated by recent QAT literature rather than a new attention mechanism:

- **EfficientQAT: Efficient Quantization-Aware Training for Large Language Models** (`arXiv:2407.11062`)
  - argues for staged QAT instead of paying full-run low-bit training cost
- **Scaling Law for Quantization-Aware Training** (`arXiv:2505.14302`)
  - reports that weight quantization error becomes increasingly important as token count grows, which lines up with this repo's long-token, low-bit export setting
- **Quantization Variation: A New Perspective on Training Transformers with Low-Bit Precision** (`arXiv:2307.00331`)
  - highlights that transformer QAT is brittle because modules differ in sensitivity and exhibit strong outlier behavior, supporting a short, targeted late phase instead of always-on QAT

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate keeps the same 11-layer architecture and export path, but changes the training/runtime behavior in four precise ways:

1. **Added an explicit eager QAT tail**
   - New flag: `LATE_QAT_EAGER=1` by default
   - Training starts on the original compiled path
   - Once LR scale drops below `LATE_QAT_THRESHOLD`, the script enables the existing STE int6 fake-quant path and switches the training forward pass to eager mode so that QAT cannot be compile-folded away

2. **Kept the original late-QAT STE logic, but made it reachable**
   - The candidate does not invent a new quantizer
   - It makes the intended late fake-quant branch actually execute during the tail phase

3. **Added a local fallback for attention**
   - If `flash_attn_interface` is not importable, the script falls back to PyTorch `scaled_dot_product_attention`
   - This is mainly for portability and smoke testing; the intended target path remains FlashAttention 3 on CUDA

4. **Fixed default data/tokenizer paths for candidate-local execution**
   - Defaults now discover the repository root by walking upward for `data/` and `README.md`, so the script keeps working when launched from either `candidates/...` or a future `records/...` folder without overriding `DATA_PATH` or `TOKENIZER_PATH`

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603272146_compile-safe-late-qat
SEED=1337 \
LATE_QAT_EAGER=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script now discovers the repository root automatically, so the command above can be launched directly from this folder and still resolve the shared tokenizer and dataset paths.

Useful knobs to compare against the base:

```bash
LATE_QAT_EAGER=0        # keep the old fully-compiled behavior
LATE_QAT_THRESHOLD=0.15 # same threshold as the chosen base record
QAT_ENABLED=1           # start with fake quant already enabled, if you want a heavier ablation
```

## Main expected risks or tradeoffs

- **Tail-phase slowdown**: switching from compiled to eager mode near the end of training may reduce final step count.
- **Threshold sensitivity**: the best `LATE_QAT_THRESHOLD` may move once QAT is real instead of a no-op.
- **Quantization benefit may be uneven**: if only a subset of matrices dominate the export gap, blanket late QAT may still be too coarse.
- **Fallback attention is not the benchmark path**: the SDPA fallback is for portability, not the target H100 submission setup.

## Validation

I ran the lightweight checks that fit this environment:

```bash
python -m compileall candidates/202603272146_compile-safe-late-qat/train_gpt.py
```

Outcome: **passed**.

I also attempted a CPU-only synthetic import/forward/backward smoke test of `GPT` from this candidate. That was blocked by the current runner environment, where `python` does not have `torch` installed even though `requirements.txt` lists it.

Attempted command:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch
path = Path('/home/runner/work/parameter-golf/parameter-golf/candidates/202603272146_compile-safe-late-qat/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
model = mod.GPT(
    vocab_size=128,
    num_layers=4,
    model_dim=64,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2.0,
    tie_embeddings=True,
    tied_embed_init_std=0.005,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
    mtp_num_heads=0,
    mtp_loss_weight=0.0,
    bigram_vocab_size=0,
    bigram_dim=16,
    xsa_last_n=0,
    rope_dims=16,
    ln_scale=True,
    dtg=False,
    ve_enabled=False,
    ve_dim=32,
    ve_layers='',
)
model.train()
mod.CastedLinear._qat_enabled = True
x = torch.randint(0, 128, (2, 16), dtype=torch.int64)
y = torch.randint(0, 128, (2, 16), dtype=torch.int64)
loss = model(x, y)
loss.backward()
print(loss.item())
PY
```

Outcome: **blocked by environment dependency issue** (`ModuleNotFoundError: No module named 'torch'`).
