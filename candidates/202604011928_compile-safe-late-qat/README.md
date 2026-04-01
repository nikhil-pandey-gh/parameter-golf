# 202604011928 Compile-Safe Late QAT

## Hypothesis

The strongest non-TTT 11-layer stack in this repo already treats quantization as the main bottleneck, but its late-QAT path was wired through a Python class attribute that a later record explicitly reported as dead-code-eliminated by `torch.compile`. The hypothesis here is that a **real, compile-safe, late-start int6 QAT path** can narrow the final roundtrip gap without paying the full throughput penalty of always-on QAT.

## Why this looks promising for this repo

- Repo history repeatedly shows that **post-training quantization quality dominates** many architecture tweaks:
  - `2026-03-19_WarmdownQuantization` argues the quantization penalty was larger than most training wins.
  - `2026-03-19_MLP3x_QAT_Int6_SlidingWindow` shows integrated int6 QAT can matter, but with meaningful overhead.
  - `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improves the best 11-layer int6 stack further with GPTQ-lite clip search and EMA.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly notes that its late-QAT flag **never actually activated** under `torch.compile`.
- That makes “make late QAT real, but keep it cheap” a strong next step that is both differentiated and tightly aligned with the repo’s current winning trend.

## Prior records that most influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - chosen as the base implementation
   - provides the 11-layer EMA + GPTQ-lite + warmdown-3500 stack
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - documents the compile-folded late-QAT bug directly
3. `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`
   - shows that real int6 QAT can help, but also why full-run QAT is expensive
4. `records/track_10min_16mb/2026-03-19_WarmdownQuantization/`
   - reinforces that quantization-aware training dynamics are first-order in this challenge

There were **no prior `candidates/` directories** in the repo when this candidate was created.

## External research that informed it

- **LSQ** (Esser et al., 2019): learning around low-bit quantization noise is effective when the quantizer is part of the training path.  
  https://arxiv.org/abs/1902.08153
- **GPTQ** (Frantar et al., 2022): strong low-bit weight quantization benefits from matching the eventual export behavior closely.  
  https://arxiv.org/abs/2210.17323
- **SmoothQuant** (Xiao et al., 2022): quantization error and outliers can dominate deployment quality even when the base model is otherwise strong.  
  https://arxiv.org/abs/2211.10438
- **SpinQuant** (Liu et al., 2024/2025): more recent work continues to show that quantization-aware conditioning is still a major lever for LLM quality.  
  https://arxiv.org/abs/2405.16406

This candidate intentionally stays much simpler than rotation-based methods or full LSQ: it keeps the repo’s current export path and only makes the late-QAT proxy both real and cheap enough for this codebase.

## What changed vs. the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Compile-safe QAT control**
   - removes the class-level `_qat_enabled` branch that could be constant-folded away
   - each `CastedLinear` now holds:
      - `qat_blend`: scalar strength for the fake-quant contribution
      - `qat_weight_cache`: cached dequantized low-bit target
   - the training path stays on the normal fast graph until late QAT starts, then recompiles once onto the QAT-enabled graph

2. **Progressive late-QAT ramp**
   - adds `QAT_RAMP_START`, `QAT_RAMP_END`, and `QAT_REFRESH_EVERY`
   - QAT ramps from `0.0 -> 1.0` as the monotonic minimum late-warmdown LR scale crosses the ramp window instead of turning on abruptly in one step

3. **Cached export-matched int6 targets**
   - refreshes QAT caches only every `QAT_REFRESH_EVERY` steps
   - refresh targets are produced with the same `quantize_int6_per_row()` family already used by export
   - only tensors that match the export path’s int6 criteria (`"mlp"`/`"attn"` **and** `numel() > 65_536`) allocate caches and receive fake quantization; the rest stay on the base path

4. **Candidate-directory runnable defaults**
   - default `DATA_PATH` and `TOKENIZER_PATH` now resolve from the repository root so the script can be run directly from this candidate directory

Not changed:

- architecture shape
- EMA / SWA / GPTQ-lite export logic
- sliding-window evaluation path
- optimizer split and hyperparameters outside the new QAT ramp controls

## How to run

From this candidate directory:

```bash
cd candidates/202604011928_compile-safe-late-qat
RUN_ID=compile_safe_late_qat \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
QAT_ENABLED=0              # compare against the base behavior
QAT_RAMP_START=0.18        # default start of the late-QAT ramp
QAT_RAMP_END=0.05          # default point where QAT reaches full strength
QAT_REFRESH_EVERY=16       # cache refresh cadence
```

## How to evaluate

The script keeps the base record’s export behavior:

- EMA weights are applied before export
- model weights are mixed-quantized with int6 on attention/MLP-style matrices
- roundtrip validation is run on the exported artifact
- sliding-window eval remains available through the base config

## Main risks and tradeoffs

- **Late-stage overhead:** cache refreshes are cheaper than per-forward fake quant, but they still add work near the end of training.
- **Extra training memory:** each `CastedLinear` stores a bf16 cached target during training.
- **Proxy mismatch:** the late-QAT cache uses the repo’s export quantizer family, but it is still a proxy for the final artifact path rather than a learned LSQ-style quantizer.
- **Tuning sensitivity:** the best `QAT_RAMP_START`, `QAT_RAMP_END`, and refresh cadence may differ from the defaults.

## Validation

Commands run for this candidate:

1. Syntax check

```bash
python -m compileall candidates/202604011928_compile-safe-late-qat/train_gpt.py
```

Outcome: **passed**

2. Minimal CPU smoke check with a local `flash_attn_interface` stub

```bash
python - <<'PY'
import importlib.util
import sys
import types
import torch
import torch.nn.functional as F

stub = types.ModuleType("flash_attn_interface")

def flash_attn_func(q, k, v, causal=True):
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    y = F.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        is_causal=causal,
        enable_gqa=(q_t.size(1) != k_t.size(1)),
    )
    return y.permute(0, 2, 1, 3).contiguous()

stub.flash_attn_func = flash_attn_func
sys.modules["flash_attn_interface"] = stub

spec = importlib.util.spec_from_file_location(
    "candidate_train_gpt",
    "candidates/202604011928_compile-safe-late-qat/train_gpt.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model = module.GPT(
    vocab_size=64,
    num_layers=2,
    model_dim=512,
    num_heads=8,
    num_kv_heads=4,
    mlp_mult=3,
    tie_embeddings=True,
    tied_embed_init_std=0.005,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.5,
    mtp_num_heads=0,
    mtp_loss_weight=0.0,
    bigram_vocab_size=0,
    bigram_dim=128,
    xsa_last_n=0,
    rope_dims=16,
    ln_scale=True,
    dtg=False,
    ve_enabled=False,
)
module.restore_low_dim_params_to_fp32(model)
module.reset_qat_state(model)
refreshed = module.refresh_qat_weight_caches(model)
module.set_qat_blend(model, 0.25)
model.train()
torch.manual_seed(0)
inputs = torch.randint(0, 64, (1, 8), dtype=torch.int64)
targets = torch.randint(0, 64, (1, 8), dtype=torch.int64)
loss = model(inputs, targets)
active_after = sum(
    int(getattr(m, "qat_active", False))
    for m in model.modules()
    if isinstance(m, module.CastedLinear)
)
cache_dtypes = sorted(
    {
        str(m.qat_weight_cache.dtype)
        for m in model.modules()
        if isinstance(m, module.CastedLinear) and m.qat_weight_cache.numel()
    }
)
print(f"torch={torch.__version__}")
print(f"refreshed={refreshed}")
print(f"active_after={active_after}")
print(f"cache_dtypes={cache_dtypes}")
print(f"smoke_loss={loss.item():.4f}")
PY
```

Outcome: **passed** (`torch=2.11.0+cu130`, `refreshed=12`, `active_after=12`, `cache_dtypes=['torch.bfloat16']`, finite `smoke_loss`)

No GPU training run was attempted in this environment.
