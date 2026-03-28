# Salient-Row GPTQ-lite Rescue

## Hypothesis

The strongest non-TTT stack in this repo already gets most of its gain from better quantization and post-training averaging, but it still spends nearly all weights at a uniform precision per matrix. A small, automatic fp16 "escape hatch" for the worst per-row quantization casualties should reduce the remaining roundtrip gap more efficiently than keeping an entire tensor in higher precision.

## Why this is promising for this repository

Repository evidence points to quantization as the main remaining bottleneck:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` showed that keeping the full tied embedding in fp16 nearly erased the embedding quantization penalty, but cost about 500 KB.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md` further showed that even keeping just the last-layer key projection in fp16 could help.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that better post-training clipping alone was still worth another `-0.0006` BPB.

That combination suggests the next reasonable step is not another whole-model rewrite, but a finer-grained mixed-precision escape hatch that spends bytes only where rowwise quantization is worst.

## Prior records and experiments that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Key influences:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` for the `LeakyReLU(0.5)^2` MLP activation.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` for the observation that full fp16 embeddings help a lot but are too expensive to apply wholesale at this 11-layer size.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md` for selective higher-precision preservation of especially sensitive weights.

## External research that informed it

- **AWQ** (`arXiv:2306.00978`): salient weights are not uniformly distributed, and protecting a small important subset can dramatically reduce quantization error.
- **SpQR** (`arXiv:2306.03078`): isolating outlier weights in higher precision enables near-lossless low-bit compression.
- **SqueezeLLM** (`arXiv:2306.07629`): dense low-bit quantization plus a sparse higher-precision outlier path is stronger than uniform low-bit assignment under the same memory budget.
- **QuaRot** (`arXiv:2404.00456`) and **SpinQuant** (`arXiv:2405.16406`): recent work continues to identify outliers as the central obstacle for aggressive quantization, even when solved with different machinery.

This candidate is intentionally a minimal repo-friendly adaptation of those ideas rather than a full AWQ/SpQR implementation.

## What changed versus the base implementation

Starting from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate changes only the copied script in this folder:

1. **LeakyReLU(0.5)^2 MLP**
   - replaces `relu^2` with the stronger activation used by the current top record.

2. **Salient-row rescue during export**
   - keep the existing GPTQ-lite style per-row clip search for int6 weights and per-row int8 for the rest;
   - compute row reconstruction error after quantization;
   - spend a fixed byte budget on the highest-error rows across the whole model;
   - store only those rows in fp16 and overwrite them after dequantization.

3. **Adaptive artifact-budget fallback**
   - if the rescued model would exceed `ARTIFACT_BUDGET_BYTES` (default `16_000_000`), the script shrinks the rescue budget and re-packs until it fits or rescue is exhausted.

## How it differs from existing records

Existing records already explored:

- whole-tensor fp16 embedding passthrough,
- whole-category int6/int8 splits,
- one-off fp16 preservation of a particularly sensitive projection,
- better row clipping (GPTQ-lite).

This candidate adds a **global, automatic, row-selective higher-precision rescue path** across all large 2D tensors. It is meant to capture the same kind of wins as those earlier selective precision tricks, but with much finer granularity and tighter control over the byte budget.

## How to run

From this candidate directory:

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
ROW_RESCUE_ENABLED=1
ROW_RESCUE_MAX_EXTRA_BYTES=262144
ARTIFACT_BUDGET_BYTES=16000000
```

The rest of the training defaults follow the copied 11-layer EMA + GPTQ-lite configuration.

## Validation

Commands run for this candidate in this workflow:

```bash
python -m compileall \
  train_gpt.py \
  train_gpt_mlx.py \
  data \
  candidates/202603281811_salient-row-gptq/train_gpt.py
python - <<'PY'
import importlib.util
import sys
import types
from pathlib import Path

import torch

stub = types.ModuleType("flash_attn_interface")
stub.flash_attn_func = lambda q, k, v, causal=True: torch.zeros_like(q)
sys.modules["flash_attn_interface"] = stub

path = Path("candidates/202603281811_salient-row-gptq/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

model = module.GPT(
    vocab_size=384,
    num_layers=2,
    model_dim=256,
    num_heads=8,
    num_kv_heads=4,
    mlp_mult=2.0,
    tie_embeddings=True,
    tied_embed_init_std=0.01,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    bigram_vocab_size=0,
    xsa_last_n=0,
    rope_dims=16,
    ln_scale=True,
    dtg=False,
    ve_enabled=False,
)
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
packed, meta, rescue = module.mixed_quantize_int6(sd, {"mlp", "attn"}, row_rescue_extra_budget_bytes=4096)
restored = module.dequantize_mixed_int6(packed, meta, sd)
assert set(restored) == set(sd)
assert rescue["selected_rows"] > 0, rescue
assert any(key.endswith(".rescue_idx") for key in packed), packed.keys()
for key in packed:
    if key.endswith(".rescue_idx"):
        base = key.removesuffix(".rescue_idx")
        idx = packed[key].long()
        restored_rows = restored[base][idx]
        source_rows = sd[base][idx].to(torch.float16).to(restored_rows.dtype)
        assert torch.equal(restored_rows, source_rows), base
print("quantization smoke ok", rescue)
PY
```

Observed outcome:

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603281811_salient-row-gptq/train_gpt.py` succeeded.
- The attempted CPU smoke test was **not runnable in this workflow container** because `torch` is not installed here (`ModuleNotFoundError: No module named 'torch'`).

I did **not** run a real training job here because this environment does not provide CUDA or the full FlashAttention runtime that this script requires for meaningful execution.

## Main expected risks / tradeoffs

- The rescue score is based on reconstruction error, not full AWQ-style activation statistics, so it may miss some truly salient rows.
- Duplicating rescued rows in fp16 increases artifact size; the adaptive budget logic is there to keep that bounded.
- The extra exported metadata may compress worse than hoped if the chosen rows are too noisy.
- `LeakyReLU(0.5)^2` was strong on the current SOTA stack, but its interaction with this exact export tweak is still unverified.
