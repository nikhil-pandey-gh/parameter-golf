# 202603260602 Bank-Aware FC2 Late-QAT

## Hypothesis

The strongest current banked stack still exports its large parameter banks through mixed int6/int8 quantization, but those bank tensors are not exposed to the old `CastedLinear` late-QAT path during training. A targeted late-QAT pass that fake-quantizes the bank weights themselves should reduce round-trip loss, and a mixed-precision suffix policy that protects the deepest MLP-down / FC2-style projections at int8 should specifically attack the FC2 outlier bottleneck.

## Why this is promising for this repository

The repo history is clear that the biggest durable wins came from quantization-aware choices rather than raw architecture churn: sliding evaluation, mixed precision, GPTQ-lite clip search, EMA, and other compression-aware tweaks consistently beat naive extra training. The latest top stack in `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` improves training throughput and eval, but its late-QAT hook still lives on `CastedLinear` modules while the dominant weight mass now lives in the parameter banks.

This candidate keeps the high-performing 11-layer banked base and only changes the part of the system that appears under-optimized for the current export path.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current banked base
  - contributes LeakyReLU(0.5)^2, parameter banking, Parallel Muon, EMA/SWA, optional legal TTT
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - shows GPTQ-lite clip search and late QAT remain relevant on strong 11L stacks
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - explicitly documents that a prior late-QAT implementation was ineffective under `torch.compile`
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/`
  - established that mixed precision on sensitive tensors can sharply reduce quantization loss

## External research that informed it

- Chen et al., *Scaling Law for Quantization-Aware Training* (arXiv:2505.14302, 2025)
  - reports that weight quantization error grows with more training tokens
  - identifies FC2-layer outliers as a primary bottleneck
  - shows mixed precision can bring weight and activation quantization errors closer together
- Yang et al., *Quantization-Aware and Tensor-Compressed Training of Transformers for Natural Language Understanding* (arXiv:2306.01076, 2023)
  - supports the broader idea that QAT recovers compression quality when the compressed tensors are actually seen during training

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Bank-aware late QAT**
   - the 3D attention/MLP banks now run through fake quantization during the late-QAT phase
   - this finally exposes the exported bank weights to quantization noise during training

2. **Suffix mixed precision for FC2-style banks**
   - deepest `mlp_down` / `blocks.*.mlp.proj.weight` slices use int8 export by default
   - all other large attention/MLP banks stay on GPTQ-lite int6
   - default knob: `MIXED_INT8_MLP_DOWN_LAST_N=4`

3. **Robust late-QAT activation path**
   - when late QAT turns on, training can switch from the compiled model back to eager mode
   - this avoids relying on compile-time constant behavior for the new bank-QAT branch

4. **CPU-safe attention fallback**
   - if `flash_attn_interface` is unavailable, the script falls back to PyTorch SDPA
   - this is mainly to make local smoke testing from the candidate directory feasible

5. **Candidate-directory-friendly data defaults**
   - default dataset and tokenizer paths resolve from the repository root instead of assuming the script is launched from repo root

## How to run or evaluate it

From the candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Important knobs:

```bash
BANK_QAT_ENABLED=1
BANK_QAT_EAGER=1
LATE_QAT_THRESHOLD=0.15
MIXED_INT8_MLP_DOWN_LAST_N=4
MIXED_INT8_ATTN_OUT_LAST_N=0
TTT_ENABLED=1
```

The defaults are already set to test the core idea, but `TTT_ENABLED=1` remains optional so the candidate can be evaluated either as a pure training/export change or with the legal score-first TTT path inherited from the base.

## Main expected risks or tradeoffs

- **Late-phase throughput loss**: switching to eager mode for late QAT may reduce step rate during the final training slice.
- **Artifact size risk**: deeper FC2 projections exported at int8 may compress slightly worse than int6-in-int8 containers, especially with the current `lzma` packaging.
- **Heuristic layer choice**: protecting the last 4 FC2-style layers is a budget-conscious heuristic, not a fully optimized bit-allocation search.
- **Potentially small gain**: the repo is already highly tuned, so even a correct quantization-path fix may only yield a modest BPB improvement.

## Validation

Commands run during implementation:

```bash
python -m compileall train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path("train_gpt.py").resolve()
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.GPT(
    vocab_size=64,
    num_layers=2,
    model_dim=32,
    num_heads=4,
    num_kv_heads=2,
    mlp_mult=2,
    tie_embeddings=True,
    tied_embed_init_std=0.01,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    mixed_int8_mlp_down_last_n=1,
).float()
x = torch.randint(0, 64, (2, 16))
y = torch.randint(0, 64, (2, 16))
loss = model(x, y)
print(float(loss))
PY
```

Observed outcomes:

- `python -m compileall train_gpt.py`: **passed**
- tiny CPU import/forward smoke test: **not feasible in this environment** because the available Python interpreter raised `ModuleNotFoundError: No module named 'torch'` before model construction
