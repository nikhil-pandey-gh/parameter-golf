# Signed Hadamard GPTQ-lite

## Hypothesis

The current frontier in this repository is heavily limited by **post-training quantization error**, not just by pre-quant model quality. A deterministic, export-time **signed Hadamard rotation search** should spread row outliers before per-row quantization, allowing the existing mixed int6/int8 export path to preserve more model quality at the same byte budget.

In short: keep the March 23 architecture and training stack, but make the saved weights easier to quantize.

## Why this is promising for this repository

Repository history points to quantization as a persistent bottleneck:

- the root baseline already uses int8 roundtrip export, and early wins came from reducing the quantization penalty,
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/README.md` shows that even a 4-hour run still degrades badly after export,
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` got a real win from a better clip search alone,
- the recent winning line already looks saturated on small architectural tweaks, so a more geometry-aware export step is one of the cleanest remaining levers.

This candidate keeps training unchanged and only improves the **quantize -> serialize -> dequantize -> eval** path, so it is cheap to test and easy to ablate.

## Records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Supporting evidence and prior art inside the repo:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` for GPTQ-lite percentile search
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for the current 11-layer XSA/partial-RoPE/LN-scale line
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` for evidence that better training alone does not remove the export bottleneck

There were **no prior experiments under `candidates/`** in this checkout.

## External research that informed this candidate

- **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** (Ashkboos et al., arXiv:2404.00456) shows that orthogonal rotations can remove outlier structure and dramatically improve low-bit quantization while preserving the underlying model function.
- **SpinQuant: LLM quantization with learned rotations** (Liu et al., arXiv:2405.16406) shows that rotation choice matters a lot, and that rotation-aware quantization can outperform plain random rotation baselines.
- **Scaling Law for Quantization-Aware Training** (Chen et al., arXiv:2505.14302) argues that weight quantization error becomes increasingly important as models train longer and see more data, which matches this repo's observed quantization bottleneck.
- **BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks** (Nielsen and Schneider-Kamp, arXiv:2407.09527) is relevant because it suggests tiny models can benefit from more aggressive low-bit approaches, but full native 1.58-bit training is too invasive for a precise candidate in this repo.

## What changed vs the chosen base implementation

This folder starts from the March 23 SOTA script and only changes the export path.

### Added

- deterministic **signed Hadamard transform helpers**,
- a **rotation search** over `none`, `hadamard_right`, `hadamard_left`, and `hadamard_both` for eligible 2D tensors,
- per-tensor metadata recording which rotation was chosen.

### Changed

- int6 GPTQ-lite export now evaluates each allowed rotation mode using reconstruction MSE after dequantization and inverse rotation,
- the same rotation-aware search is also applied to large int8-exported tensors such as embeddings,
- dequantization inverts the saved rotation before loading the eval model.

### Unchanged

- training loop,
- optimizer stack,
- Parameter Banking / Parallel Muon,
- LeakyReLU(0.5)^2 activation,
- XSA / partial RoPE / VE / EMA / optional TTT behavior.

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603260855_signed-hadamard-gptq
ROTATION_GPTQ=1 TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablation:

```bash
cd candidates/202603260855_signed-hadamard-gptq
ROTATION_GPTQ=0 TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `ROTATION_GPTQ=1` is the default for this candidate.
- Rotation search only activates on sufficiently large **power-of-two** matrix axes (`ROTATION_GPTQ_MIN_DIM`, default `256`).
- `TTT_ENABLED=0` keeps the comparison focused on the export change. You can still re-enable TTT later to test whether the gain stacks with the current best evaluation recipe.

## Main expected risks and tradeoffs

- Lower MSE after inverse rotation may not always translate into lower validation BPB.
- The extra search increases export-time CPU work.
- The current best record is already close to the 16MB limit, so the extra code must earn its keep.
- Some matrices may prefer no rotation at all; this is why the candidate searches `none` alongside the rotated options.

## Validation

Commands run in this repository checkout:

```bash
python -m compileall candidates/202603260855_signed-hadamard-gptq/train_gpt.py
python - <<'PY'
import importlib.util
mods = ['flash_attn_interface', 'torch', 'sentencepiece', 'numpy']
for name in mods:
    print(f'{name}={bool(importlib.util.find_spec(name))}')
PY
```

Observed outcomes:

- `python -m compileall ...` **succeeded**.
- A runtime smoke test was **not feasible in this container** because the required Python runtime packages were not installed locally (`flash_attn_interface=False`, `torch=False`, `sentencepiece=False`, `numpy=False`).
- The checkout also does not contain local FineWeb shard data or tokenizer model files under `data/datasets/...`, so there is no safe local training/eval input to launch even after dependency installation.

The intended next validation step is a normal remote GPU run comparing `ROTATION_GPTQ=0` versus `ROTATION_GPTQ=1` on the same seed and measuring the post-roundtrip `val_bpb` delta.
