# Mirror-Shared KV/MLP Banks + Expanded Lexical Memory

## Hypothesis

The current best stack is strong enough that the next useful gain is probably not another small optimizer or eval tweak. Instead, this candidate tries to reclaim artifact budget by **sharing the heavy KV and MLP parameter banks across mirrored encoder/decoder layers** in the 11-layer U-Net stack, while **keeping Q/out banks and per-layer control tensors unique** so depth-specific routing is preserved.

The saved bytes are then reinvested in cheap lexical memory:

- **BigramHash default: 8192** buckets
- **Shared value embedding default: 160 dim on layers 7,8,9,10**

This aims to keep most of the winning 11-layer behavior while moving more parameters into token-pair and token-identity features that are relatively cheap at runtime.

## Why this is promising here

Repository evidence points to a very stable winning recipe:

- top stack: legal TTT + LeakyReLU(0.5)^2 + parameter banking + Parallel Muon  
  `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- strongest non-TTT family: 11L + XSA4 + EMA + partial RoPE + LN scale + GPTQ-lite  
  `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
  `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- larger BigramHash tables helped earlier records, but the best recent stack was artifact-tight  
  `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md`

At the same time, prior notes say **naive depth recurrence / looped layers were not worth it under the 10-minute budget**, so this candidate avoids extra recurrent steps and only shares weights across the already-existing 11 layers.  
Reference: `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`

## Prior experiments that influenced this candidate

- **03-23 top record**: base scaffold, LeakyReLU^2 MLP, parameter banks, Parallel Muon, legal TTT, GPTQ-lite int6 + lzma.
- **03-22 / 03-21 records**: XSA4, EMA, partial RoPE, LN scale, VE path, warmdown 3500.
- **03-20 Int5 BigramHash record**: evidence that larger hashed bigram tables can still buy measurable BPB gains.
- **No prior `candidates/` directory existed** when this candidate was created.

## External research that informed it

Primary sources:

- **ALBERT** — cross-layer parameter sharing as a way to trade redundant depth parameters for other capacity: <https://arxiv.org/abs/1909.11942>
- **Universal Transformer** — repeated/shared blocks can keep transformer-style modeling power while changing the depth/parameter tradeoff: <https://arxiv.org/abs/1807.03819>

The adaptation here is deliberately conservative: unlike a full recurrent-depth design, this candidate keeps the existing 11 forward layers and only shares the expensive KV/MLP banks across mirrored U-Net positions.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Mirrored bank sharing**
   - default `MIRROR_SHARE_KV=1`
   - default `MIRROR_SHARE_MLP=1`
   - default `MIRROR_SHARE_QO=0`
   - mirror map for 11 layers: `[0,1,2,3,4,5,4,3,2,1,0]`

2. **Expanded lexical memory**
   - `BIGRAM_VOCAB_SIZE=8192`
   - `VE_DIM=160`
   - `VE_LAYERS=7,8,9,10`

3. **Recipe defaults aligned to the candidate**
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`
   - `EMA_DECAY` is now configurable and defaults to `0.997`

4. **Import / CPU-forward friendliness**
   - the script falls back to PyTorch SDPA when `flash_attn_interface` is unavailable, which makes local import and non-CUDA forward passes possible when PyTorch is installed

Everything else intentionally stays close to the strongest existing stack: LeakyReLU(0.5)^2, XSA4, partial RoPE, LN scale, Parallel Muon, EMA/SWA, GPTQ-lite-style int6 export, and legal score-first TTT.

## How to run

From this directory:

```bash
RUN_ID=mirror_shared_kvmlp \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful toggles for ablation:

```bash
MIRROR_SHARE_KV=0
MIRROR_SHARE_MLP=0
MIRROR_SHARE_QO=1
BIGRAM_VOCAB_SIZE=4096
VE_DIM=128
VE_LAYERS=9,10
TTT_ENABLED=0
```

## Main risks / tradeoffs

- **Too much sharing may underfit** even with unique Q/out banks and per-layer norms/scales.
- **BigramHash / VE expansion may saturate** before it repays the lost KV/MLP uniqueness.
- **TTT remains expensive** at evaluation time, so any training win still has to coexist with a heavy eval path.
- The export path is still int6+lzma based, so gains depend on both model quality and the quantized artifact shape.

## Validation

Executed from repo root:

```bash
python -m compileall candidates/202604041248_mirror-shared-kvmlp/train_gpt.py train_gpt.py train_gpt_mlx.py data
```

Outcome: **passed**.

Attempted CPU smoke check:

```bash
cd candidates/202604041248_mirror-shared-kvmlp
python - <<'PY'
import torch
import train_gpt as tg
...
PY
```

Outcome: **blocked on this runner** because the environment does not currently have PyTorch installed (`ModuleNotFoundError: No module named 'torch'`), so only syntax-level validation was possible here.
