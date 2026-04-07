# Mirror-Shared Depth on the 2026-03-22 EMA + GPTQ-lite Base

## Hypothesis

The late encoder and decoder blocks in the current 11-layer U-Net-style stack are similar enough that their heavy attention/MLP matrices can be shared in a mirrored pattern without collapsing layer specialization, as long as each logical layer keeps its own cheap residual controls (`resid_mix`, `attn_scale`, `mlp_scale`, optional gate) and its own XSA / value-embedding placement.

If that works, the model should get:

1. stronger compression regularity from repeated block weights,
2. fewer exported heavy tensors for the same logical depth,
3. enough recovered artifact budget to safely raise the hashed bigram side channel from 2048 to 3072 buckets.

## Why this is promising here

Local record history shows the repository has already squeezed a lot out of:

- deeper 11-layer U-Net stacks,
- 3x MLPs,
- XSA on late layers,
- EMA + tight SWA,
- partial RoPE + layer scaling,
- GPTQ-lite / int6 export,
- evaluation-aware tricks like sliding windows and TTT.

What it has **not** really explored is cross-layer parameter sharing inside the best pure train/export stack. Earlier local recurrence experiments were weak, but those were not this design: this candidate keeps the mature 2026-03-22 architecture, preserves per-layer control tensors, and only shares the expensive core weights.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
  Chosen base stack: 11L, EMA, GPTQ-lite, warmdown 3500, partial RoPE, LN scaling, VE, BigramHash, XSA.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`  
  Reinforced that partial RoPE + LN scaling are stable zero-parameter wins.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`  
  Reinforced the strong late-layer XSA + EMA recipe.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`  
  Motivated the 3072-bucket BigramHash bump via its positive 2048 -> 3072 ablation, even though this candidate does **not** adopt the TTT or LeakyReLU changes.

## External research that informed it

- **ALBERT** (Lan et al., 2019), arXiv:1909.11942  
  Cross-layer parameter sharing can preserve quality while reducing parameter count.
- **Universal Transformers** (Dehghani et al., 2018), arXiv:1807.03819  
  Shared-depth computation can improve sequence modeling when recurrence is stabilized.
- **Basis Sharing** (Wang et al., 2024), arXiv:2410.03765  
  Cross-layer shared bases improve compression under tight parameter budgets.
- **CommonKV** (Wang et al., 2025), arXiv:2508.16134  
  Adjacent layers often have enough similarity to support cross-layer sharing.
- **Thinking Deeper, Not Longer** (Chen, 2026), arXiv:2603.21676  
  Identity-biased recurrence and layer-scale style stabilizers make shared-depth computation more viable; this candidate keeps LN scaling plus per-layer residual mixing for that reason.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Mirror-shared cores**
   - The model still runs **11 logical layers**.
   - The heavy block weights are reduced to **6 shared cores** with logical-to-core map:
     - `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]`
   - This shares the attention/MLP weights across mirrored encoder/decoder depths.

2. **Per-layer adapters retained**
   - Each logical layer still owns:
     - `resid_mix`
     - `attn_scale`
     - `mlp_scale`
     - optional gating path
     - its own XSA enable/disable choice
   - This is the main difference from a naive “just loop one block” recurrence.

3. **Bigger lexical side channel**
   - `BIGRAM_VOCAB_SIZE` default: **3072** instead of 2048.

4. **Slightly broader VE placement**
   - `VE_LAYERS` default: **`8,9,10`** instead of `9,10`.

5. **Validation ergonomics**
   - Default data/tokenizer paths resolve relative to the repository root even when run from this candidate directory.
   - FlashAttention falls back to SDPA when unavailable.
   - `SMOKE_TEST=1` runs a small CPU-safe forward/backward check without dataset access.

## How to run

From the candidate directory:

```bash
cd candidates/202604072124_mirror-shared-depth
RUN_ID=mirror_shared_depth \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# disable sharing and compare against a close base-like path
SHARE_MIRROR=0 RUN_ID=no_sharing torchrun --standalone --nproc_per_node=8 train_gpt.py

# revert the bigger bigram side channel
BIGRAM_VOCAB_SIZE=2048 RUN_ID=bigram_2048 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Evaluation

The script keeps the mature 2026-03-22 evaluation/export flow:

- EMA application before export,
- GPTQ-lite-style int6 export,
- roundtrip validation,
- sliding-window evaluation.

## Main expected risks / tradeoffs

- **Over-sharing risk**: mirrored encoder/decoder layers may need more specialization than cheap control tensors can provide.
- **No training-speed win**: sharing reduces artifact parameters, not FLOPs; 10-minute wallclock training time does not get easier.
- **Past local recurrence skepticism**: earlier recurrence-style attempts were weak in this repo, so this candidate is explicitly betting that mirrored sharing on top of the mature 03-22 stack is materially different from naive looping.
- **Compression vs. representation tradeoff**: the exported model should compress more easily, but shared cores could cap representational diversity.

## Validation run here

Commands executed in this workspace:

```bash
cd candidates/202604072124_mirror-shared-depth
/tmp/gh-aw/agent/pg-venv/bin/python -m compileall train_gpt.py
SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python train_gpt.py
```

Observed outcome:

```text
smoke_test:ok device=cpu loss=6.9584 params=15328309 shared_cores=6 layer_map=[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
```

I did **not** run a full GPU train/eval job in this environment.
