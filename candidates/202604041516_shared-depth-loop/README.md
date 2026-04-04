# Shared-Depth Loop + Per-Layer Controls

## Hypothesis

Under the 16MB artifact cap, **effective depth is probably more valuable than fully unique block weights** once the repo's quantization stack is already strong. This candidate keeps a 12-layer forward path, but only stores **6 unique transformer block cores reused across 2 loops**, while preserving **per-effective-layer norms, residual mixers, scales, skip weights, XSA placement, and value-embedding injection** so the model can still specialize by depth.

## Why this is promising here

The current record progression already squeezed a lot out of:

- 11-layer / 2048-token training,
- MLP 3x,
- SmearGate + BigramHash,
- XSA on deep layers,
- partial RoPE + LN scale,
- EMA + GPTQ-lite / mixed int6 export,
- eval-time sliding windows and TTT.

That means the clearest remaining gap is an **architecture change that buys artifact headroom without adding much infrastructure**. Naive recurrence was previously negative in this repo, but this candidate is a different bet:

1. it shares only the **heavy attention/MLP weights**,
2. it keeps **unique lightweight per-layer controls**,
3. it preserves the repo's existing deep-layer tricks instead of replacing them.

## Prior local influence

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Activation carryover:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` motivated porting **LeakyReLU(0.5)^2**
- **Dead-end to avoid:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` both note that naive depth recurrence/looping can lose too many steps; this candidate tries to keep the good part (byte savings) while restoring specialization through unique controls

There were **no prior `candidates/` directories** in this checkout.

## External research that informed it

- **Universal Transformer** — recurrent depth reuse can trade parameters for computation: <https://arxiv.org/abs/1807.03819>
- **ALBERT** — cross-layer parameter sharing is a strong parameter-efficiency primitive: <https://arxiv.org/abs/1909.11942>
- **DenseFormer** — learned reuse of earlier depth states can improve perplexity with very few extra parameters: <https://arxiv.org/abs/2402.02622>
- **Attention Residuals** — fixed unit residual accumulation is suboptimal; depth-wise selection matters: <https://arxiv.org/abs/2603.15031>
- **Residual Stream Duality in Modern Transformer Architectures** — shortcut design across depth is a legitimate optimization target, not just plumbing: <https://arxiv.org/abs/2603.16039>

## What changed vs the chosen base

Starting from the 2026-03-22 pure-training stack, this candidate:

1. replaces 11 unique blocks with **12 effective layers = 6 shared block cores x 2 loops**
2. keeps **unique per-effective-layer** `attn_norm`, `mlp_norm`, `attn_scale`, `mlp_scale`, `resid_mix`, optional `dtg_gate`, and skip weights
3. keeps XSA on the **last effective layers**, instead of baking it into the shared cores
4. ports **LeakyReLU(0.5)^2** into the MLP
5. keeps the existing **EMA + GPTQ-lite mixed int6 export** path intact
6. makes dataset/tokenizer defaults **relative to the repository root**, so the script can be launched from this candidate directory directly

## Files added

- `README.md`
- `train_gpt.py`

## How to run

From this candidate directory:

```bash
RUN_ID=shared_depth_loop \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Important defaults baked into this script:

- `NUM_UNIQUE_LAYERS=6`
- `NUM_LOOPS=2`
- `NUM_LAYERS=12`
- `MLP_NEGATIVE_SLOPE=0.5`
- `XSA_LAST_N=4`
- `VE_LAYERS=10,11`

The script resolves `DATA_PATH` and `TOKENIZER_PATH` relative to the repo root by default, so it is runnable from `candidates/202604041516_shared-depth-loop/` without extra path edits.

## How to evaluate

The script keeps the base record's flow:

1. training under the wall-clock cap,
2. EMA application,
3. mixed int6 export with GPTQ-lite clip search,
4. round-trip validation,
5. sliding-window eval at stride 64.

## Validation

Commands run in this workflow:

```bash
# from the repo root
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604041516_shared-depth-loop/train_gpt.py

# from candidates/202604041516_shared-depth-loop/
python -m compileall train_gpt.py

# dependency check
python -m pip show torch sentencepiece
```

Outcomes:

- `compileall` succeeded for the root scripts, `data/`, and this candidate script
- a CPU forward smoke test was **not feasible in this container** because both `torch` and `sentencepiece` are absent here (`pip show` reported them missing), so importing the runtime module stack already fails before model construction

## Main risks and tradeoffs

- **Repo evidence cuts both ways:** previous naive recurrence was negative, so shared-depth may still hurt if specialization from the lightweight controls is not enough
- **Slightly more effective depth:** 12 effective layers may cost some step throughput versus the 11-layer base
- **Hyperparameter retuning likely matters:** block sharing changes the optimization landscape, especially for LR, warmdown, and XSA placement
- **Byte savings may not automatically become BPB gains:** the main upside is artifact efficiency; converting that into a better score may require a second pass that reinvests saved bytes into width, bigger bigram tables, or different eval-time tricks
