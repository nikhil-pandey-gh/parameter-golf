# Late Block Sharing on the 11L EMA + GPTQ-lite Stack

## Hypothesis

Pairwise sharing of the **heavy projection modules** in the deeper half of the best non-TTT 11-layer stack can improve the parameter-efficiency/artifact-efficiency tradeoff without giving up the layer-specific control tensors that recent records rely on. The expected upside is that late layers are redundant enough for shared Q/K/V/O and MLP weights to regularize training, while untied norms, residual mixes, scales, RoPE/XSA configuration, and value-embedding scalars preserve per-layer specialization.

## Why this is promising here

- The repository's strongest runs mostly squeeze gains from **quantization, eval, EMA/SWA, XSA, partial RoPE, and MLP width**.
- The repo survey found **no prior `candidates/` directory** and no prior record centered on **cross-layer parameter sharing**.
- Recent compact-model work argues that at sub-billion scale, **architecture choices matter as much as raw parameter count**, and MobileLLM-LS reports gains from **immediate block-wise weight sharing** with only marginal latency overhead.
- This challenge scores under a strict **16 MB artifact cap**, so a model-side efficiency idea is more interesting than another eval-only trick.

## Prior repo work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- **Architecture lineage:**  
  `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` -> `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` -> `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
- **Why not the 2026-03-23 record as the base:** that run is stronger overall, but its legal TTT + parameter-banked optimizer stack makes it harder to attribute the effect of a new architecture-side idea cleanly.

## External research that informed it

1. **MobileLLM / MobileLLM-LS** — <https://arxiv.org/abs/2402.14905>  
   Key takeaway used here: on sub-billion models, **deep-thin architectures plus immediate block-wise weight sharing** can beat stronger baselines with **no model-size increase and marginal latency overhead**.
2. **ALBERT** — <https://arxiv.org/abs/1909.11942>  
   Key takeaway used here: **cross-layer parameter sharing** is a practical way to cut parameter count and training memory without collapsing model quality if the sharing is structured.

## What changed vs the chosen base

1. **Late pairwise block sharing**
   - New defaults:
     - `SHARE_GROUP_SIZE=2`
     - `SHARE_START_LAYER=3`
     - `SHARE_ATTN=1`
     - `SHARE_MLP=1`
   - Layers `3-10` are grouped as `(3,4)`, `(5,6)`, `(7,8)`, `(9,10)`.
   - Within each pair, the candidate shares:
     - attention `c_q`, `c_k`, `c_v`, `proj`
     - MLP `fc`, `proj`

2. **Per-layer control remains untied**
   - Each layer still keeps its own:
     - `attn_norm`, `mlp_norm`
     - `q_gain`
     - `attn_scale`, `mlp_scale`
     - `resid_mix`
     - `dtg_gate` if enabled
     - RoPE/XSA/value-embedding wiring

3. **Alias-aware export**
   - The candidate deduplicates exact tensor aliases before int6 quantization.
   - Quantized exports now carry an alias map and restore shared tensors on load, so the artifact path actually respects the sharing instead of silently serializing the same weight twice.

4. **Repo-root default paths**
   - The copied script now walks upward until it finds the repository root (`README.md` + `data/`) before building default `DATA_PATH` and `TOKENIZER_PATH`.
   - This keeps the script runnable both from **this candidate directory** and after a future move into a `records/...` submission folder.

## How to run

From the repository root:

```bash
cd candidates/202604040106_late-block-sharing
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Or, if you want the new knobs to be explicit:

```bash
cd candidates/202604040106_late-block-sharing
SHARE_GROUP_SIZE=2 SHARE_START_LAYER=3 SHARE_ATTN=1 SHARE_MLP=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the `2026-03-22` defaults for the rest of the stack: 11 layers, MLP3x, XSA on the last 4 layers, partial RoPE, LN scale, EMA, GPTQ-lite-style clip search, warmdown 3500, and stride-64 sliding eval.

## Expected risks / tradeoffs

- **Quality risk:** sharing late projections may remove too much capacity and hurt BPB even if the artifact gets smaller.
- **Compute risk:** sharing does not reduce forward FLOPs, so training speed should be close to the base rather than dramatically faster.
- **Export risk:** this first candidate only deduplicates exact aliases; it does not try to re-layout weights into a more compressor-friendly custom format.
- **Search risk:** the best `SHARE_START_LAYER` may not be `3`; deeper-only or attention-only sharing may work better.

## Validation

Commands run locally in this workflow:

```bash
python -m compileall candidates/202604040106_late-block-sharing/train_gpt.py
python - <<'PY'
try:
    import torch
    print('torch_import:ok')
except Exception as exc:
    print(f'torch_import:failed:{exc.__class__.__name__}:{exc}')
PY
```

Outcomes:

- `python -m compileall ...` **succeeded**
- A real CPU import/smoke run was **not feasible in this runner** because `import torch` fails with `ModuleNotFoundError`, and this script also depends on the CUDA/FlashAttention training stack

## Suggested next experiments

1. Sweep `SHARE_START_LAYER` across `{5, 7, 9}` to test whether deeper-only sharing is safer.
2. Try **attention-only** vs **MLP-only** sharing (`SHARE_ATTN` / `SHARE_MLP`) to see which side is more redundant in this stack.
3. If artifact bytes fall comfortably, reinvest the saved budget into a slightly larger `BIGRAM_VOCAB_SIZE` or `VE_DIM`.
