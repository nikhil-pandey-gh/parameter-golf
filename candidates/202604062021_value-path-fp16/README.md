# Value-Path FP16 on the 11L GPTQ-lite Stack

## Hypothesis

The current frontier in this repo looks more bottlenecked by **post-training quantization damage** than by pre-quant model quality. Instead of paying the large byte cost of keeping the full tied embedding in fp16, this candidate keeps only the most sensitive **value-path tensors** in fp16 during export:

- `ve_shared.embed.weight`
- the final block's `attn.c_k.weight`
- the final block's `attn.c_v.weight`

The bet is that these tensors carry a disproportionate amount of late-context and token-identity signal, so preserving them should recover more post-quant BPB per byte than a broader precision rollback.

## Why this is promising here

Three repo trends point in the same direction:

1. The best runs improved faster by reducing quantization loss than by chasing raw pre-quant loss.
2. The old fp16-embedding record showed that a small set of sensitive tensors can dominate roundtrip degradation.
3. The strongest 11-layer stacks already add extra structure on the **value path** (`VE128` on late layers), which makes that path a natural place to spend the remaining byte budget.

This candidate is deliberately conservative on training-time changes and aggressive only on the export side, because that is where the recent records have repeatedly found cheap gains.

## Prior repo work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best pre-TTT stack in the repo
  - already has the 11L / EMA / GPTQ-lite / late-QAT / partial-RoPE recipe I want to preserve
- **Quantization sensitivity precedent:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - demonstrated that protecting a sensitive tensor in fp16 can erase most of the quantization gap
- **Late value-path structure precedent:** the same March 22 stack's `VE128` late-layer value embedding
- **Repository review result:** there were **no prior `candidates/` directories**, so this is the first candidate folder rather than a follow-up to an earlier candidate

## External research that informed it

- **LLM.int8()** — Dettmers et al., 2022  
  https://arxiv.org/abs/2208.07339  
  Key takeaway: transformer quality can be preserved by isolating a small set of outlier-sensitive dimensions/tensors into higher precision instead of reverting everything.
- **SmoothQuant** — Xiao et al., 2023  
  https://arxiv.org/abs/2211.10438  
  Key takeaway: quantization difficulty is uneven, and the right move is often to handle the sensitive parts specially rather than applying uniform low precision everywhere.

Those papers are about larger LLMs, but the repo's own records suggest the same principle is even more relevant in this tiny-model, hard-byte-cap regime.

## What changed vs the chosen base

Starting from the March 22 record stack, this candidate makes two precise changes:

1. **Script-relative repo defaults**
   - `DATA_PATH` and `TOKENIZER_PATH` now locate the repository root by walking upward until they find the repo's `data/` directory and top-level `README.md`, so the script can be launched from this candidate directory and still survives a later move into a normal `records/...` layout.
2. **Selective fp16 export exceptions in the mixed int6/int8 serializer**
   - add explicit fp16 passthrough for the large shared value embedding table
   - add explicit fp16 passthrough for the final block's K/V projections
   - log the exact fp16 export patterns at runtime

There is also a small **runtime fallback**: if `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA instead of failing at import time. The intended fast path is still FlashAttention 3.

## Expected byte impact

Using the default 11L / 512d / 4-KV-head setup, the added raw export cost versus per-row int8 is approximately:

| Tensor set | Estimated extra bytes |
|---|---:|
| `ve_shared.embed.weight` | 129,024 |
| final `attn.c_k.weight` | 130,560 |
| final `attn.c_v.weight` | 130,560 |
| **Total** | **390,144** |

`ve_shared.proj.weight` is not counted above because it is already auto-passthrough fp16 in the source stack (`numel <= 65,536`), so it is not an additional byte cost introduced by this candidate.

That is small enough to be plausible on top of the March 22 artifact budget, but it still needs a real GPU export run to confirm final compressed size.

## How to run

From this candidate directory:

```bash
cd candidates/202604062021_value-path-fp16
SEED=1337 \
VALUE_FP16_VE=1 \
VALUE_FP16_LAST_KV_LAYERS=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

- disable the idea entirely: `VALUE_FP16_VE=0 VALUE_FP16_LAST_KV_LAYERS=0`
- keep only VE tables: `VALUE_FP16_VE=1 VALUE_FP16_LAST_KV_LAYERS=0`
- keep more late KV layers if the artifact still fits: `VALUE_FP16_LAST_KV_LAYERS=2`
- add any extra explicit fp16 names: `EXTRA_FP16_EXPORT_PATTERNS=blocks.9.attn.c_v.weight`

## Validation

Commands run in this container:

```bash
python -m compileall candidates/202604062021_value-path-fp16/train_gpt.py
python - <<'PY'
import importlib.util
print({
    "torch": importlib.util.find_spec("torch") is not None,
    "flash_attn_interface": importlib.util.find_spec("flash_attn_interface") is not None,
})
PY
```

Outcomes:

- `python -m compileall ...` **passed**
- runtime dependency probe returned `{"torch": False, "flash_attn_interface": False}`
- because this container does not have the training runtime installed, a true import-time or CPU forward-pass smoke test was **not feasible here**

## Main risks and tradeoffs

- The extra fp16 tensors may still push the compressed artifact too close to 16MB; if so, the first trim should be `VALUE_FP16_LAST_KV_LAYERS=0` or a smaller `BIGRAM_VOCAB_SIZE`.
- The chosen tensors are a targeted bet, not yet repo-ablated. It is possible that VE-only or V-only protection is the better byte/quality trade.
- The SDPA fallback is for robustness only; the 10-minute target should still be judged with the FlashAttention path used by the source record stack.
