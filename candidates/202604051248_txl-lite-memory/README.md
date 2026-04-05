# TXL-lite top-layer memory

## Hypothesis

Train the strongest current compact stack with **Transformer-XL-style detached segment recurrence** on the deepest layers, so the model learns to use longer context during training instead of relying only on sliding-window evaluation at the end.

The core bet is that this repository has already shown three things:

1. **Longer context helps** (`seq2048`, `seq4096`, sliding eval).
2. **Tiny zero/low-parameter architectural tweaks still move the needle** on the 11-layer stack.
3. **Depth recurrence is a bad fit** for a 10-minute wallclock budget, but **segment recurrence is different** because it expands context, not depth.

This candidate tries to capture some of the long-context benefit with almost no extra parameters and without changing the root baseline.

## Why this is promising here

The repo evidence points in a very specific direction:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the current strongest overall stack, so this candidate starts there.
- `records/track_10min_16mb/2026-03-18_LongContextSeq2048/` and `records/track_10min_16mb/2026-03-19_TrainingOptSeq4096/` both show that more context helps, but they still pay for conventional full attention.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/` shows a large gain from giving each scored token more left context at evaluation time.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` and `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` both report **layer/depth recurrence** as a dead end. This candidate deliberately avoids that path.

There was **no prior `candidates/` directory** in the repo when this was implemented, so this is the first candidate iteration in that namespace.

## External research that informed it

1. **Transformer-XL** (Dai et al., 2019): segment-level recurrence improves long-range dependency modeling without increasing parameter count.  
   https://arxiv.org/abs/1901.02860
2. **Compressive Transformer** (Rae et al., 2019): memory can be extended and compressed rather than paid for as new core weights.  
   https://arxiv.org/abs/1911.05507
3. **Recurrent Memory Transformer** (Kuratov et al., 2022): lightweight memory augmentation remains competitive and can be added with relatively small architectural changes.  
   https://arxiv.org/abs/2207.06881

The implementation here is intentionally **Transformer-XL-lite**, not a full reproduction: it uses a short detached memory window on the top layers only, which is much easier to graft onto the current record code.

## Base implementation and prior influences

- **Chosen base:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`
- **Direct influences from prior records:**
  - LeakyReLU(0.5)^2 MLP from the current best record
  - parameter banking + parallel Muon from the current best record
  - XSA / partial RoPE / LN scale / VE from the late 11-layer record lineage
  - longer-context motivation from the `seq2048` and `seq4096` runs
- **Prior dead ends explicitly avoided:**
  - looping/reusing layers to simulate extra depth
  - another pure LR / warmdown / quant-only tweak without a new modeling change

## What changed vs. the chosen base

1. Added **script-relative dataset/tokenizer defaults**, so `train_gpt.py` can be run from this candidate directory directly.
2. Added new recurrence knobs:
   - `SEGMENT_LEN` (default `1024`)
   - `MEM_LEN` (default `512`)
   - `MEM_LAST_N` (default `4`)
3. Changed the default training/eval length to **4096** tokens, while processing the sequence internally in 1024-token segments.
4. Added **detached top-layer memory**:
   - only the deepest `MEM_LAST_N` layers receive recurrent memory
   - memory stores the prior segment’s layer inputs
   - only the last `MEM_LEN` tokens are retained
5. Added an attention path that:
   - keeps FlashAttention on the normal no-memory path
   - falls back to SDPA when recurrent memory is active
   - applies RoPE over the current `[memory + segment]` window
6. Added a small **`CPU_SMOKE_TEST=1`** mode so the candidate can be sanity-checked without CUDA.

## How to run

From this candidate directory:

```bash
cd candidates/202604051248_txl-lite-memory
RUN_ID=txl_lite_memory torchrun --standalone --nproc_per_node=8 train_gpt.py
```

That uses the candidate defaults:

- `TRAIN_SEQ_LEN=4096`
- `EVAL_SEQ_LEN=4096`
- `SEGMENT_LEN=1024`
- `MEM_LEN=512`
- `MEM_LAST_N=4`
- `TTT_ENABLED=0`

If you want to try the current repo’s legal TTT recipe on top of this candidate later, re-enable it explicitly:

```bash
cd candidates/202604051248_txl-lite-memory
RUN_ID=txl_lite_memory_ttt TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Evaluation intent

The hope is to get part of the “more left context helps” effect **inside training**:

- full 4096-token dense attention is avoided
- lower layers stay local and cheap
- top layers get a short recurrent memory window
- the artifact budget barely changes because the memory is runtime state, not stored weights

If this works, the most likely win is against the older long-context runs and potentially against the current pre-TTT stack, with legal TTT as a follow-up rather than part of the initial bet.

## Validation run locally

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604051248_txl-lite-memory/train_gpt.py` | Passed |
| `cd candidates/202604051248_txl-lite-memory && CPU_SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python train_gpt.py` | Passed: `cpu_smoke_ok loss≈6.95 logits_shape:(1, 256, 1024)` |

The CPU smoke path uses a tiny segmented forward pass and confirms that the recurrent-memory code path starts correctly from the candidate directory.

## Main risks and tradeoffs

1. This is **not exact Transformer-XL positional handling**; it is a pragmatic RoPE-window approximation over `[memory + current segment]`.
2. Only the deepest layers get memory. If the right inductive bias needs earlier-layer recurrence, this may be too conservative.
3. The memory path uses SDPA instead of FlashAttention, so step time could regress if `MEM_LAST_N` or `MEM_LEN` is too large.
4. The interaction with the current legal TTT recipe is still uncertain, so this candidate leaves TTT off by default.
5. If the benchmark’s biggest gains are still dominated by evaluation protocol rather than trained context use, this could underperform despite being a cleaner modeling idea.
