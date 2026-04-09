# 202604090932_mtp-aux

## Hypothesis

A small **training-only multi-token prediction (MTP)** objective can improve sample efficiency for this repo's strong 11-layer stack without increasing exported artifact size, because the auxiliary heads are excluded from the final quantized checkpoint. Seeding those heads from the tied embedding weights and weighting nearer future horizons more strongly should make the auxiliary loss easier to optimize than the dormant zero-init uniform-loss path already present in recent records.

## Why this is promising here

- The current strongest non-TTT 11-layer line already has a dormant MTP path in code, but all shipped runs keep `MTP_NUM_HEADS=0`, so this is a real gap rather than a repeated record.
- This challenge rewards **train-time-only helpers** that improve the exported trunk without paying inference or artifact cost. The base script already strips `mtp_heads.*` from export.
- Multi-token prediction is directly aimed at improving sample efficiency and future-token representations, which matters in a hard 600-second wallclock regime.

## Prior repository evidence

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Strong 11-layer recipe: EMA + GPTQ-lite clip search + int6 mixed quantization + partial RoPE + XSA + VE + BigramHash.
- **Influential records:**
  - `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` showed that the 11-layer XSA/partial-RoPE stack is strong and stable.
  - `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` is the current top record, but it adds significantly more infrastructure (TTT + parameter banking). This candidate instead targets a simpler training-path gain on the earlier strong stack.
- **Negative / cautionary evidence:**
  - The 1x5090 exploration in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` reports that brute-force recurrence/depth reuse was harmful under fixed wallclock, so this candidate avoids compute-heavy recurrence tricks.

## External research that informed this candidate

- Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** (arXiv:2404.19737)  
  https://arxiv.org/abs/2404.19737
- Guoliang Zhao et al., **Self-Distillation for Multi-Token Prediction** (arXiv:2603.23911)  
  https://arxiv.org/abs/2603.23911
- John Kirchenbauer et al., **Multi-Token Prediction via Self-Distillation** (arXiv:2602.06019)  
  https://arxiv.org/abs/2602.06019

These papers support the core idea that future-token auxiliary objectives can improve learning efficiency, and they also highlight that **jointly training auxiliary MTP heads well is non-trivial**. That motivated the small repo-specific twist here: initialize auxiliary heads from the already-strong lexical head geometry instead of leaving them at zero.

## What changed vs the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **MTP enabled by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Near-horizon weighted MTP loss**
   - Added `MTP_LOSS_DECAY=0.7` so the +1 token target has the strongest weight and farther horizons are treated as softer auxiliary supervision.
3. **Embedding-seeded MTP heads**
   - Added `MTP_COPY_LM_INIT=1`, which initializes each auxiliary MTP head from the token embedding matrix instead of leaving the auxiliary heads at zero.
4. **Self-describing exported checkpoints**
   - `final_model.pt` and `final_model.int6.ptz` now carry the stripped artifact `model_kwargs`, so reload paths can reconstruct the export-time no-MTP model shape explicitly.

Everything else stays on the proven 11-layer GPTQ-lite/EMA stack so this remains a precise candidate instead of a broad rewrite.

## How to run

From this candidate directory:

```bash
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key defaults inherited from the base:

- 11 layers, 512 width, 8 heads / 4 KV heads
- seq_len 2048, train batch 786,432 tokens
- EMA + tight SWA
- GPTQ-lite int6/int8 export with zstd (or zlib fallback)
- Partial RoPE, XSA on last 4 layers, shared value embeddings, BigramHash(2048)

To ablate the new idea:

```bash
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Risks / tradeoffs

- Extra MTP heads increase **training-time compute**, so gains in sample efficiency must outweigh any step-count loss.
- The best MTP horizon count and loss weight are not known for this exact stack; `2` heads and `0.15` weight are conservative defaults, not tuned optima.
- Embedding-seeded auxiliary heads are a pragmatic optimization trick, not a published guarantee.

## Validation

Commands run for this candidate:

```bash
python -m compileall candidates/202604090932_mtp-aux/train_gpt.py
python - <<'PY'
import importlib.util
print("torch_spec", importlib.util.find_spec("torch") is not None)
print("flash_attn_interface_spec", importlib.util.find_spec("flash_attn_interface") is not None)
PY
command -v nvidia-smi >/dev/null 2>&1 && echo nvidia_smi_present true || echo nvidia_smi_present false
```

Outcomes:

- `compileall` succeeded.
- This environment does **not** have `torch`, `flash_attn_interface`, or `nvidia-smi`, so a real runtime smoke test was not possible here.
- Even with Python dependencies present, this script inherits the base record's hard CUDA / FlashAttention 3 runtime requirement.
