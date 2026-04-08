# Tail-Tied Shared Cores + Lexical Memory

## Hypothesis

The strongest next step is to **tie only the deep block cores** of the best pre-TTT 11-layer stack, while keeping the cheap per-layer controls untied and reinvesting some of the saved bytes into slightly richer lexical/value memory.

The key distinction from the repo's earlier negative results on naive depth recurrence is that this candidate **does not reduce logical depth or add extra recurrent passes**. It keeps the same 11 logical layers and training-time compute, but stores fewer unique late-layer matrices in the final artifact.

## Why this is promising here

This repo's history says three things clearly:

1. **Compression matters as much as raw train loss.** The jump from the baseline to later records came from fp16-sensitive embeddings, mixed int6/int8 export, QAT, GPTQ-lite, and better averaging.
2. **Late-layer architectural tweaks keep paying off.** XSA, partial RoPE, LN scale, VE, and legal TTT all target the deepest part of the stack.
3. **Full recurrence was a bad fit for the 10-minute budget.** The candidate therefore uses a narrower version of sharing: pair-tie the tail layers only, keep early layers unique, and preserve all logical depth.

That points to a simple hypothesis: **deep layers are redundant enough to share their heavy matrices, but not their control surfaces**. If that is true, the artifact gets smaller with much less risk than a full recurrent rewrite, and some of the recovered budget can be spent on cheap token-memory components instead of more FLOPs.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` — chosen base stack: 11L, EMA, GPTQ-lite, partial RoPE, LN scale, VE, XSA4, warmdown 3500.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` — confirms partial RoPE + LN scale are worthwhile, while also documenting that one late-QAT flag was dead code.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` — motivated keeping EMA and deep-only XSA.
- `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/` and `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/` — motivated preserving SmearGate/BigramHash.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` — important negative result: naive depth recurrence needed too many steps, so this candidate avoids full recurrence.

## External research that informed it

- **ALBERT** — cross-layer parameter sharing can preserve model quality while cutting stored parameters: <https://arxiv.org/abs/1909.11942>
- **Subformer** — recent evidence that grouped/shared transformer structure can improve parameter efficiency without needing a full architecture swap: <https://arxiv.org/abs/2101.00234>
- **Compacter** — supports the broader idea of sharing a large basis and leaving only small layer-specific adaptations: <https://arxiv.org/abs/2106.04647>
- **Universal Transformer** — useful as a cautionary reference: recurrent depth reuse is attractive in theory, but this repo's own results suggest the full version is too compute-sensitive here: <https://arxiv.org/abs/1807.03819>

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Tail grouped block-core tying**
   - New defaults: `SHARE_START_LAYER=5`, `SHARE_GROUP_SIZE=2`
   - Layers `0..4` stay unique.
   - Layers `5..10` share heavy block cores in pairs.
   - Only the large matrices are shared; per-layer controls stay untied.

2. **Per-layer adapters stay unique**
   - `q_gain`
   - `attn_scale`
   - `mlp_scale`
   - `resid_mix`
   - optional DTG gate
   - skip weights and VE layer scales

3. **Modest lexical/value-memory expansion**
   - `BIGRAM_VOCAB_SIZE`: `2048 -> 4096`
   - `VE_DIM`: `128 -> 160`
   - `VE_LAYERS`: `"9,10" -> "7,8,9,10"`

4. **Safer local execution path**
   - Optional FlashAttention fallback to `scaled_dot_product_attention`
   - `SMOKE_TEST=1` path that instantiates a tiny CPU-safe model and runs a forward/backward sanity check when PyTorch is available

## How to run

From this candidate directory:

```bash
cd candidates/202604082119_tail-tied-lexical-memory

DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful candidate-specific knobs:

```bash
SHARE_START_LAYER=5
SHARE_GROUP_SIZE=2
BIGRAM_VOCAB_SIZE=4096
VE_DIM=160
VE_LAYERS=7,8,9,10
```

Optional smoke path on a machine that already has PyTorch installed:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Expected tradeoffs and risks

- **Late-layer sharing may over-constrain specialization.** If the best 11L stack depends on highly distinct deepest blocks, tying them will hurt despite smaller artifacts.
- **Recovered bytes may be better spent elsewhere.** This candidate spends them on larger BigramHash/VE memory because those are cheap in FLOPs, but wider MLPs or different quantization rules might be better.
- **Compute is not reduced.** Logical depth stays fixed at 11, so this is mainly an artifact-budget hypothesis, not a throughput optimization.
- **The smoke path is only a structural sanity check.** It does not prove GPU training quality.

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall train_gpt.py train_gpt_mlx.py data` | Passed at repo level before candidate-specific validation |
| `python -m compileall candidates/202604082119_tail-tied-lexical-memory/train_gpt.py` | Passed |
| `SMOKE_TEST=1 python candidates/202604082119_tail-tied-lexical-memory/train_gpt.py` | Not runnable in this workflow runner because PyTorch is not installed here, and a temporary CPU-wheel install was blocked by the runner's network/proxy policy |

The script still includes the smoke path so it can be run on any environment that already has PyTorch available.
