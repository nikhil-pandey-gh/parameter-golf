# Candidate: Paired Late Block Sharing + fp16 Embedding

## Hypothesis

The current 11-layer EMA/XSA/partial-RoPE family appears quantization-limited more than pre-quantization-limited. If we share only the heavy **late decoder attention/MLP weights** while keeping each logical layer's norms, residual scales, skip structure, XSA placement, and value-embedding scales unique, we can recover enough artifact budget to keep the tied token embedding in `fp16`.

That trade should be stronger than naive recurrence for this repo: it preserves the same forward depth and step count, but spends bytes where prior runs say they matter most.

## Why this is promising for this repository

Two repo-wide trends point in the same direction:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` showed the tied embedding is unusually quantization-sensitive because it doubles as the output head.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` and `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` both reported that **extra recurrent depth / looped layers were net negative** under the 10-minute budget because they reduced step count too much.

So this candidate uses **same-depth parameter sharing**, not extra passes: it tries to buy back artifact bytes without paying additional wall-clock compute.

## Prior records that influenced this candidate

This candidate is built directly on the strongest self-contained non-TTT stack in the repo:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`

What I kept from that line:

- 11-layer, 512-dim, GQA, 3x MLP, SmearGate, BigramHash, shared value embeddings
- XSA on the last 4 logical layers
- partial RoPE + LN scaling
- EMA-based evaluation/export path
- GPTQ-lite style mixed int6 export with sliding-window evaluation

What I changed is the parameter allocation strategy.

## External research that informed it

- **ALBERT** (`arXiv:1909.11942`) showed that cross-layer parameter sharing can cut memory substantially while preserving much of the model's capacity.
- **Universal Transformer** (`arXiv:1807.03819`) motivates depth reuse, but the repo's own negative results suggest using that idea here only in a compute-neutral form.
- **MobileLLM** (`arXiv:2402.14905`) is the most directly relevant: it reports that deep-and-thin small LMs benefit from grouped-query attention, embedding sharing, and an **immediate block-wise weight-sharing** variant with only marginal latency overhead.

This candidate adapts that MobileLLM-style block sharing to the repo's existing 11-layer U-Net/XSA stack.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Split each transformer block into a shared heavy core and a per-layer wrapper.**
   - `BlockCore` owns the attention and MLP weights.
   - `LayerBlock` owns per-layer RMSNorm usage, residual mix, XSA flag, and learned scales.

2. **Share the last 4 logical layers as a 2-block cycle by default.**
   - Default settings: `SHARED_TAIL_START=7`, `SHARED_TAIL_CYCLE=2`
   - Logical layers `7` and `9` share one core.
   - Logical layers `8` and `10` share the other core.
   - Earlier layers remain unique.

3. **Keep the tied token embedding in `fp16` during export.**
   - Default: `FP16_EMBED=1`
   - This directly targets the quantization-sensitive tensor highlighted by the earlier fp16-embedding record.

4. **Add a non-FlashAttention fallback for model import/smoke use.**
   - If `flash_attn_interface` is unavailable, the attention path falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - This is not for leaderboard speed; it just makes the candidate easier to instantiate outside the exact H100 environment.

5. **Disable late-QAT by default.**
   - Default: `LATE_QAT_THRESHOLD=0.0`
   - Earlier repo notes already showed compile-time fragility around the late-QAT flag, so this candidate keeps the hypothesis focused on sharing + export precision instead of a potentially dead code path.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603290726_tailshare-fp16-embed
RUN_ID=tailshare_fp16 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
FP16_EMBED=1 SHARED_TAIL_START=7 SHARED_TAIL_CYCLE=2 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate script is self-contained and keeps the training/eval/export flow inside the candidate directory.

## Validation

Commands run in this workflow:

```bash
python -m compileall candidates/202603290726_tailshare-fp16-embed/train_gpt.py
```

Outcome: **passed**.

Attempted smoke validation:

```bash
python - <<'PY'
# import candidate script and run a tiny CPU forward pass
PY
```

Outcome: **not feasible on this runner** because the local Python environment does not have the repo's required dependencies installed (`torch`, `numpy`, and `sentencepiece` were all missing). The candidate script now includes a non-FlashAttention fallback so this smoke path should work once the normal repo dependencies are installed.

## Main expected risks / tradeoffs

- **Late-layer sharing may overcompress the exact layers that currently carry XSA and value-embedding refinements.** If the unique late stack matters more than the fp16 embedding gain, this could regress.
- **Artifact savings may be smaller than expected after zstd compression.** Shared structure does not always translate linearly into compressed-byte savings.
- **The best sharing pattern is not obvious.** `7/9` and `8/10` is the first strong guess, but `6/8/10`-style or `shared_tail_cycle=1` could still be better or worse.
- **Disabling late-QAT removes one optimization knob.** That is intentional here for correctness/simplicity, but it means this candidate is not exhaustively stacked.

## Suggested next experiments

- Sweep `SHARED_TAIL_START` and `SHARED_TAIL_CYCLE` to test `6..10` sharing versus only `7..10`.
- Spend any additional artifact headroom on a larger `BIGRAM_VOCAB_SIZE` or a slightly larger `VE_DIM`.
- Try reintroducing a verified QAT path only after confirming the sharing/fp16-embed trade by itself.
