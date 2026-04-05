# Paired MLP Sharing + Partial RoPE + LN Scale

## Hypothesis

The repo's strongest training-side record pattern is now clear: move from the 9-layer baseline toward 10-11 layers, widen the MLP, and then spend the saved artifact bytes on better compression-aware tricks and evaluation. This candidate asks a different question: **can we keep the deeper 11-layer unrolled stack, preserve unique attention per layer, and reclaim bytes by sharing only adjacent MLPs?**

The bet is that the feedforward stack is more redundant than attention under this artifact budget. If that is true, then a **6-bank MLP layout for 11 blocks** should buy a better depth-per-byte tradeoff than either the root baseline or naive recurrence, especially when paired with the repo's best zero-parameter structural refinements: **Partial RoPE** and **layerwise RMS output scaling**.

## Why this is promising here

1. **Winning repo trend:** later record runs repeatedly improved by moving to 11 layers and roughly 3x MLP capacity, then polishing the compression/eval stack around that shape.
2. **Important dead-end signal:** the 1x5090 non-record sweep found that naive recurrence hurt badly when it increased compute and cut step count. This candidate avoids that failure mode by keeping the usual unrolled layer count and sharing only parameters, not adding extra recurrent passes.
3. **Byte pressure is the right bottleneck:** the non-record 4-hour run showed that extra training compute alone does not solve the post-quantization bottleneck under the 16 MB cap.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/` for the first convincing 11L + wider-MLP jump.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for **Partial RoPE (16/64)** and **LN scale**, which were the real win while Late QAT turned out to be inert.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` as the current best overall stack and as evidence that the repo's latest gains are mostly happening on top of an already-strong 11-layer architecture.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` because its recurrence ablation is a useful warning: extra depth only helps if it does not destroy step count.

There were **no prior `candidates/` directories** to avoid or supersede.

## External research that informed it

- **ALBERT** (arXiv:1909.11942) showed that cross-layer parameter sharing can sharply improve parameter efficiency without needing a full architecture rewrite.
- **Subformer** (arXiv:2101.00234) is the most relevant language-model reference here: in generative transformers, sandwich-style sharing can beat naive untied baselines with fewer parameters.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (arXiv:2505.01855) argues that reuse is most effective when targeted, with earlier layers especially worth reusing.
- **Parameter Reduction Improves Vision Transformers** (arXiv:2512.01059) found that **sharing adjacent MLP blocks** can outperform the untied baseline at the same compute, which is exactly the mechanism tested here.
- **A Survey on Transformer Compression** (arXiv:2402.05964) reinforces the broader lesson from this repo: architectural compression and quantization should be designed together, not treated as separate stages.

## Chosen base implementation

This candidate starts from the repository root `train_gpt.py`, not from the most aggressive record folders.

That is intentional: the root trainer is self-contained, avoids record-specific FlashAttention/auxiliary infrastructure, and is the cleanest place to test a new structural idea without dragging in a large unrelated feature stack. The candidate then pulls in only the record-proven architectural deltas that fit naturally into the baseline script.

## What changed vs. the base implementation

1. **Adjacent MLP sharing (`MLP_SHARE_GROUP=2`)**  
   Eleven blocks now use six unique MLP banks. Attention, norms, residual mixing, and skip wiring remain per-layer.

2. **Deeper default stack (`NUM_LAYERS=11`)**  
   This follows the repo's strongest architecture trend, but uses sharing to keep the byte growth under control.

3. **Moderately wider MLP (`MLP_MULT=2.5`)**  
   I did not set this to 3.0 because the root exporter is still the simpler int8+zlib path, not the int6/GPTQ-lite record stack. `2.5x` is the compromise that keeps the architecture pointed at the 11L/MLP3x family without blindly overspending bytes.

4. **Partial RoPE (`ROPE_DIMS=16`)**  
   Only the first 16 of 64 head dimensions receive rotary position encoding, following the strongest zero-parameter structural tweak in the records.

5. **Layerwise RMS output scaling (`LN_SCALE=1`)**  
   Each block scales its normalized inputs by `1/sqrt(layer_idx + 1)` before attention/MLP, again following the winning 2026-03-21 record.

6. **Record-style optimizer defaults**  
   Warmdown, Muon momentum, learning rates, tied-embedding LR, and grad clipping were shifted toward the better-tuned late-record defaults, but without adding new training infrastructure.

7. **Self-describing int8 export metadata**  
   The quantized export now carries the core architectural fields for this variant (`num_layers`, `mlp_share_group`, `rope_dims`, `ln_scale`, etc.) so roundtripped artifacts are less brittle than the root baseline export.

## How to run

From the repository root:

```bash
cd candidates/202604052114_paired-mlp-sharing

RUN_ID=paired_mlp_share \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to push closer to the recent long-context record regime, the first sweep I would try is:

```bash
cd candidates/202604052114_paired-mlp-sharing

RUN_ID=paired_mlp_share_seq2048 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
TRAIN_SEQ_LEN=2048 \
TRAIN_BATCH_TOKENS=786432 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still reports the baseline-style final `int8+zlib` roundtrip metrics, so this candidate is intended to measure the **training-side architecture change** first. It does **not** attempt to reproduce the repo's later sliding-window / TTT evaluation stack inside this folder.

## Expected risks and tradeoffs

- **Later layers may want unique MLPs.** Sharing only the feedforward path is milder than full block tying, but it can still over-regularize decoder-side specialization.
- **This is not the repo's strongest quantization stack.** The root exporter is intentionally left simple, so this candidate may leave some byte-budget performance on the table versus int6/GPTQ-lite record code.
- **The best setting may be asymmetric.** External work and the repo's own negative recurrence result both suggest that early layers are safer to reuse than late ones. If this candidate underperforms, the next experiment should likely share only the encoder-side or early blocks.

## Validation

| Command | Outcome |
|---|---|
| `python -m compileall candidates/202604052114_paired-mlp-sharing/train_gpt.py` | Passed. |
| CPU import/forward smoke using `python - <<'PY' ...` | Blocked in this environment because the Python runtime does not have `torch` installed. |
| Full training start from this checkout | Not feasible here because the repository checkout does not include local `data/**/*.bin` shards or a SentencePiece `.model` file. |
