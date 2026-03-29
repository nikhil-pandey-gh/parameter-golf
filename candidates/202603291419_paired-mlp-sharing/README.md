# Paired MLP Sharing + FP16 Tied Embedding Budget Reuse

## Hypothesis

The current strong 11-layer stack appears limited more by post-quantization artifact quality than by raw pre-quant model loss. This candidate tests whether **sharing only the heaviest middle-layer MLP weights across mirrored U-Net positions** can reclaim enough artifact budget to restore **fp16 tied-embedding export** and expand the **BigramHash** side channel, while keeping per-step compute almost unchanged.

The key bet is that middle MLPs are more redundant than late attention layers, so sharing them is a lower-risk way to buy back bytes than repeating full-block recurrence. The recovered bytes are then spent on components that prior records say matter most after quantization.

## Why this is promising for this repository

Three repository trends point in the same direction:

1. `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` found that keeping `tok_emb.weight` in fp16 nearly eliminated the quantization gap, but the extra artifact bytes forced a smaller MLP.
2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` reports that increasing `BIGRAM_VOCAB_SIZE` from 2048 to 3072 helped on the strongest recent stack.
3. `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` explicitly says full layer recurrence x2 was a dead end because it cut too many training steps in the fixed wall-clock budget.

That suggests a good next candidate should **reuse bytes, not compute**: preserve the fast 11-layer schedule, avoid more recurrent depth, and move artifact budget toward embedding/export quality and lexical side channels.

## Prior experiments that influenced this candidate

Base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Most relevant influences:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`

## External research that informed it

- **ALBERT** (`arXiv:1909.11942`) showed that cross-layer parameter sharing can substantially reduce parameter count while retaining strong performance.
- **Subformer: Exploring Weight Sharing for Parameter Efficiency in Generative Transformers** (arXiv search result, 2021) specifically argues for sandwich-style sharing in generative transformers rather than broad architectural replacement.
- **ShishuLM** (`arXiv:2510.13860`) argues that small language models contain architectural redundancy and that replacing or simplifying some transformer substructure can cut memory/latency without destroying language-model quality.

This candidate does not copy those papers directly. Instead, it adapts the common lesson to this codebase's constraint profile: share only the large middle MLP weights, keep the late attention-heavy stack intact, and cash the saved bytes into more quantization-sensitive tensors.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate adds four focused changes:

1. **Mirrored middle-layer MLP sharing** via `SHARED_MLP_GROUPS`, defaulting to `3,6;4,5`.
   - Only MLP weights are shared.
   - Attention, norms, residual scales, skip weights, and late-layer XSA behavior remain layer-specific.
   - Export strips alias keys so sharing reduces artifact bytes, not just in-memory parameters.

2. **FP16 tied-embedding export** by default.
   - `FP16_EMBED_EXPORT=1` keeps `tok_emb.weight` as fp16 during mixed quantized export.
   - This directly targets the repository's most repeatable post-quant sensitivity.

3. **Larger BigramHash table** by default.
   - `BIGRAM_VOCAB_SIZE` increases from `2048` to `3072`.
   - The intent is to reuse some of the saved artifact budget on a lexical side channel that already looked promising on the latest strong stack.

4. **Portability improvements for iteration**.
   - FlashAttention import now falls back to PyTorch scaled-dot-product attention when unavailable.
   - `SMOKE_ONLY=1` runs a tiny CPU roundtrip check (model forward + quantize/dequantize + logits shape) when the Python deps are installed.

## How to run or evaluate it

From this candidate directory:

```bash
# Standard 8xH100-style run
RUN_ID=paired_mlp_share \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Important defaults in this candidate:

```bash
BIGRAM_VOCAB_SIZE=3072
SHARED_MLP_GROUPS="3,6;4,5"
FP16_EMBED_EXPORT=1
```

Optional smoke check for local iteration when dependencies are installed:

```bash
SMOKE_ONLY=1 python train_gpt.py
```

## Main risks and tradeoffs

- **Reduced middle-layer specialization**: sharing MLPs may hurt the encoder/decoder halves if the mirrored layers are less redundant than expected.
- **Artifact savings may be smaller than hoped**: fp16 tied embeddings spend bytes quickly, so the net budget win depends on how well the deduplicated shared MLP export compresses.
- **Late-stack quality still dominates**: because XSA and value embeddings remain unshared, this candidate is conservative. It may be too conservative to move the score if most of the remaining gain is in optimization or evaluation rather than artifact allocation.
- **Fallback attention is slower**: the PyTorch SDP fallback is for portability and smoke testing, not for leaderboard-speed training.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603291419_paired-mlp-sharing/train_gpt.py
python -c "import torch"  # to determine whether CPU smoke was feasible here
```

Outcomes:

- `compileall`: passed for `train_gpt.py`, `train_gpt_mlx.py`, `data/`, and `candidates/202603291419_paired-mlp-sharing/train_gpt.py`.
- CPU smoke: not run in this workflow because `python -c "import torch"` failed with `ModuleNotFoundError: No module named 'torch'`, so the required runtime dependency was not available here.
