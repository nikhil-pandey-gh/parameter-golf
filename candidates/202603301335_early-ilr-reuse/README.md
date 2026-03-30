# Candidate: Early-layer gated ILR reuse

## Hypothesis

Add one **gated extra shared-weight pass** to the earliest transformer layers in the current best stack. The hope is to buy a bit of extra effective depth and token-mixing capacity for almost no artifact bytes, without paying the full wall-clock penalty of naively looping the entire network.

Concretely, this candidate reuses the existing block weights for layers `0` and `1`, runs one extra intra-layer pass (`RECURRENT_STEPS=1`), and learns small per-dimension control vectors that gate how much of that second pass is actually used.

## Why this is promising for this repository

The repo's strongest runs have converged on a fairly stable recipe: 11 layers, 3x MLP, XSA on deep layers, Partial RoPE, LN scaling, value embeddings, mixed low-bit export, and better averaging/eval. At this point, most obvious "free" wins have already been harvested.

The remaining opening is to get **more effective computation per parameter byte**. Recurrent depth and parameter sharing are a natural fit for Parameter Golf because they add compute faster than they add artifact size.

This repo *has* one negative recurrence result already: `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` shows that a naive "layer recurrence x2" was disastrous on a small 5090 baseline. This candidate is intentionally different in three ways:

1. It does **not** double the whole network.
2. It applies recurrence only to the **first two layers**, where the compute overhead is much smaller.
3. The extra pass is **gated and initialized conservatively** (`RECURRENT_GATE_INIT=-2.0`), so training can learn to keep or discard the reused computation instead of being forced to use it from step 1.

## Prior records and history that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Highest-scoring current stack (`1.1194` mean).
  - This candidate copies that implementation as the direct base so the new idea is tested on the strongest known recipe, not on an outdated baseline.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Strongest pre-TTT stack and a good sanity reference for the architecture family.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Reinforced that Partial RoPE + LN scaling are high-leverage, zero-byte-ish improvements worth preserving.

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - Important counterexample: naive full recurrence was bad. This candidate exists specifically as a narrower, more compute-aware twist on that dead end.

- Prior `candidates/`
  - None existed in the repository when this candidate was created.

## External research that informed it

- **Intra-Layer Recurrence in Transformers for Language Modeling** (Nguyen and Lin, 2025), arXiv:2505.01855  
  <https://arxiv.org/abs/2505.01855>
  - Motivated the idea of applying recurrence *selectively* inside layers instead of bluntly looping large chunks of the model.
  - The paper reports that targeted recurrence can be effective and that earlier-layer allocation can be especially promising.

- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** (Lan et al., 2019), arXiv:1909.11942  
  <https://arxiv.org/abs/1909.11942>
  - Classic motivation for cross-layer sharing / parameter reuse when parameter budget is the bottleneck.

- **Scaling Law for Quantization-Aware Training** (Chen et al., 2025), arXiv:2505.14302  
  <https://arxiv.org/abs/2505.14302>
  - Reinforced the need to treat quantization-sensitive pieces carefully.
  - In this candidate, the new recurrence controls stay as small passthrough control tensors rather than being aggressively low-bit quantized.

## What changed versus the chosen base implementation

Base file: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes made in this candidate:

1. Added repo-root-relative defaults for `DATA_PATH` and `TOKENIZER_PATH`, so `train_gpt.py` can be run directly from this candidate directory without having to `cd` back to repo root.

2. Added three new hyperparameters:
   - `RECURRENT_LAYERS` (default: `"0,1"`)
   - `RECURRENT_STEPS` (default: `1`)
   - `RECURRENT_GATE_INIT` (default: `-2.0`)

3. Extended `Block` with a small **intra-layer recurrence path**:
   - reuse the same attention/MLP weights,
   - learn per-dimension input/skip/attn/mlp scales,
   - apply a learned sigmoid gate to the extra pass.

4. Wired recurrence into `GPT` so only the configured early layers get the extra pass.

5. Logged the active recurrent layers in training output for easier ablation/debugging.

Everything else from the top record stack stays intact: parameter banking, Parallel Muon, LeakyReLU(0.5)^2, XSA on later layers, Partial RoPE, LN scale, value embeddings, GPTQ-lite-style export path, and legal score-first TTT.

## How to run or evaluate it

Run from this candidate directory:

```bash
cd candidates/202603301335_early-ilr-reuse

RUN_ID=early_ilr_reuse \
BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
RECURRENT_LAYERS=0,1 RECURRENT_STEPS=1 RECURRENT_GATE_INIT=-2.0 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The script now resolves dataset/tokenizer defaults relative to the repo root, so you do not need to set `DATA_PATH` or `TOKENIZER_PATH` when launching from this folder.
- Like the inherited base script, it still assumes `WORLD_SIZE` divides `8`.
- A clean ablation is to compare:
  - `RECURRENT_LAYERS=` (disabled)
  - `RECURRENT_LAYERS=0`
  - `RECURRENT_LAYERS=0,1`
  - `RECURRENT_LAYERS=0,1 RECURRENT_STEPS=2`

## Main expected risks and tradeoffs

- **Step-count risk**: even a small amount of recurrence reduces throughput. If the gained effective depth is too small, this can replay the older recurrence failure in milder form.

- **Compile risk**: the inherited script uses `torch.compile(fullgraph=True)`. The recurrence loop is static, but it still adds graph complexity and needs real GPU validation.

- **Interaction risk with TTT**: the base stack already contains legal TTT. If recurrence changes feature geometry, it could help or hurt downstream TTT adaptation.

- **Quantization sensitivity**: the new low-dimensional recurrence controls are kept as passthrough tensors, but the extra computation may still shift the weight distribution enough to change the post-quantization gap.

## Validation

Commands run in this workflow:

1. Syntax validation

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603301335_early-ilr-reuse/train_gpt.py
```

Outcome: **passed**.

2. CPU smoke attempt

I attempted a tiny CPU-only import/forward smoke test with a temporary `flash_attn_interface` stub, but the runner does not have `torch` installed in either `/usr/bin/python` or `/usr/bin/python3`, so runtime smoke validation was **not feasible in this environment**.

Observed limitation:

```text
ModuleNotFoundError: No module named 'torch'
```

That means this candidate has been syntax-checked locally, but still needs a real PyTorch + FlashAttention environment for end-to-end runtime confirmation.
