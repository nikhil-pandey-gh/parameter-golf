# Shared Upper MLP + Low-Rank Deltas

## Hypothesis

The strongest recent Parameter Golf models are all converging on a very similar 11-layer recipe: XSA in the late stack, larger MLPs, EMA/SWA smoothing, partial RoPE, quantization-aware export, and richer front-end features like BigramHash and value embeddings. That suggests some of the *late-layer MLP capacity is redundant in bytes even if it is still useful in compute*.

This candidate tests an ALBERT-style compromise: **share the upper MLP bank across the top 4 layers, then reintroduce per-layer specialization with tiny rank-8 low-rank deltas**. The saved artifact budget is then spent first on a repo-proven lever:

- a larger `BigramHash` table (`4096` buckets by default).

The script also leaves an explicit `FP16_EXPORT_NAME_PATTERNS` hook for future follow-up sweeps on quantization-sensitive tensors, but that is **not** the default behavior of this candidate.

## Why this is promising for this repository

This repo's records show a very consistent pattern:

- more model quality usually came from **freeing bytes to buy better capacity**, not from shrinking the model outright,
- deeper/wider MLP-heavy stacks keep winning when compression can pay for them,
- the tied embedding is unusually sensitive to quantization,
- and simple full depth recurrence was a negative result under the 10-minute budget.

This candidate tries to exploit the same trend without repeating the failed recurrence idea. It **does not add virtual depth or extra recurrent passes**. Instead, it shares only the late MLP weights and keeps compute roughly aligned with the current 11-layer stack, while recovering per-layer flexibility with small deltas.

## Prior records that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Most relevant influences:

- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: best overall stack to date; contributed parameter banking, LeakyReLU^2 MLPs, legal score-first TTT, VE layers, XSA, and the export/eval path.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: reinforced the value of better post-training quantization and EMA smoothing.
- `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`: showed that spending bytes on a larger `BigramHash` table can still improve bpb when the artifact budget allows it.
- `2026-03-18_FP16Embed_WD3600`: showed that the tied embedding is one of the most quantization-sensitive tensors in the model and is a strong follow-up target once the sharing tradeoff is measured.
- `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090` and `2026-03-18_FP16Embed_WD3600`: both documented that naive recurrence or overly expensive structural changes can lose under a strict wallclock budget. This candidate avoids extra depth passes entirely.

There were no pre-existing experiments under `candidates/` in this checkout.

## External research that informed it

- **ALBERT**: cross-layer parameter sharing can reduce parameter count substantially while preserving useful depth when the model still executes the same sequence of layers. <https://arxiv.org/abs/1909.11942>
- **LoRA**: low-rank updates are an efficient way to recover layer-specific behavior without paying for full dense matrices. <https://arxiv.org/abs/2106.09685>
- **Hyper-Connections**: recent work argues that depth-wise flexibility matters; that motivated avoiding hard tying without any per-layer adaptation. <https://arxiv.org/abs/2409.19606>

I also considered LayerDrop, ALiBi hybrids, and multi-token prediction based on current literature, but chose shared upper MLPs because it best matched this repo's strongest empirical trend: conserve artifact bytes, keep training compute close to the known-good stack, and reinvest the savings in proven components.

## What changed versus the chosen base implementation

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate:

- shares the **upper MLP bank** across the final `SHARED_MLP_UPPER_LAYERS` layers (default `4`),
- adds a tiny `SharedMLPDelta` module per shared upper layer with rank `SHARED_MLP_DELTA_RANK` (default `8`),
- increases the default `BIGRAM_VOCAB_SIZE` from `2048` to `4096`,
- adds a FlashAttention fallback path so the script can execute a local CPU smoke test without `flash_attn_interface`, and
- adds `SMOKE_TEST=1` mode for a one-step synthetic forward/backward startup check.

Everything else stays intentionally close to the best existing stack: parameter banking, Parallel Muon, late XSA, value embeddings, partial RoPE, LeakyReLU^2 MLP nonlinearity, lzma-compressed int6 export, and optional legal TTT.

## How to run or evaluate it

### Main 8xH100-style run

From the candidate directory:

```bash
cd candidates/202604010035_shared-upper-mlp

RUN_ID=shared_upper_mlp \
SEED=1337 \
SHARED_MLP_UPPER_LAYERS=4 \
SHARED_MLP_DELTA_RANK=8 \
BIGRAM_VOCAB_SIZE=4096 \
TTT_ENABLED=1 \
TTT_FREEZE_BLOCKS=0 \
TTT_EPOCHS=3 \
TTT_LR=0.002 \
TTT_CHUNK_TOKENS=32768 \
TTT_BATCH_SEQS=32 \
TTT_GRAD_CLIP=1.0 \
python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

If you want to isolate the training-time architecture change before paying the legal TTT cost, start with `TTT_ENABLED=0`.

### Minimal local smoke test

This smoke path does **not** require the real dataset or tokenizer and is intended only to verify that the candidate starts, builds the shared-MLP model, runs a forward/backward pass, and updates weights:

```bash
SMOKE_TEST=1 \
NUM_LAYERS=3 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2.0 VOCAB_SIZE=64 BIGRAM_VOCAB_SIZE=64 \
VE_ENABLED=0 XSA_LAST_N=1 \
SHARED_MLP_UPPER_LAYERS=2 SHARED_MLP_DELTA_RANK=4 \
python train_gpt.py
```

## Validation run in this workflow

I ran the following validations:

```bash
python -m compileall candidates/202604010035_shared-upper-mlp/train_gpt.py
```

Outcome: passed.

```bash
SMOKE_TEST=1 \
NUM_LAYERS=3 MODEL_DIM=64 NUM_HEADS=4 NUM_KV_HEADS=2 \
MLP_MULT=2.0 VOCAB_SIZE=64 BIGRAM_VOCAB_SIZE=64 \
VE_ENABLED=0 XSA_LAST_N=1 \
SHARED_MLP_UPPER_LAYERS=2 SHARED_MLP_DELTA_RANK=4 \
python candidates/202604010035_shared-upper-mlp/train_gpt.py
```

Outcome: passed in a local CPU-only startup check with `smoke_test:ok loss:4.166059 device:cpu`.

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

p = Path("candidates/202604010035_shared-upper-mlp/train_gpt.py")
spec = importlib.util.spec_from_file_location("cand_train_gpt", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

state = {
    "mlp_up_bank": torch.randn(3, 192, 384),
    "qo_bank": torch.randn(4, 192, 192),
    "tok_emb.weight": torch.randn(64, 64),
}
q, meta = mod.mixed_quantize_int6(state, {"mlp", "attn"})
restored = mod.dequantize_mixed_int6(q, meta, state)
print(tuple(q["mlp_up_bank.scale"].shape), tuple(restored["mlp_up_bank"].shape))
print(tuple(q["qo_bank.scale"].shape), tuple(restored["qo_bank"].shape))
PY
```

Outcome: passed. This specifically exercises the bank-preserving int6 export path on oversized 3-D bank tensors and confirmed that the restored bank shapes matched the originals after quantize/dequantize round-trip.

Repository baseline syntax check also passed before the candidate change via:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
```

## Main expected risks and tradeoffs

- **Too much sharing could blur late-layer specialization.** If the top 4 layers are not redundant enough, the low-rank deltas may be too weak to recover lost quality.
- **The byte trade may still be misallocated.** `BigramHash=4096` is a reasonable first place to spend the recovered budget, but a different rank/hash trade or a selective fp16 export sweep may still be better.
- **The best share extent is unknown.** `SHARED_MLP_UPPER_LAYERS=2`, `3`, and `4`, plus delta ranks `4`, `8`, and `16`, are all worth sweeping.
- **Quantized artifact size still needs a real GPU run.** The code now preserves the intended sharing at export time, but final compressed size and bpb need a full dataset-backed run to confirm the trade lands under the 16MB cap with a net win.

## Suggested next experiments

1. Run a no-TTT ablation first (`TTT_ENABLED=0`) to measure the pure training/export effect.
2. Sweep `SHARED_MLP_UPPER_LAYERS in {2,3,4}`.
3. Sweep `SHARED_MLP_DELTA_RANK in {4,8,16}`.
4. Compare `BIGRAM_VOCAB_SIZE=2048` vs `4096` under the new sharing regime.
5. Use `FP16_EXPORT_NAME_PATTERNS` to test whether selectively preserving `tok_emb.weight` or a tiny set of late tensors is a better use of the recovered artifact budget than a larger hash table alone.
