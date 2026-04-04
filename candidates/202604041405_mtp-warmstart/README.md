# Warm-Started MTP on the 2026-03-23 record stack

## Hypothesis

A small **training-only multi-token prediction (MTP)** auxiliary head should improve sample efficiency inside the fixed 600-second training budget, and **warm-starting** that head from the main token interface should make the auxiliary loss useful immediately instead of spending early steps relearning vocabulary geometry.

This is attractive for Parameter Golf because the current best stack already carries dormant MTP support, and the export path already drops `mtp_heads`, so the idea targets training efficiency **without increasing artifact bytes**.

## Why this is promising here

- The current best record is the 2026-03-23 **LeakyReLU + legal score-first TTT + Parallel Muon** stack, so it is the strongest base to inherit.
- Recent external work on MTP reports better sample efficiency from predicting multiple future tokens with auxiliary heads on top of a shared trunk:
  - Fabian Gloeckle et al., **Better & Faster Large Language Models via Multi-token Prediction** — <https://arxiv.org/abs/2404.19737>
- Unlike AWQ/OmniQuant/SliderQuant-style ideas, this candidate does not need a new calibration/export pipeline.
- Unlike recurrence or layer looping, this keeps the high-performing 11-layer stack intact and only changes the training objective.

## Prior repository work that influenced this candidate

- **`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`**  
  Best current stack and direct code base: LeakyReLU(0.5)^2, legal TTT, parameter banking, Parallel Muon, EMA/SWA, GPTQ-lite int6, lzma export.
- **`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`**  
  Confirms the value of the current 11-layer GPTQ-lite/EMA quantization path that this candidate keeps unchanged.
- **`records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`**  
  Explicitly notes that the late-QAT toggle was compiled away in that lineage, so this candidate disables late QAT by default rather than relying on a known-fragile branch.

## What changed versus the chosen base implementation

1. **Enabled a training-only MTP head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`
2. **Warm-started MTP head weights from the main token interface**
   - `MTP_INIT_FROM_MAIN=1`
   - The auxiliary head copies `tok_emb.weight` so it starts from the model's existing token geometry instead of a zero-init head
3. **Conservative multi-horizon weighting**
   - If more than one MTP head is enabled, farther future heads are downweighted by `1 / (k + 1)`
4. **Disabled late QAT by default**
   - `LATE_QAT_THRESHOLD=0.0`
   - This keeps the candidate focused on MTP instead of a compile-fragile quantization toggle
5. **Added a PyTorch SDPA fallback for attention**
   - Keeps FlashAttention 3 when available on CUDA
   - Allows importability and CPU-side smoke checks when `flash_attn_interface` is unavailable

The export path is still unchanged in the important way: `mtp_heads` are excluded from the serialized artifact and the eval model is rebuilt with `mtp_num_heads=0`.

## How to run / evaluate

Run from this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 MTP_INIT_FROM_MAIN=1 \
LATE_QAT_THRESHOLD=0.0 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run for this candidate in the workflow environment:

```bash
python -m compileall train_gpt.py
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location(
    "candidate_train_gpt",
    "train_gpt.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
```

Outcomes:

- `python -m compileall ...` **passed**
- The CPU import/forward smoke path was **not feasible in this workflow environment** because the available Python interpreter does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`)

## Main expected risks / tradeoffs

- The extra MTP logits may reduce steps enough to erase the sample-efficiency gain.
- Warm-starting from the main token interface may bias the auxiliary head toward short-range lexical reuse rather than genuinely helpful lookahead.
- The gain may appear in pre-TTT or pre-export metrics but wash out after legal TTT and int6 roundtrip evaluation.
- If the best setting wants more than one MTP head, the compute overhead may grow faster than the quality gain under the fixed wall-clock cap.
