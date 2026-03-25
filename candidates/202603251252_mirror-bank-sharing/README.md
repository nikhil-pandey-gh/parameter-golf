# Mirror-Tied Bank Sharing

## Hypothesis

Mirror-tie only the **heavy banked weights** across the U-Net-symmetric encoder/decoder layers of the current best stack, while keeping the per-layer control surfaces untied (`RMSNorm`, layer scales, residual mixes, XSA flags, VE layer scales, and TTT behavior). This should preserve most of the effective-depth behavior of the 11-layer record while reducing counted parameters enough to safely **reinvest bytes into a larger `BigramHash` table**.

Concretely, this candidate maps the 11 effective layers onto 6 unique bank slots with the mirror schedule `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]`, so the shared bank weights follow the existing U-Net geometry instead of naively looping the whole model more times.

## Why it is promising for this repository

This repository's leaderboard is already heavily optimized along the obvious axes: sliding-window eval, mixed quantization, wider MLPs, SmearGate/BigramHash, XSA, EMA/SWA, partial RoPE, GPTQ-lite, and legal TTT. That makes **parameter sharing without extra effective depth** one of the cleaner remaining directions.

The repo history also points to the exact twist to try. `records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py` carried dormant `NUM_LOOPS` / LoRA recurrence support, while the 1×5090 non-record sweep reported naive layer recurrence as a negative result because it **increased compute and reduced steps**. This candidate avoids that failure mode by keeping the effective depth at 11 and sharing only the heavy matrices.

## Prior records and candidates that influenced it

There were **no prior `candidates/` directories** in this checkout.

The main implementation base is the current best record:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

That record contributes the strongest current stack: parameter banking + parallel Muon, LeakyReLU(0.5)^2, XSA on late layers, partial RoPE, VE, EMA, and legal score-first TTT.

Additional repo influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` for the cleanest pre-TTT 11-layer training stack.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval/train_gpt.py` for the earlier looped/shared-depth exploration hooks.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` for the negative result warning on naive recurrence.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md` and the 2026-03-23 record ablation for evidence that a larger BigramHash table is a sensible reinvestment target.

## External research that informed it

The core idea is borrowed from weight-sharing / recurrent-depth work showing that **sharing the large transformation weights while leaving smaller per-depth controls untied** can preserve much more quality than fully collapsing the whole layer stack:

- **ALBERT** — cross-layer parameter sharing for Transformer compression: <https://arxiv.org/abs/1909.11942>
- **Universal Transformer** — iterative refinement with shared transition functions: <https://arxiv.org/abs/1807.03819>
- **Block-Recurrent Transformer** — recurrent depth as a language-model scaling tool: <https://arxiv.org/abs/2203.07852>
- **MoEUT** — newer evidence that shared iterative Transformer structure can still be competitive when done carefully: <https://arxiv.org/abs/2405.16039>

I considered LSQ-style QAT and newer QAT papers as well, but the repo already spends a lot of complexity budget on compression-aware training. Mirror-tied bank sharing felt like the stronger “open” axis here because it is both underexplored in this repo and compatible with the current best systems stack.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `MIRROR_TIED_BANKS` (default `1`).
2. Changed the banked weight tensors so they store only the unique mirror-tied bank slots instead of one heavy slice per effective layer.
3. Kept all 11 `Block` modules and all per-layer control parameters untied.
4. Updated the bank export / unbank / re-bank path so quantization operates on the **unique bank slots** instead of silently duplicating shared layers during serialization.
5. Increased the default `BIGRAM_VOCAB_SIZE` to `3072` as a low-compute reinvestment of the recovered bytes.
6. Made the candidate runnable from **inside its own directory** by resolving default `DATA_PATH` and `TOKENIZER_PATH` relative to `__file__` rather than the current working directory.
7. Added a **FlashAttention fallback** to `scaled_dot_product_attention`, which makes local import / CPU smoke paths possible when `flash_attn_interface` is unavailable.
8. Baked in the current best record's TTT defaults more directly by setting `TTT_ENABLED=1` and `TTT_FREEZE_BLOCKS=0` as candidate defaults. Set `TTT_ENABLED=0` if you want to compare against the pre-TTT roundtrip/sliding metrics only.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603251252_mirror-bank-sharing
RUN_ID=mirror_bank_trial \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
# Compare against the untied-bank record-style baseline inside the same script.
MIRROR_TIED_BANKS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Disable legal TTT to inspect only the training-time / quantized model quality.
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Because the script resolves repo-root-relative defaults from `__file__`, it can be launched directly from this directory without manually setting `DATA_PATH` or `TOKENIZER_PATH`, as long as the standard repository dataset/tokenizer layout is present.

## Main expected risks or tradeoffs

- **Capacity risk:** tying the heavy banks across mirror layers may over-regularize the model and erase useful encoder/decoder asymmetry.
- **Quantization interaction risk:** quantization now acts on the shared bank slots directly, which preserves the byte savings but also means each mirror pair must agree on one quantized representation.
- **Systems risk:** this keeps the parameter-banked / parallel-Muon implementation, so the compile + distributed overlap path remains relatively complex.
- **Evaluation-time cost:** legal TTT is still expensive; this candidate intentionally keeps that path because it is part of the current best stack.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202603251252_mirror-bank-sharing/train_gpt.py
```

Outcome: **passed**.

I also attempted a minimal CPU-only import/forward smoke check, but it was **not feasible in this runner** because the environment does not currently have `torch` installed (`requirements.txt` declares it, but the local Python environment here does not provide it). The candidate therefore received a full syntax check plus manual code review of the shared-bank and export paths, but not a live forward-pass smoke test in this environment.
