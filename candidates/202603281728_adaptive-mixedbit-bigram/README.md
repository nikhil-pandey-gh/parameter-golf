# Adaptive Mixed-Bit GPTQ-lite + Bigger Bigram

## Hypothesis

The current strongest stack in this repo already has a very good training recipe; the next easy win is more likely to come from **smarter export-time compression** than from another disruptive architectural rewrite.

This candidate tests a narrow hypothesis:

- keep the current high-performing 11-layer LeakyReLU^2 + Parallel Muon + legal TTT stack,
- make the **MLP export row-adaptive** instead of fixed-bit,
- quantize most MLP rows at **int5**, but promote the rows that benefit most from extra precision to **int6**, and
- spend the recovered compression headroom on a **larger BigramHash table (3072 buckets)**.

The expected outcome is a better tradeoff between artifact entropy and lexical capacity than the fixed int6 export in the current SOTA stack.

## Why this is promising for this repository

Two repo trends are unusually consistent:

1. **Export quality matters almost as much as training quality.**
   The repo improved repeatedly through fp16 embedding passthrough, mixed precision export, and GPTQ-lite clip search.

2. **Cheap lexical bias keeps paying off.**
   SmearGate + BigramHash reappears throughout the strongest records, and the best older mixed-bit run explicitly showed that a larger bigram table helped once compression headroom existed.

This candidate combines those two trends instead of betting on a slower or riskier training-time change.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current stack in this repo,
  - already includes LeakyReLU(0.5)^2, parameter banking, Parallel Muon, partial RoPE, XSA, VE, EMA/SWA, GPTQ-lite-style export, and legal score-first TTT.

Key supporting precedents:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strong evidence that **better rowwise clip selection** gives measurable zero-training-cost gains.

- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - strong evidence that **int5 MLP weights** can compress materially better than int6 and can fund extra model capacity.

- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - evidence that SmearGate + BigramHash remain high-value in this challenge.

- `train_gpt.py` in the repository root
  - baseline reference for the repo's evaluation, optimizer split, and self-contained export conventions.

There were **no prior experiments under `candidates/`** at review time, so this candidate is informed entirely by the root baseline plus `records/`.

## External research that informed it

The guiding theme from the literature is that **quantization sensitivity is heterogeneous** and low-bit export works better when you selectively protect the most sensitive weights or channels.

- **GPTQ**: https://arxiv.org/abs/2210.17323
  - One-shot weight quantization with approximate second-order information can preserve quality surprisingly well even at very low bitwidths.

- **AWQ**: https://arxiv.org/abs/2306.00978
  - Not all weights are equally important; protecting a small salient subset can greatly reduce quantization error.

- **HAWQ-V2**: https://arxiv.org/abs/1911.03852
  - Mixed-precision quantization should follow layer sensitivity rather than using one global bitwidth.

- **AdaDim**: https://arxiv.org/abs/2309.15531
  - Weight sensitivity is structured and heterogeneous even within a layer, especially in the sub-4-bit regime.

- **BitNet b1.58**: https://arxiv.org/abs/2402.17764
  - Extreme low-bit training is increasingly plausible, which makes conservative mixed-bit export variants worth exploring in compact-model settings.

This candidate does **not** implement full AWQ/HAWQ/AdaDim calibration. Instead, it takes a lightweight repo-friendly step in that direction by using **rowwise GPTQ-lite clip search** and a **row-adaptive int5/int6 decision** for MLP matrices.

## What changed versus the chosen base implementation

Compared with `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes only a few targeted changes:

1. **Bigger BigramHash by default**
   - `BIGRAM_VOCAB_SIZE` default changed from `2048` to `3072`.

2. **Candidate defaults match the current strongest evaluation stack**
   - `TTT_ENABLED=1` by default.
   - `TTT_FREEZE_BLOCKS=0` by default.

3. **Mixed-bit GPTQ-lite for MLP rows**
   - Added `quantize_intN_gptqlite_per_row(...)` so GPTQ-lite percentile search works for both int5 and int6.
   - Added `quantize_mlp_rowmix(...)`:
     - quantize every MLP row once at int5 and once at int6,
     - compute the rowwise reconstruction gain from promotion,
     - promote only the highest-gain rows.

4. **New tuning knob for adaptive promotion**
   - `MLP_INT6_PROMOTE_FRAC` defaults to `0.0625` (top 6.25% of MLP rows, capped to rows with positive gain).

5. **Bigram weights now participate in low-bit export**
   - the export classifier now treats `bigram` tensors as a first-class quantization category and exports them with the low-bit path instead of leaving them on the generic fallback path.

6. **Clearer mixed-bit export logging**
   - export artifact renamed to `final_model.mixbits.ptz`.
   - logs now report the promoted-row fraction explicitly.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603281728_adaptive-mixedbit-bigram
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Disable legal TTT to isolate the training/export change
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Recover the old fixed-int5 MLP export for comparison
MLP_INT6_PROMOTE_FRAC=0.0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Revert the lexical expansion for a cleaner export-only ablation
BIGRAM_VOCAB_SIZE=2048 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script performs:

- training,
- EMA application,
- mixed-bit export roundtrip,
- standard and sliding-window evaluation,
- legal score-first TTT if `TTT_ENABLED=1`.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202603281728_adaptive-mixedbit-bigram/train_gpt.py
```

Outcome:

- **Passed**.

Safe CPU smoke check status:

- A direct import-run smoke test was **not feasible in this container** because both `torch` and `flash_attn_interface` are unavailable here.
- Because of that, I limited validation to syntax/bytecode compilation instead of a runtime start test.

## Main expected risks and tradeoffs

- **Weight-only, not activation-aware.**
  The row-promotion rule uses reconstruction gain on the weights themselves, not activation calibration data. That keeps the implementation small but is weaker than a true AWQ-style saliency estimate.

- **Compression savings may be smaller than hoped.**
  Since the export still stores quantized values inside int8 tensors before lzma compression, most of the gain depends on improved entropy rather than a literally packed 5-bit format.

- **Bigger bigram table could consume the saved headroom.**
  If the adaptive int5/int6 mix compresses less efficiently than expected, `BIGRAM_VOCAB_SIZE=3072` may be too aggressive and should be ablated back toward `2048`.

- **TTT can hide export-only effects.**
  Since the candidate defaults to legal TTT, export improvements should also be checked with `TTT_ENABLED=0` to separate pre-TTT and post-TTT behavior.

## Suggested next experiments if this works

- Replace the weight-only promotion heuristic with a tiny **activation-aware calibration pass**.
- Let **bigram** or **VE** tensors use their own adaptive mixed-bit rule instead of fixed low-bit export.
- Try the same adaptive row-mix on the **attention output projection** if artifact headroom is still tight.
- If export-side gains saturate, the next large unexplored branch is **shared-depth / recurrent block reuse** rather than yet another optimizer tweak.
