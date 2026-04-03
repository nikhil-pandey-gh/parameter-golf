# QDrop-style bank late-QAT on the March 23 stack

## Hypothesis

The current strongest stack in this repository already trains the large attention and MLP weights inside four parameter banks, but its late-QAT path only touches `CastedLinear` modules. That means the artifact-dominating bank tensors are still trained fully in float and only quantized at export time. This candidate adds **bank-aware stochastic late-QAT** so those banked weights see int6-like noise during the last part of training, with a **QDrop-style probability ramp** instead of an abrupt on/off switch.

If this works, the final int6+lzma artifact should quantize more cleanly and improve validation BPB **without adding parameters or bytes**.

## Why this is promising here

This repo's best results are already heavily compression-aware: int6/int8 mixed export, EMA/SWA smoothing, GPTQ-lite clip search, and architecture choices that spend saved bytes on more useful capacity. The remaining gap is that the strongest March 23 record moved most heavy weights into banks for speed, but that also left late-QAT coverage incomplete. This candidate targets exactly that repo-specific gap instead of introducing a broader architectural rewrite.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - strongest current stack
   - introduced parameter banks, parallel Muon, LeakyReLU(0.5)^2, and legal score-first TTT
2. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - best pre-March-23 export recipe
   - kept GPTQ-lite percentile clip search for final artifact quantization
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - documented that an earlier late-QAT path could be neutralized by `torch.compile`
   - reinforced the need for a compile-safe or eager late-quant phase

## External research

This candidate is mainly motivated by:

1. **QDrop** — *Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization* (arXiv:2203.05740). The key takeaway is that mixing quantized and full-precision paths during low-bit adaptation improves flatness and robustness.
2. **LSQ** — *Learned Step Size Quantization* (arXiv:1902.08153). The relevant lesson here is that exposing the model to the quantizer during training matters, especially at low precision.
3. **GPTQ** — *Accurate Post-Training Quantization for Generative Pre-trained Transformers* (arXiv:2210.17323). This remains the right export-time reference point for transformer weight compression, so the candidate keeps the repo's GPTQ-lite end-stage quantizer.
4. **BRECQ / AdaRound** (arXiv:2102.05426, arXiv:2004.10568). These reinforce the same direction: low-bit success depends on adapting weights to the eventual quantizer rather than only rounding at the end.

## What changed vs the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added **bank-aware fake quantization** for:
   - `qo_bank`
   - `kv_bank`
   - `mlp_up_bank`
   - `mlp_down_bank`
2. Added a **QDrop-style probability ramp**:
   - `BANK_QAT_START` (default `0.20` LR scale): start mixing quantized rows into training
   - `BANK_QAT_FULL` (default `0.05` LR scale): reach fully quantized bank forwards
   - legacy `LATE_QAT_THRESHOLD` still controls when the smaller `CastedLinear` modules start fake-quantizing
3. Switched the late bank-QAT phase to **eager mode** once it activates, so the repo does not depend on `torch.compile` observing mutable QAT state correctly.
4. Kept the final artifact path unchanged:
   - unbank weights
   - run the existing GPTQ-lite percentile-search int6 quantizer
   - compress with lzma
5. Added a **FlashAttention fallback** to `scaled_dot_product_attention`, which makes import-time / CPU smoke tests possible once `torch` is available.

The training-time fake quantizer intentionally stays cheap: row-wise absmax int6 rather than per-step percentile search. The expensive GPTQ-lite clip search still happens only at export time.

## How to run

From the repo root:

```bash
cd candidates/202604031746_qdrop-bank-qat
BIGRAM_VOCAB_SIZE=1536 TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 \
BANK_QAT_ENABLED=1 BANK_QAT_START=0.20 BANK_QAT_FULL=0.05 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate resolves its default dataset and tokenizer paths relative to the repository root, so it can be launched directly from this directory without rewriting `DATA_PATH` or `TOKENIZER_PATH`.

The candidate still supports the original env-var interface, so `BANK_QAT_START` / `BANK_QAT_FULL` can be swept independently of `LATE_QAT_THRESHOLD` without editing code.

## Validation

Commands run on this runner:

1. `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604031746_qdrop-bank-qat/train_gpt.py`
   - **Passed**
2. Minimal CPU import/forward smoke
   - **Not runnable on this runner** because both `/usr/bin/python` and `/usr/bin/python3` lack `torch`, even though `requirements.txt` declares it.

## Main risks / tradeoffs

1. **Late eager mode costs throughput.** If bank fake quantization is too slow, fewer final steps could erase the gain from better quantization.
2. **Training/export mismatch remains.** The late-QAT path uses cheap absmax int6 fake quant, while final export still uses GPTQ-lite percentile search.
3. **TTT interaction is uncertain.** Better post-quant weights should help the final artifact, but the downstream legal TTT behavior may move differently than pre-TTT BPB.
