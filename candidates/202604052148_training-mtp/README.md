# Training-only MTP on the LeakyReLU² + Legal TTT stack

## Hypothesis

Add a **single training-only multi-token prediction (MTP) head** to the current best in-tree stack so the shared trunk learns a slightly richer future prediction problem during the fixed 600s training budget, while **keeping exported artifact size effectively unchanged** because the extra head is dropped before serialization and quantization.

This candidate predicts the token after the standard next-token target (`t+2` relative to the input token) with one auxiliary head and a modest loss weight.

## Why this is promising here

Repository review suggests the leaderboard is already squeezing most of the obvious artifact-side gains out of the 11-layer family:

- the strongest line is now **11L + MLP3x + Bigram/Smear + XSA + EMA + Partial RoPE/LN-scale + better quantization/export + legal TTT**;
- several prior scripts already carry dormant MTP plumbing, but the published record READMEs and logs still keep it at `mtp_num_heads:0`;
- negative results repeatedly warn that **heavier architectural rewrites** like naive recurrence or slower activations can lose under the 10-minute wallclock cap even when they look attractive on paper.

That makes **training efficiency** the best remaining angle. MTP is attractive because it spends extra compute only during training, and this codebase already excludes `mtp_heads.*` from the exported state dict.

## Prior runs that influenced this candidate

1. **Base implementation:** [`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`](../../records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md)  
   This is the current best in-tree stack and already has the right export path for training-only MTP heads.
2. **Architecture lineage:** [`records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`](../../records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md) and [`records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`](../../records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md)  
   These established the winning 11-layer/XSA/EMA/Partial-RoPE family that the latest record builds on.
3. **Dead-end evidence:** [`records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`](../../records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md)  
   Its recurrence ablation was strongly negative, which pushed this candidate away from ALBERT-style depth sharing even though that looked interesting in external research.

There were **no prior folders under `candidates/`** when this candidate was prepared.

## External research that informed the choice

- **Fabian Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_**  
  <https://arxiv.org/abs/2404.19737>  
  The core claim is that asking a shared trunk to predict multiple future tokens improves sample efficiency and encourages stronger induction-style behavior.
- **Considered but not chosen**
  - **ALBERT** (<https://arxiv.org/abs/1909.11942>) and **Universal Transformer** (<https://arxiv.org/abs/1807.03819>) suggested parameter sharing / recurrent depth reuse, but repository evidence says naive recurrence is a bad fit for this challenge's wallclock limit.
  - **DeepSeek-V2 / MLA** (<https://arxiv.org/abs/2405.04434>) and more aggressive low-bit ideas such as **LSQ** (<https://arxiv.org/abs/1902.08153>) or **NF4/FP4** (<https://arxiv.org/abs/2305.14314>, <https://arxiv.org/abs/2310.16836>) looked interesting, but they require a larger refactor or a riskier quantization path than this candidate.
  - **BLT** (<https://arxiv.org/abs/2412.09871>) and **SSMax** (<https://arxiv.org/abs/2501.19399>) looked promising longer-term, but both would force broader tokenizer or attention-kernel changes than I wanted for a clean single-file candidate.

## What changed versus the chosen base implementation

Starting from `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes only candidate-local changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`
   - wire `mtp_heads` into AdamW so the auxiliary head actually trains when enabled
2. **Make the script runnable from the candidate directory**
   - dataset/tokenizer defaults now resolve from the repository root via `Path(__file__)`, instead of assuming execution from repo root.
3. **Set defaults to the actual base stack**
   - `BIGRAM_VOCAB_SIZE=1536`
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`
4. **Add a FlashAttention fallback**
   - if `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA **inside the existing CUDA-first training path**. This helps when the fast import is missing, but it is **not** a full CPU portability layer.
5. **Validate batch-shape assumptions early**
   - `TRAIN_BATCH_TOKENS` now raises a clear error if it is incompatible with `WORLD_SIZE`, `GRAD_ACCUM_STEPS`, or `TRAIN_SEQ_LEN`.

The export path still removes `mtp_heads.*` before saving `final_model.pt` / `final_model.int6.ptz`, so the auxiliary head does **not** count against the final artifact.

## How to run

From this directory:

```bash
cd candidates/202604052148_training-mtp
RUN_ID=training_mtp SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides while iterating:

```bash
MTP_NUM_HEADS=0            # disable the auxiliary head for A/B testing
MTP_LOSS_WEIGHT=0.10       # softer auxiliary pressure
TTT_ENABLED=0              # isolate the training-only MTP effect
EVAL_SEQ_LEN=1536          # cheap follow-up sweep suggested by prior records
```

## Main risks and tradeoffs

- **Step-time tax:** even one extra vocabulary head adds training compute, so the candidate could lose total optimizer steps if the auxiliary loss is too expensive.
- **Tiny-model mismatch:** the MTP paper is promising, but its strongest gains were not demonstrated on this exact tiny-LLM / hard-wallclock regime.
- **Interaction risk with TTT:** MTP may improve pre-TTT quality but still interact unpredictably with the record's score-first adaptation path.
- **Potential under-tuning:** `1` head and `0.15` weight are conservative defaults, not a finished sweep.

## Validation

Commands run while preparing this candidate:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202604052148_training-mtp/train_gpt.py
python - <<'PY'
try:
    import flash_attn_interface
    print('flash_attn_interface:ok')
except Exception as e:
    print(f'flash_attn_interface:missing:{type(e).__name__}:{e}')
try:
    import torch
    print('torch:ok')
except Exception as e:
    print(f'torch:missing:{type(e).__name__}:{e}')
PY
```

Observed outcomes:

- both `compileall` commands passed;
- a real CPU smoke run was **not feasible in this runner** because the environment does not currently have `torch` installed, and it also lacks `flash_attn_interface`.
