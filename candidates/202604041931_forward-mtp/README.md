# Forward-Curriculum MTP on the 11L GPTQ-lite/EMA stack

## Hypothesis

The repo already contains dormant multi-token prediction (MTP) support, but every prior record kept `MTP_NUM_HEADS=0`. Recent literature suggests that this is exactly what we should revisit for tiny models: MTP improves sample efficiency, and a **forward curriculum** is specifically the missing ingredient that lets small language models benefit from it instead of destabilizing early training.

This candidate therefore starts from the strongest **training-only** record stack and adds a compile-safe forward MTP curriculum, while also backporting the low-risk **LeakyReLU(0.5)^2** activation from the current overall winner.

## Why this looks promising here

1. The best non-TTT base in this repo is `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which already has the strongest training-side ingredients: 11 layers, MLP3x, XSA4, partial RoPE, LN scale, shared value embeddings, EMA, GPTQ-lite, and warmdown 3500.
2. The overall winner `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` shows that **LeakyReLU(0.5)^2** is a real gain, not a speculative tweak.
3. The non-record recurrence sweep in `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` already showed that naive layer recurrence/weight reuse is a poor next bet under a fixed wall-clock budget, so this candidate deliberately avoids that dead end.
4. MTP is attractive for Parameter Golf because the extra heads are **training-only**: this script still strips `mtp_heads.*` from the export, so the candidate pays compute during training but not artifact bytes at submission time.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Activation backport:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Dead-end avoided:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` (naive layer recurrence regressed badly)

There were no prior `candidates/` directories in this checkout.

## External research

- **Fabian Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_** (arXiv:2404.19737) argues that auxiliary multi-token heads improve sample efficiency and help induction-style behavior.
- **Ansar Aynetdinov and Alan Akbik, _Pre-Training Curriculum for Multi-Token Prediction in Language Models_** (arXiv:2505.22757) is the key paper for this repo: it reports that **small language models struggle with always-on MTP**, and that a **forward curriculum** from NTP to MTP is the more effective schedule.

The implementation here follows that second paper's main lesson rather than simply turning on fixed-weight MTP from step 0.

## What changed vs. the chosen base

1. **Forward MTP curriculum enabled by default**
   - `MTP_NUM_HEADS` now defaults to `2`.
   - `MTP_CURRICULUM_ENABLED=1` by default.
   - `MTP_CURRICULUM_START_FRAC=0.35` delays MTP until the model has already learned a reasonable next-token baseline.
   - `MTP_CURRICULUM_RAMP_FRAC=0.25` then turns on the heads gradually instead of all at once.
   - The schedule is applied through a **tensor buffer** (`mtp_head_weights`) rather than a Python flag so it remains visible inside `torch.compile`.

2. **LeakyReLU(0.5)^2 MLP**
   - Backported from the 2026-03-23 winner.
   - Controlled by `LEAKY_RELU_SLOPE` (default `0.5`).

3. **CPU smoke/fallback path**
   - `SMOKE_TEST=1` now instantiates the model on CPU, exercises the MTP path, strips MTP heads from export, reloads the export-only model, and exits.
   - If `flash_attn_interface` is unavailable or the model is not on CUDA, attention falls back to PyTorch SDPA instead of crashing immediately. This is primarily for validation and local bring-up; real leaderboard runs should still use the CUDA/FlashAttention path.

## How to run

From this candidate directory:

```bash
cd candidates/202604041931_forward-mtp

RUN_ID=forward_mtp_seed1337 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.2 \
MTP_CURRICULUM_ENABLED=1 \
MTP_CURRICULUM_START_FRAC=0.35 \
MTP_CURRICULUM_RAMP_FRAC=0.25 \
LEAKY_RELU_SLOPE=0.5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The rest of the strong 2026-03-22 stack remains in-script by default: 11L/512d, MLP3x, XSA on the last 4 layers, partial RoPE (16 dims), LN scale, EMA, GPTQ-lite, shared value embeddings, bigram hash, warmdown 3500, and sliding-window evaluation.

## How to evaluate or smoke test

```bash
cd candidates/202604041931_forward-mtp
SMOKE_TEST=1 python train_gpt.py
```

That path does **not** require dataset shards or a tokenizer. It is only a startup/export sanity check.

## Main risks and tradeoffs

- **Training overhead:** even two auxiliary heads cost extra matmuls and losses, so the gain has to outweigh the reduced step count within the 10-minute cap.
- **Schedule sensitivity:** the start and ramp fractions are informed by the ACL 2025 curriculum paper, but they still may need retuning for this specific 11L/512d stack.
- **Interaction effects:** LeakyReLU^2, EMA, and GPTQ-lite each helped separately in prior repo history, but their combination with MTP has not been validated on 8xH100 yet.
- **Compile behavior:** this candidate avoids the repo's earlier late-QAT constant-folding pitfall by using tensorized head weights, but the real proof is a GPU run with `torch.compile` enabled end to end.

## Validation run here

Commands executed on this runner:

```bash
python -m compileall candidates/202604041931_forward-mtp/train_gpt.py
SMOKE_TEST=1 python candidates/202604041931_forward-mtp/train_gpt.py
```

Outcomes:

- `python -m compileall ...` passed.
- `SMOKE_TEST=1 ...` passed after creating a temporary venv from the repo's existing requirements because the base runner image did not have `torch`/`numpy`/`sentencepiece` preinstalled.
- Smoke output: `smoke_test:ok train_loss:8.3481 eval_loss:6.9618 flash_attn_available:False`
