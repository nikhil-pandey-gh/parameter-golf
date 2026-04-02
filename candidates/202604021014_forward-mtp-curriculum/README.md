# Forward-Curriculum MTP on the 11L GPTQ-lite + EMA Base

## Hypothesis

A small-model-friendly **forward curriculum for multi-token prediction (MTP)** can improve sample efficiency under the 10-minute training cap without increasing final artifact bytes, because the extra prediction heads are used only during training and are excluded from export. This candidate also cherry-picks the latest record's **LeakyReLU(0.5)^2** MLP activation, since that change already showed a clear training-side gain on the same family of models.

## Why this is promising here

The record history in this repository shows that the biggest early gains came from compression-aware training and evaluation, while the most recent gains are smaller and come from cleaner training dynamics:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the strongest clean pre-TTT core stack: 11L, XSA4, Partial RoPE, LN scale, VE128, SmearGate, BigramHash, EMA, tight SWA, GPTQ-lite int6 export, and warmdown 3500.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` shows that **LeakyReLU(0.5)^2** alone is a real training-side win on top of that broader design family.
- Prior records mostly pushed compression, evaluation, and optimizer tuning. They do **not** appear to have actually turned on the dormant MTP code path that already exists in the 2026-03-22 script family.

That makes MTP attractive here: it aims at the next bottleneck, **time-limited training efficiency**, while leaving the final exported model unchanged apart from the trained shared trunk.

## Prior records that influenced this candidate

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
   - chosen as the base implementation;
   - best clean stack before TTT and parameter-banking complexity.
2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
   - source of the LeakyReLU(0.5)^2 MLP change.
3. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
   - confirms Partial RoPE + LN scale as stable zero-parameter improvements already inherited through the base.
4. `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
   - confirms XSA4 + EMA as part of the durable winning stack already inherited through the base.

## External research that informed it

1. **Better & Faster Large Language Models via Multi-token Prediction** ([Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737))
   - argues that predicting multiple future tokens with independent heads improves sample efficiency and can improve downstream capability.
2. **Pre-Training Curriculum for Multi-Token Prediction in Language Models** ([Aynetdinov et al., 2025](https://arxiv.org/abs/2505.22757))
   - specifically relevant to this challenge because it finds that **small language models benefit more from a forward curriculum** than from abruptly enabling a full MTP objective.

## What changed versus the chosen base

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Enabled MTP by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.1`
2. **Added a forward curriculum**
   - new `MTP_CURRICULUM_STEPS` knob (default `2000`);
   - head weights ramp from pure NTP to full 2-head MTP gradually instead of switching on all auxiliary targets at once;
   - zero-weight heads are skipped entirely so the curriculum does not spend compute on inactive auxiliary losses.
3. **Threaded curriculum weights through training/eval safely**
   - training passes explicit per-head weights into `GPT.forward(...)`;
   - validation passes zero MTP weights and still evaluates pure next-token loss.
4. **Kept export artifact clean**
   - training-only `mtp_heads` remain excluded from `export_sd`, so the final serialized/quantized artifact still contains only the trunk model.
5. **Cherry-picked LeakyReLU(0.5)^2**
   - replaces ReLU^2 in the base MLP, following the 2026-03-23 record.
6. **Made the script runnable from this directory**
   - default `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root via `__file__`, so the candidate can be launched from inside its own folder without extra path overrides.

## How to run

From this directory:

```bash
RUN_ID=forward_mtp_curriculum \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.1 \
MTP_CURRICULUM_STEPS=2000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves its default dataset/tokenizer paths from the repository root, so the command above works even when launched from inside `candidates/202604021014_forward-mtp-curriculum/`.

Useful ablations:

```bash
# Turn off the new idea entirely
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep MTP but remove the curriculum
MTP_CURRICULUM_STEPS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Evaluation stays the same as the base stack: EMA weights are exported without MTP heads, quantized to mixed int6/int8, then reloaded for full-val and sliding-window BPB evaluation.

## Main expected risks and tradeoffs

- **Training overhead**: even two auxiliary vocab heads add extra logits/loss work during training.
- **Small-model sensitivity**: MTP can hurt small models when enabled too aggressively; the curriculum is meant to reduce that risk, but the schedule may still need tuning.
- **Weighting uncertainty**: `MTP_LOSS_WEIGHT=0.1` and `MTP_CURRICULUM_STEPS=2000` are plausible defaults, not yet H100-tuned.
- **Interaction risk**: LeakyReLU(0.5)^2 and MTP are both reasonable individually, but their interaction on this exact stack is not yet measured.

## Validation

Commands run in this repository:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604021014_forward-mtp-curriculum/train_gpt.py
```

Outcome:

- **Passed**.

Attempted lightweight CPU smoke strategy:

```bash
python - <<'PY'
# import candidate module and run a tiny monkeypatched CPU smoke test
PY
```

Outcome:

- **Not feasible in this container** because `torch` is not installed locally, and the real script also depends on CUDA/FlashAttention for its actual execution path.
