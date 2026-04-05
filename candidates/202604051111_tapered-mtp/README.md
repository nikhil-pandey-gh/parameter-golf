# 202604051111_tapered-mtp

## Hypothesis

The strongest pre-TTT branch in this repo is already close to saturating the 16MB artifact budget, so the next useful gain should come from **better sample efficiency during the 600s training window**, not from adding more exported parameters.

This candidate turns on a **training-only multi-token prediction (MTP) auxiliary loss** on top of the 2026-03-22 `11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack. The core bet is:

1. **One extra future-token head can improve token efficiency** without changing the final exported artifact, because the auxiliary `mtp_heads` are still excluded from serialization.
2. **Initializing the MTP head from the main prediction head** should avoid the cold-start problem of zero-init auxiliary heads.
3. **Tapering the auxiliary loss to zero during warmdown** should let the final checkpoint re-focus on next-token quality and quantized export, instead of paying a permanent interference tax from the auxiliary objective.

## Why this is promising here

- Repo history says the durable wins came from **compression-aware training/export + cheap eval/training tricks**, not from slower architectures or recursive depth.
- Both the early and late record branches show that **wall-clock throughput is precious**: recurrence and SwiGLU underperformed when they cut step count too much, while EMA, Partial RoPE, XSA, and GPTQ-lite all won because they added little or no training-time cost.
- The strongest 03-20 through 03-23 scripts already carry dormant `mtp_heads` code, but the winning README configs never enable it. That makes MTP a rare case where the repo already has most of the plumbing, but the idea is still underexplored experimentally.

## Influential prior records

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest pre-TTT score in the repo's main lineage
  - already includes EMA, GPTQ-lite, Partial RoPE, LN scale, XSA on late layers, VE, BigramHash, and the export path this candidate should preserve
- **Related lineage:** `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - useful because it already contained `mtp_heads` support, even though the published config did not use it
- **What not to do:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - explicit negative result for layer recurrence under a fixed wall-clock budget
- **Orthogonal but not the base for this candidate:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest post-TTT result, but much of that gain is eval-side and more operationally complex than needed for a candidate experiment

There were **no prior runs under `candidates/`** when this candidate was created.

## External research that informed this candidate

- **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737)**  
  Primary motivation. The paper argues that predicting multiple future tokens from a shared trunk improves sample efficiency and can be treated as an auxiliary training objective.

- **Guoliang Zhao et al., "Self-Distillation for Multi-Token Prediction" (arXiv:2603.23911)**  
  Motivated the "preserve the main head" angle. This candidate does not implement full self-distillation, but it borrows the same intuition: auxiliary future heads should help without permanently degrading the main next-token head.

- **Considered but not chosen as the main idea**
  - **Mixture-of-Recursions** (arXiv:2507.10524): interesting, but the repo already has negative recurrence results under the 10-minute budget and MoR would require broader architectural changes.
  - **HESTIA** (arXiv:2601.20745): promising differentiable QAT idea, but more invasive and harder to validate here than a training-only MTP auxiliary head.

## What changed vs the chosen base

Starting from the 2026-03-22 record script, this candidate makes only the following surgical changes:

1. **Default MTP is on**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.15`

2. **New MTP schedule controls**
   - `MTP_WARMUP_STEPS=250`
   - `MTP_TAPER_SCALE=0.30`
   - Effective auxiliary weight ramps in early, then linearly tapers toward zero once LR scale drops below `0.30`

3. **Main-head initialization for the auxiliary head**
   - `MTP_INIT_MODE=main`
   - The future-token head is initialized from the tied embedding / main prediction weights instead of zeros

4. **Runtime MTP lambda**
   - The model now receives the auxiliary weight as a runtime scalar tensor
   - Validation always passes `0`, so the eval path stays pure next-token scoring

5. **Candidate-directory usability fix**
   - Default `DATA_PATH` and `TOKENIZER_PATH` are now resolved relative to the repository root, so `train_gpt.py` can be launched from inside this candidate directory as required

Everything else stays on the same 03-22 stack: 11 layers, EMA, GPTQ-lite export, Partial RoPE, LN scaling, XSA on the last 4 layers, VE, BigramHash, and the same export-time exclusion of `mtp_heads`.

## How to run

From this candidate directory:

```bash
cd candidates/202604051111_tapered-mtp
RUN_ID=mtp_candidate \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs to sweep first:

```bash
MTP_NUM_HEADS=1
MTP_LOSS_WEIGHT=0.10
MTP_WARMUP_STEPS=250
MTP_TAPER_SCALE=0.30
MTP_INIT_MODE=main
```

If you want to disable the idea entirely while keeping the candidate script:

```bash
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected risks / tradeoffs

- **Step-time tax:** even one future-token head adds another vocab projection during training, so the sample-efficiency gain has to outweigh the reduced steps.
- **Main-head interference:** if the auxiliary weight is too high or tapers too late, the final next-token loss can regress despite better intermediate features.
- **Init bias:** copying the main head into the future head may make the auxiliary task too conservative; zero-init might still win after tuning.
- **No 8xH100 confirmation yet:** this is a candidate implementation, not a validated record.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604051111_tapered-mtp/train_gpt.py
```

Outcome:

- `compileall` succeeded

CPU smoke-test status:

- A real local smoke launch was **not feasible in this environment**
- `torch` is not installed here
- `flash_attn_interface` is not installed here
- the local repo checkout does not contain the expected FineWeb shards or tokenizer files under `data/`

So this candidate was validated to the syntax/packaging level here, but not executed end-to-end locally.
