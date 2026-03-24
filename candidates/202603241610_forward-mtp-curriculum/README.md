# Forward-Curriculum MTP on the 11L EMA + GPTQ-lite record

## Hypothesis

The current best 11-layer recipe is already close to the 16MB byte ceiling, so the strongest next move is a **training-only improvement** that does not enlarge the exported artifact. Multi-token prediction (MTP) is a good fit: it adds auxiliary future-token heads during training, improves representation learning and sample efficiency, and the heads can be **excluded at export time**.

For this repo specifically, I expect the safest version to be a **small, wallclock-aware forward curriculum**: start as pure next-token prediction, then gradually ramp in a single auxiliary MTP head once the model has stabilized. That follows the small-model finding that naive MTP can be too hard early, while still keeping the zero-artifact-cost upside.

## Why this is promising for this repository

This challenge is constrained by both a **10-minute training budget** and a **16MB submission budget**. The frontier record stack already spends almost all available bytes on model quality, so adding more persistent parameters is risky.

This candidate keeps the exported architecture effectively unchanged and instead spends a little extra **training compute** on an auxiliary objective. That matches the repo's recent trend: the best gains now come from refinements that are either byte-neutral or export-free, like EMA and GPTQ-lite clipping.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Important prior trends:

- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` established the strong 11L + XSA + EMA line.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` showed that low-parameter architectural tweaks still help, but also documented a `torch.compile` late-QAT pitfall.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that the most recent wins came from **low-overhead finishing refinements**, not from broad architectural changes.

I also reviewed the non-record `2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`, which is useful mainly as a reminder that wallclock-aware schedules matter in this repo.

## External research that informed this candidate

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (arXiv:2404.19737). This paper argues that predicting multiple future tokens with independent auxiliary heads improves sample efficiency and downstream quality while leaving inference flexible.
- Ansar Aynetdinov and Alan Akbik, **"Pre-Training Curriculum for Multi-Token Prediction in Language Models"** (arXiv:2505.22757). This is the key paper for the repo setting here: it reports that **smaller language models struggle with naive MTP**, and that a **forward curriculum** helps them benefit from MTP during pre-training.

These two results together motivate a small-head, ramped-loss MTP variant on top of the current record implementation.

## What changed versus the chosen base implementation

Base file:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Candidate changes:

1. **Enabled MTP by default** with a conservative setting:
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.10`

2. Added a **forward curriculum** for the MTP loss:
   - `MTP_START_FRAC=0.10`
   - `MTP_FULL_FRAC=0.35`
   - The schedule is tied to **wallclock progress** when `MAX_WALLCLOCK_SECONDS` is active, which fits this repository better than iteration-based scheduling.

3. Kept the MTP schedule **driven from the training loop** instead of relying on a dynamically toggled class attribute. Before the curriculum starts, training stays on the pure next-token path; once active, the current MTP weight is passed into `forward(...)`.

4. Kept warmup and evaluation on the **pure next-token path**, so the auxiliary MTP branch is only exercised during the active part of training.

5. Made the script **self-contained when launched from the candidate directory** by resolving default dataset/tokenizer paths relative to the repository root instead of the current working directory.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603241610_forward-mtp-curriculum
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The default candidate-specific knobs are already baked into the script, but they can still be overridden:

```bash
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.10 \
MTP_START_FRAC=0.10 \
MTP_FULL_FRAC=0.35 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603241610_forward-mtp-curriculum/train_gpt.py
```

Outcomes:

- Both compile checks **passed**.
- A stronger smoke run was **not feasible in this container**:
  - `torch` is not installed here (`ModuleNotFoundError: torch` from an import probe).
  - the default local dataset/tokenizer assets are also absent in this environment.
  - the script requires a CUDA + FlashAttention runtime to do a meaningful startup test.

## Main expected risks or tradeoffs

- **Small-model MTP can regress without a good schedule.** That is the main reason this candidate uses a forward curriculum instead of always-on full-strength MTP.
- **There is still some training overhead.** Although export bytes stay flat because MTP heads are dropped before serialization, the extra auxiliary head does add per-step compute.
- **The best record is already strong.** This may be a modest-gain idea rather than a large architectural leap, but it is intentionally targeted at the repo's most likely remaining headroom: better sample efficiency at unchanged artifact size.
