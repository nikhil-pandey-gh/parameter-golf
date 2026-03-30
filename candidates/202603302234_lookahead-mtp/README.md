# Conservative Lookahead MTP on the 11L EMA + GPTQ-lite Stack

## Hypothesis

A **single training-only multi-token-prediction (MTP) head** should improve sample efficiency on the strongest self-contained 11-layer lineage in this repo, without materially increasing final artifact bytes, because the auxiliary head is already stripped from export in the underlying code path.

This challenge is simultaneously constrained by **artifact size** and **10-minute wallclock training**, so a lookahead loss is attractive if it can add supervision during training while disappearing at export time.

## Why this is promising for this repository

The record history in `records/` suggests a few things very clearly:

- The winning path is the 11-layer `EMA + mixed-int6 + XSA + partial-RoPE + BigramHash/SmearGate` family, especially `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` and the later `2026-03-23` TTT extension.
- Full depth recurrence / looped layers looked like a dead end under the 10-minute cap. See `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` and `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`.
- The 2026-03-20 through 2026-03-23 record lineage already carries a dormant `MTP_NUM_HEADS` / `MTP_LOSS_WEIGHT` path in code, and the export path explicitly removes `mtp_heads` before saving the final artifact.

That combination makes conservative MTP a better next bet than more aggressive recurrence or a broad architecture rewrite: it is **aligned with the strongest repo lineage**, **low-risk to implement**, and **nearly free at export time**.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Chosen as the base implementation because it is the strongest self-contained pre-TTT 11-layer script and already contains the dormant MTP hook.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Confirms that this 11-layer family is still the strongest overall direction, even though that script is substantially more complex.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - Explicitly notes that depth recurrence needed too many steps to pay off under this budget.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Records a direct negative result for layer recurrence, reinforcing that the next candidate should avoid a FLOP-heavy recurrence pivot.

## External research that informed it

- **Self-Distillation for Multi-Token Prediction** — https://arxiv.org/abs/2603.23911
- **Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing** — https://arxiv.org/abs/2603.17942

These papers reinforce the same broad idea: future-token lookahead contains useful training signal even when the final deployment path remains standard next-token prediction. That maps unusually well onto this repo because the MTP head can be used during training and excluded from the final export.

## What changed versus the chosen base implementation

Base copied from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

- Default `MTP_NUM_HEADS` changed from `0` to `1`.
- Default `MTP_LOSS_WEIGHT` changed from `0.2` to `0.1`.
- No other architectural or export-path changes were made.

The goal is to isolate the effect of **one conservative lookahead head** rather than stacking multiple new ideas at once.

## How to run / evaluate

From this candidate directory:

```bash
RUN_ID=lookahead_mtp \
SEED=1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful sweeps if the first run is stable:

```bash
# Disable MTP for direct ablation against the copied base
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Slightly stronger auxiliary objective
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.15 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Validated in this workflow with lightweight checks that fit the current environment:

```bash
python -m compileall candidates/202603302234_lookahead-mtp/train_gpt.py
```

Outcome: **passed**.

I also checked whether a CPU smoke test was feasible in this runner:

```bash
python - <<'PY'
import importlib.util
for mod in ('torch', 'flash_attn_interface'):
    print(f'{mod}:{importlib.util.find_spec(mod)}')
PY
```

Outcome: both `torch` and `flash_attn_interface` are unavailable in this workflow environment, so a real runtime smoke test of this H100-oriented script was **not feasible here** without adding new infrastructure.

## Main expected risks / tradeoffs

- Even one auxiliary MTP head adds some training-time overhead, so the gain from denser supervision has to beat the loss in steps.
- If `MTP_LOSS_WEIGHT` is too high, the model may optimize the auxiliary objective at the expense of final next-token quality.
- This candidate intentionally does **not** stack LeakyReLU², TTT, or other new changes on top of the base so the effect is easier to attribute; that also means it may leave performance on the table versus a more aggressively stacked follow-up.
