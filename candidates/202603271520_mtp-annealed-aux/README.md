# MTP-Annealed Auxiliary Head on the GPTQ-lite + EMA 11L Base

## Hypothesis

Enable a **single train-only multi-token prediction (MTP) head** on top of the current best non-TTT 11-layer stack, then **anneal that auxiliary loss away during late training**.

The bet is that this improves sample efficiency early, when the run is still representation-limited, while preserving the late-stage next-token and quantization behavior that actually determines the exported artifact score.

## Why this is promising for this repository

This repository's strongest runs are clearly **wallclock-limited**. The best recipe is already a stable 11-layer `GPTQ-lite + EMA + XSA + partial RoPE + VE` stack, and the main open question is which low-overhead training objective gives more value per minute.

`train_gpt.py` in the strongest recent record family already contains dormant MTP support, and that support already **drops `mtp_heads.*` from the exported state dict**. That makes MTP unusually attractive here: it can act as a training-only regularizer without consuming final artifact bytes.

## Prior experiments that influenced this candidate

There was **no existing `candidates/` directory** in this checkout, so this is the first candidate folder.

The main record influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best non-TTT base in this repo.
  - Supplies the 11L `EMA + tight SWA + GPTQ-lite + XSA4 + partial RoPE + LN scale + VE128` stack.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Shows that small zero/low-param bias tweaks keep paying off on top of the same architecture family.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Confirms that frontier gains now mostly come from better objectives/eval/training refinements, not wholesale architectural resets.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - Important because it already carried dormant MTP hooks, but the shipped runs kept `mtp_num_heads=0`.

## External research that informed it

- **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction"** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737))
  - Argues that predicting multiple future tokens improves **sample efficiency** and encourages stronger induction behavior.
- **DeepSeek-V3 Technical Report** ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))
  - Explicitly reports using a **multi-token prediction training objective** as part of a strong modern LLM recipe.

The repository-specific twist is that Parameter Golf cares about **10-minute training** and **16MB final artifact size**, so a **train-only** auxiliary objective is much more attractive than in a standard deployment setting.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

- `MTP_NUM_HEADS` now defaults to `1` instead of `0`.
- `MTP_LOSS_WEIGHT` now defaults to `0.1`.
- Added `MTP_ANNEAL_THRESHOLD` (default `0.20`).
  - While LR scale is above `0.20`, MTP runs at full strength.
  - During the last 20% of warmdown, the auxiliary weight linearly decays to zero.
- Marked `mtp_heads` as **QAT-skipped** so late fake-quantization only targets tensors that actually ship in the artifact.
- The MTP scale is logged during training so late-stage annealing is visible in the logs.

Crucially, export remains trunk-only:

- `mtp_heads.*` are excluded from `final_model.pt`
- quantized export still excludes them
- both raw and quantized exports now carry trunk-only `model_kwargs`, so reload shape stays self-consistent
- roundtrip eval still reconstructs an eval model with `mtp_num_heads=0`

So the candidate pays **training-time compute only**, not submission bytes.

## How to run

From this candidate directory:

```bash
cd candidates/202603271520_mtp-annealed-aux

RUN_ID=mtp_annealed_aux \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The new knobs are:

```bash
MTP_NUM_HEADS=1
MTP_LOSS_WEIGHT=0.1
MTP_ANNEAL_THRESHOLD=0.20
```

If you want to compare against the exact base behavior, set:

```bash
MTP_NUM_HEADS=0
```

## Evaluation expectations

- Training-time overhead should be small but non-zero, because the extra head only touches the final hidden states.
- Exported artifact size should remain very close to the base because the auxiliary head is stripped before serialization.
- If the hypothesis is correct, the gain should appear mostly as better trunk quality at the same wallclock.

## Main risks and tradeoffs

- Even a small extra head can slightly reduce steps completed in 600s.
- MTP gains in the literature are strongest at larger scales; a tiny 11-layer model may benefit less.
- The anneal threshold (`0.20`) is only lightly reasoned, not tuned.
- Because the auxiliary head is dropped at export, all benefit must be absorbed into the shared trunk.

## Validation

Commands run in this workspace:

- `python -m compileall candidates/202603271520_mtp-annealed-aux/train_gpt.py`

Outcome:

- `compileall` succeeded.
- A real CPU smoke run is **not feasible in this workspace** unless the challenge tokenizer and FineWeb shard files are present locally. This checkout currently does not include `data/tokenizers/fineweb_1024_bpe.model` or `data/datasets/fineweb10B_sp1024/fineweb_{train,val}_*.bin`.
