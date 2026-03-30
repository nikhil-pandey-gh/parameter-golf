# MTP Warmstart on the 11L EMA + GPTQ-lite Stack

## Hypothesis

The current strongest training-only recipe in this repository has largely exhausted obvious architecture and quantization tweaks, but it still trains on a pure next-token objective. A small multi-token prediction (MTP) auxiliary loss should improve sample efficiency during the fixed 10-minute training budget by forcing the trunk to model slightly longer-horizon continuations early, then exporting back to the same single-token artifact because the auxiliary heads are excluded from the saved model.

## Why this is promising for this repository

Repository review shows a clear winning trend toward a stable 11-layer family: 512 width, 2048 training context, 3x MLP, SmearGate + BigramHash, XSA in late layers, EMA, and increasingly sophisticated int6 export. The biggest recent gains have mostly come from compression and evaluation tricks, while genuinely new training-objective changes have been sparse.

That makes MTP attractive here for two reasons:

1. It attacks a relatively underexplored axis in this repo: training-time sample efficiency rather than another quantization retune.
2. It fits the challenge constraints unusually well because the extra prediction heads are used only during training and are already excluded from export, so the final artifact stays on the same budgeted architecture.

One especially useful repo-specific observation is that several stronger record scripts already carried dormant MTP plumbing, but every published configuration I found kept `MTP_NUM_HEADS=0`. This candidate turns that unused path into the main experiment instead of redoing another minor LR or clip-percentile sweep.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Supporting lineage:

- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`: partial RoPE + layerwise LN scaling are kept.
- `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`: established the 11L / XSA4 / EMA training family.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: shows there is still room in the stack for training-side gains, but this candidate intentionally stays training-only and avoids importing the much larger TTT / banking changes.

More broadly, repo review suggested these winning trends:

- sliding-window evaluation became table stakes,
- compression-aware training and export dominate the mid/late leaderboard,
- SmearGate + BigramHash + XSA + EMA form the most successful core family,
- simple long-context-only or recurrence-heavy detours were weaker or riskier under the 10-minute constraint.

## External research that informed it

- Fabian Gloeckle et al., _Better & Faster Large Language Models via Multi-token Prediction_ (arXiv:2404.19737). The main takeaway for this repo is that MTP can improve sample efficiency and downstream quality while reusing a shared trunk.
- Guoliang Zhao et al., _Self-Distillation for Multi-Token Prediction_ (arXiv:2603.23911). This reinforces that multi-token supervision can be useful without sacrificing main-head quality, which matters here because leaderboard scoring is still standard next-token BPB after export.
- Lorenzo Noci et al., _Thinking into the Future: Latent Lookahead Training for Transformers_ (arXiv:2603.20219). This paper is more ambitious than what is implemented here, but it supports the broader hypothesis that future-token/foresight supervision is a promising direction when sequence modeling quality is bottlenecked by one-step-only training.

## What changed versus the chosen base implementation

This candidate starts from the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` training-only record and makes the smallest set of changes needed to test the idea cleanly:

- enable MTP by default with `MTP_NUM_HEADS=2`
- set a conservative auxiliary weight with `MTP_LOSS_WEIGHT=0.15`
- keep export excluding `mtp_heads`, so the auxiliary parameters do not count toward the final artifact
- add a PyTorch SDPA fallback when `flash_attn_interface` is unavailable, including enabling non-flash SDPA kernels in that path
- add CPU-safe guards for compile, fused optimizers, CUDA synchronization, and bf16 casting so the script can be smoke-tested off-GPU when the runtime has PyTorch installed

Everything else intentionally stays close to the proven March 22 stack:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- MLP 3x
- SmearGate + BigramHash
- XSA on the last 4 layers
- partial RoPE + LN scaling
- EMA + tight SWA
- GPTQ-lite int6 export + warmdown 3500

## Why this differs from existing records and candidates

There was no pre-existing `candidates/` directory in the repo when this was created.

Compared with the published `records/` lineage, this candidate is differentiated because it changes the training objective itself rather than adding another compression heuristic, bucket-count retune, or evaluation trick. The repo already explored long-context scaling, multiple quantization schemes, EMA/SWA variants, XSA, value embeddings, and TTT. Nonzero MTP was the clearest promising axis that appeared supported in code but unused in the published record history.

## How to run or evaluate it

From the candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key defaults baked into this candidate:

- `MTP_NUM_HEADS=2`
- `MTP_LOSS_WEIGHT=0.15`
- same March 22 training stack for the rest of the model/export recipe

Useful overrides for local experimentation:

```bash
ENABLE_TORCH_COMPILE=0 MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.10 python train_gpt.py
```

## Validation

Validated here with the cheapest checks available in this environment:

```bash
python -m compileall candidates/202603301625_mtp-warmstart/train_gpt.py
```

Outcome:

- `compileall` succeeded.

I also attempted a minimal CPU smoke harness using:

- a throwaway SentencePiece tokenizer,
- tiny synthetic FineWeb-format train/val shards,
- reduced model settings (`NUM_LAYERS=2`, `MODEL_DIM=64`, `TRAIN_SEQ_LEN=32`, etc.),
- `ENABLE_TORCH_COMPILE=0`.

That full smoke run was **not feasible in this runner** because the environment does not currently have a `torch` installation available, so the script could not be executed past import time. The candidate script itself now includes CPU/SDPA fallbacks so this smoke path should work in any environment that has PyTorch plus the repo's normal Python dependencies.

## Main expected risks or tradeoffs

- Extra MTP heads add training-time compute and optimizer state, so the gain must beat the step-time slowdown.
- If the auxiliary loss is weighted too strongly, it may improve horizon modeling but slightly harm the main next-token head that actually determines BPB.
- The repo's strongest recent gains have been tightly coupled to quantization; MTP could improve pre-quant quality more than post-quant quality.
- Two heads at weight `0.15` is an informed but still unvalidated default for this exact stack; the best setting may be `1` head or a smaller weight.

## Suggested next experiments if this looks promising

- sweep `MTP_NUM_HEADS in {1, 2, 3}` with `MTP_LOSS_WEIGHT in {0.05, 0.10, 0.15}`
- combine this candidate with the later LeakyReLU² activation change from `2026-03-23`
- test whether MTP helps more on the March 23 parameter-banked stack or the simpler March 22 training-only stack
- try a late-phase MTP anneal if compiled toggling can be done safely without reintroducing the constant-folding issue that previously affected late QAT
