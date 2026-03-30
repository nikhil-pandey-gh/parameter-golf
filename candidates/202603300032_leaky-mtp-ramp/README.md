# LeakyReLU^2 + Warmdown-Decayed Horizon-Weighted MTP

## Hypothesis

The strongest non-TTT stack in this repository already gets most of the way to SOTA, so the next high-leverage change should improve **sample efficiency during training** without increasing the final artifact size.

This candidate keeps the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` stack as the base, then adds two training-side changes:

1. **LeakyReLU(0.5)^2** in the MLP, cherry-picked from the current best record, to preserve negative-slope gradient flow while keeping the successful squared-activation bias.
2. **Training-only multi-token prediction (MTP)** with two auxiliary heads, but adapted for this challenge by:
   - weighting nearer horizons more strongly with `1 / (k + 1)^p`, and
   - decaying the MTP loss to zero during late warmdown so the final phase focuses on the standard next-token objective and late-QAT/quantization-sensitive weights.

The key bet is that MTP helps the model learn useful predictive structure earlier in the 10-minute budget, while the warmdown decay reduces train/eval mismatch and avoids paying a late-stage optimization tax on an auxiliary objective that is not used at evaluation.

## Why this is promising for this repository

Recent winning runs already concentrate on **artifact-neutral or artifact-light** improvements:

- better averaging (`EMA`),
- better post-training quantization (`GPTQ-lite`),
- better positional handling (`Partial RoPE`),
- deeper but still compact 11-layer stacks,
- evaluation improvements such as sliding-window eval and legal TTT.

MTP fits that pattern well because the extra heads are **training-only** and are explicitly excluded from export in the script, so the final saved artifact still contains only the base model weights. That is especially attractive under the challenge's 16 MB artifact cap.

This candidate also avoids directions that repo evidence already made less attractive under a fixed 600-second budget:

- pure longer training,
- layer recurrence / depth reuse that costs too many steps,
- wider or slower MLP variants like SwiGLU in the tight wall-clock regime,
- more evaluation-only complexity.

## Prior records that influenced this candidate

### Base implementation

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

This is the cleanest strong non-TTT base in the repo:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- 3x MLP
- XSA on the last 4 layers
- Partial RoPE (16/64)
- layerwise LN scaling
- VE128 on layers 9-10
- SmearGate + BigramHash
- EMA + GPTQ-lite export
- warmdown 3500, late QAT threshold 0.15

### Additional repo evidence used here

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - provided the `LeakyReLU(0.5)^2` MLP activation, which that README reports as a meaningful win.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms Partial RoPE + LN scale are part of the winning stack.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative result: layer recurrence hurt badly under a strict wall-clock cap.

There were **no prior experiments under `candidates/`** when this candidate was created.

## External research that informed this candidate

### Multi-token prediction as an auxiliary objective

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-Token Prediction"** (arXiv:2404.19737)
  - <https://arxiv.org/abs/2404.19737>

This paper argues that training LMs to predict multiple future tokens improves sample efficiency and generative performance with no inference-time dependency on the auxiliary heads. That maps unusually well onto Parameter Golf because the challenge penalizes final artifact size, not temporary training-only heads.

### Alternatives considered but not chosen

- Zhenzhong Lan et al., **"ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"** (arXiv:1909.11942)
  - <https://arxiv.org/abs/1909.11942>
- Mostafa Dehghani et al., **"Universal Transformers"** (arXiv:1807.03819)
  - <https://arxiv.org/abs/1807.03819>

These are good reminders that parameter sharing and recurrence can help parameter efficiency, but the repository's own evidence makes them less attractive here: the 1x5090 recurrence exploration was clearly negative under a 600-second wall-clock limit, so this candidate prioritizes a **training-only auxiliary loss** over more invasive recurrence/tied-depth changes.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. changes the MLP activation from `relu^2` to `LeakyReLU(0.5)^2`,
2. enables MTP by default with `MTP_NUM_HEADS=2`,
3. adds **horizon weighting** so nearer targets get more weight than farther ones,
4. adds **warmdown-decayed MTP weighting**:
   - full auxiliary weight while the LR scale is above `MTP_DECAY_SCALE`,
   - linear decay to zero as the run enters the final warmdown regime,
5. keeps exporting the final artifact **without** the MTP heads,
6. resolves dataset/tokenizer defaults relative to the repo root so `train_gpt.py` can be run from inside this candidate directory.

## How to run

From the repository root:

```bash
cd candidates/202603300032_leaky-mtp-ramp

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate defaults already encode the main hypothesis:

- `MTP_NUM_HEADS=2`
- `MTP_LOSS_WEIGHT=0.15`
- `MTP_DECAY_SCALE=0.25`
- `MTP_HORIZON_POWER=1.0`

An explicit launch with the core settings shown is:

```bash
cd candidates/202603300032_leaky-mtp-ramp

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.15 MTP_DECAY_SCALE=0.25 MTP_HORIZON_POWER=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
LATE_QAT_THRESHOLD=0.15 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Evaluation behavior

Evaluation remains the standard path used by the base record:

- EMA weights are applied before export.
- The saved model excludes MTP heads.
- The exported model is quantized with GPTQ-lite-style int6/int8 mixed quantization.
- Final evaluation still uses the standard next-token model and sliding-window BPB.

## Main expected risks and tradeoffs

- **Throughput risk:** even two MTP heads add training compute; the sample-efficiency gain may or may not offset losing some optimizer steps.
- **Late-phase heuristic risk:** decaying MTP during warmdown is motivated by the challenge setup, but the exact threshold (`0.25`) is heuristic.
- **Quantization transfer risk:** a better pre-export representation does not always survive quantization equally well.
- **Interaction risk:** `LeakyReLU(0.5)^2` is promising in repo evidence, but its best learning-rate / regularization pairing may differ slightly from the relu-squared-tuned base.

## Validation

Validation commands and outcomes for this candidate are recorded below and should be updated after each local check:

- From the candidate directory: `python -m compileall train_gpt.py` — **passed**
- From the repo root: `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603300032_leaky-mtp-ramp/train_gpt.py` — **passed**
- Runtime smoke check — **not run in this environment**
  - This runner does not have `torch` installed and also lacks the expected tokenizer file plus cached FineWeb dataset directory (`data/tokenizers/fineweb_1024_bpe.model`, `data/datasets/fineweb10B_sp1024/`), so a startup run would not have been a meaningful validation of the candidate itself.
