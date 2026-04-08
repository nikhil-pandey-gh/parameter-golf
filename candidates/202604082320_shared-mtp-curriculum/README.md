# Shared-Head MTP Curriculum on the PR #549 Stack

## Hypothesis

The current best stack already carries dormant multi-token prediction (MTP) hooks, but all known record runs left them off. A small-model-friendly **forward MTP curriculum** should improve sample efficiency during the 10-minute training budget, and using the **existing tied LM head** for a single auxiliary horizon should preserve that gain without adding export bytes.

## Why this is promising here

1. **Repository evidence**: the frontier has moved through better training/eval efficiency on the same 11-layer backbone rather than broad architectural rewrites, and the strongest record already includes unused MTP plumbing in its training script.
2. **Challenge fit**: MTP is a training-only objective, so it can target convergence without requiring a new tokenizer, a new inference path, or extra serialized weights.
3. **Small-model-specific research**: recent work suggests plain MTP can be hard for SLMs, but a forward curriculum helps them benefit from it.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - chosen as the direct base because it is the current best full stack and already contains dormant MTP code paths.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - the strongest clean pre-TTT core, reinforcing that the right place to add a new idea is training efficiency, not a bigger rewrite.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - confirms that the modern stack is cumulative: partial RoPE, LN scaling, EMA, and efficient compression already earn their keep and should stay intact.

## External research that informed it

- **Better & Faster Large Language Models via Multi-token Prediction** (Gloeckle et al., 2024) — <https://arxiv.org/abs/2404.19737>  
  Motivates MTP as a sample-efficiency objective that improves downstream next-token quality.
- **Pre-Training Curriculum for Multi-Token Prediction in Language Models** (Aynetdinov & Akbik, 2025) — <https://arxiv.org/abs/2505.22757>  
  The key paper for this repo specifically: it reports that smaller language models struggle with naive MTP and benefit from a forward curriculum.

I also reviewed recent compression-oriented ideas such as LSQ/QDrop-style quantization and ALBERT-style weight sharing, but they looked more invasive for a single candidate than a targeted MTP change on the current best stack.

## What changed versus the chosen base

Base: `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate keeps the PR #549 architecture, optimizer split, EMA/TTT setup, and quantization path, then swaps the dormant MTP implementation for a budget-aware version:

| Area | Base | This candidate |
|---|---|---|
| Auxiliary objective | Optional extra MTP heads, disabled in known runs | **Shared-head MTP** reusing the tied LM projection |
| Default MTP setting | `MTP_NUM_HEADS=0` | `SHARED_MTP_STEPS=1` |
| Small-model stabilization | None | **Forward curriculum**: linearly ramp auxiliary weight to `0.12` over `1500` steps |
| Export path | Drops `mtp_heads` if present | No extra MTP weights are created, so export stays clean automatically |
| Eval / TTT cost | No MTP at eval | Eval model is instantiated with `shared_mtp_steps=0`, so there is still no eval-time MTP overhead |

Concretely, the training loss stays standard next-token cross-entropy plus a ramped auxiliary loss on future offset `+2`, using the same output projection already used for next-token prediction.

## How to run

From the candidate directory:

```bash
cd candidates/202604082320_shared-mtp-curriculum

NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SHARED_MTP_STEPS=1 SHARED_MTP_LOSS_WEIGHT=0.12 SHARED_MTP_WARMUP_STEPS=1500 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main risks and tradeoffs

- **Step-time overhead**: shared-head MTP still adds extra logits / cross-entropies during training, so any quality gain must exceed the throughput hit.
- **Shared-head constraint**: reusing one projection is byte-efficient, but it may underperform dedicated horizon-specific MTP heads or deeper MTP stacks.
- **Curriculum sensitivity**: the best ramp length and loss weight are unknown; too much auxiliary pressure could hurt next-token quality.
- **Interaction with TTT**: stronger pretraining representations may help TTT, but the gain could also partially saturate once legal TTT is applied.

## Validation

Executed in this repository:

- `python -m compileall candidates/202604082320_shared-mtp-curriculum/train_gpt.py` — **passed**
- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604082320_shared-mtp-curriculum/train_gpt.py` — **passed**

Minimal CPU import smoke was **not feasible** in this runner:

- the runner does not have the repo Python dependencies installed from `requirements.txt` (`numpy` import failed immediately), and
- the full training/eval path is CUDA/FlashAttention-oriented anyway, so a meaningful start-up smoke still requires the normal challenge environment.
