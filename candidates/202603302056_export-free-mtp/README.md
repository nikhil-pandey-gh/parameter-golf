# Candidate: Export-Free Multi-Token Prediction on the 11L EMA + GPTQ-lite stack

## Hypothesis

Training the strongest pre-TTT 11-layer stack with small auxiliary multi-token prediction (MTP) heads should improve sample efficiency and representation quality without increasing the submitted artifact size, because the auxiliary heads are used only during training and are excluded from export.

## Why this is promising for this repository

The recent winning trend in this repository is to stack small, low-risk improvements on the 11-layer 512d backbone: deep-layer XSA, Partial RoPE, LN scaling, EMA/SWA, smarter int6 quantization, and evaluation-aware tricks. Those changes mostly optimize architecture, averaging, and quantization. In contrast, the repository does **not** appear to have actually tried non-zero MTP despite multiple strong scripts already containing an implementation and explicit export exclusion for `mtp_heads`.

For this challenge, that makes MTP unusually attractive: it is a training-only improvement with no direct artifact-size penalty. The core idea also maps closely onto the challenge regime, where the score depends on compressive modeling quality and the training run is short enough that a sample-efficiency improvement can matter.

## Prior repository evidence that influenced this candidate

The base implementation is closest to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which combined the strongest pre-TTT ingredients into a 1.1233 three-seed mean record. It already carries:

- the 11-layer U-Net-like stack,
- XSA on the deepest layers,
- Partial RoPE + LN scaling,
- BigramHash + SmearGate,
- shared value embeddings,
- EMA,
- and GPTQ-lite style int6 quantization.

The current leaderboard winner `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` shows that the repository is now pushing marginal gains by composing training improvements with evaluation improvements, but it also confirms that the underlying 11-layer family is still the right base.

A useful clue is that the MTP hooks already exist in strong scripts, but they were not used in the published configs. For example, `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md` explicitly ran with `MTP_NUM_HEADS=0`, and the 2026-03-22 / 2026-03-23 scripts both implement training-only `mtp_heads` that are omitted from `export_sd`.

## External research that informed this candidate

- Fabian Gloeckle et al., [*Better & Faster Large Language Models via Multi-token Prediction*](https://arxiv.org/abs/2404.19737), arXiv:2404.19737. The paper argues that predicting multiple future tokens with independent heads improves sample efficiency and downstream capability, while also enabling faster inference. The most relevant takeaway here is the *training* result: MTP acts as an auxiliary objective on top of a shared trunk and can improve the main model with no training-time overhead in their larger-scale setting.
- Guoliang Zhao et al., [*Self-Distillation for Multi-Token Prediction*](https://arxiv.org/abs/2603.23911), arXiv:2603.23911. This newer work highlights that plain MTP can suffer from head-training difficulty and proposes self-distillation to preserve main-head quality while improving auxiliary-head quality. I did **not** implement their self-distillation machinery here because it would add meaningful complexity; instead I use it as evidence to keep the first candidate conservative (2 heads, low loss weight).
- He Li et al., [*Continuous Approximations for Improving Quantization Aware Training of LLMs*](https://arxiv.org/abs/2410.10849), arXiv:2410.10849. This is not the headline idea for this candidate, but it reinforces a repo-specific lesson: training-aware auxiliary objectives and training-aware quantization are still fertile. In this candidate I keep the quantization path unchanged and isolate the MTP hypothesis.

## What changed versus the chosen base implementation

This candidate copies the 2026-03-22 record script and makes a deliberately narrow set of changes:

- enable MTP by default with `MTP_NUM_HEADS=2`,
- set a conservative auxiliary weight `MTP_LOSS_WEIGHT=0.1`,
- set `BIGRAM_VOCAB_SIZE=1536` by default to align with the later stronger stack direction,
- disable the runtime late-QAT toggle by default (`LATE_QAT_THRESHOLD=0.0`) so this candidate cleanly isolates the MTP hypothesis instead of relying on the known compile-folded late-QAT path.

Everything else stays close to the proven 11L EMA + GPTQ-lite recipe. The intent is to test whether export-free auxiliary prediction gives a clean gain before mixing in additional new machinery.

## How to run

From the repository root:

```bash
cd candidates/202603302056_export-free-mtp

NUM_LAYERS=11 XSA_LAST_N=4 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
ROPE_DIMS=16 LN_SCALE=1 SWA_ENABLED=1 SWA_EVERY=50 \
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.1 BIGRAM_VOCAB_SIZE=1536 \
MUON_WD=0.04 ADAM_WD=0.04 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

The script's default `DATA_PATH` and `TOKENIZER_PATH` are resolved from `Path(__file__)`, so running it from inside the candidate directory works without extra path overrides as long as the repository keeps the standard `data/` layout.

If you want to layer this onto the current TTT-based top stack later, the natural next experiment is to port the same non-zero MTP defaults onto the 2026-03-23 banking/TTT script and compare pre-TTT and post-TTT effects separately.

## Validation performed

I ran the following low-cost checks from the repository root in this environment:

```bash
python -m compileall candidates/202603302056_export-free-mtp/train_gpt.py
python -m py_compile candidates/202603302056_export-free-mtp/train_gpt.py
```

Both passed locally.

A CPU-only runtime smoke test was **not feasible** here because this script hard-requires CUDA and imports Hopper-oriented FlashAttention bindings (`flash_attn_interface`) at module import time, which are not available in the current CPU-only validation environment.

## Expected risks / tradeoffs

- Plain MTP can improve or hurt the main next-token head depending on head count and loss weight; the 2026 MTP-D paper is evidence that naive multi-head training is not always stable.
- Even though export size is unchanged, training throughput may drop because each step now computes extra vocabulary projections and losses. That could erase any sample-efficiency gain if the slowdown is too large.
- This candidate does not yet include the self-distillation tricks from the 2026 MTP paper, so if auxiliary heads interfere with the main head, the right follow-up is probably a lightweight teacher-target variant rather than simply increasing head count.
