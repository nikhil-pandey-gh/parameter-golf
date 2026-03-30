# Forward-Curriculum MTP on the 11L EMA + GPTQ-lite stack

## Hypothesis

Multi-token prediction (MTP) should improve sample efficiency for this challenge because it adds future-token supervision during training but the extra heads can be dropped at export time, so the 16MB artifact budget barely changes. For this small-model regime, a forward curriculum should be safer than enabling full MTP from step 0: start with ordinary next-token prediction, then phase in 1-step and 2-step auxiliary heads later in training once the trunk has stabilized.

## Why it is promising for this repository

Recent winning records in this repository already saturate the obvious gains from sliding-window evaluation, quantization-aware export, EMA/SWA, bigger MLPs, and small trunk tweaks. The strongest non-TTT stack already contains dormant MTP hooks, which suggests an unusually low-friction opportunity: try a more sample-efficient objective without paying permanent artifact bytes.

The candidate therefore targets a remaining underexplored axis that matches the challenge constraints well:

- training-time-only auxiliary supervision,
- no required extra files or external runtime dependencies,
- negligible export overhead because MTP heads are excluded from the artifact,
- minimal disruption to a stack that already scores well.

## Prior records that influenced this candidate

The implementation starts from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which is the strongest non-TTT record in the repo. That record contributes the core 11-layer architecture, sliding-window evaluation, EMA, GPTQ-lite int6 export, Partial RoPE, XSA, BigramHash, SmearGate, and VE.

Other influential records:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/` for the XSA + EMA transition.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` for Partial RoPE and late-QAT style scheduling.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` was reviewed, but not used as the direct base because this candidate is meant to isolate a training-objective change rather than depend on score-first TTT.

## External research that informed it

- Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737). This paper reports that predicting multiple future tokens with multiple heads on a shared trunk improves sample efficiency and downstream capabilities while keeping the core model unchanged.
- Ansar Aynetdinov and Alan Akbik, "Pre-Training Curriculum for Multi-Token Prediction in Language Models" (ACL 2025; arXiv:2505.22757). Their key finding for small language models is especially relevant here: smaller models struggle more with static MTP, and a forward curriculum from NTP to MTP makes the objective more usable.

## What changed versus the chosen base implementation

Compared with the `2026-03-22` base stack, this candidate makes four focused changes:

1. **Enables MTP by default** with two auxiliary future-token heads.
2. **Adds a forward MTP curriculum** that stages auxiliary horizons from 0 heads to 1 head to 2 heads using fixed global step thresholds, so all DDP ranks stay on the same objective.
3. **Adds a CPU-safe smoke mode** (`SMOKE_TEST=1`) that exercises forward/backward plus the compressed export roundtrip without a dataset or GPU.
4. **Adds a FlashAttention fallback** to PyTorch SDPA so the candidate remains runnable when FlashAttention-3 is unavailable or the CUDA device is not Hopper-class.

The rest of the stack intentionally stays close to the base record.

## How to run or evaluate it

Training on the intended stack:

```bash
RUN_ID=mtp_curriculum \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 candidates/202603301835_mtp-curriculum/train_gpt.py
```

Useful MTP knobs:

```bash
MTP_NUM_HEADS=2
MTP_LOSS_WEIGHT=0.15
MTP_START_STEP=1800
MTP_FULL_STEP=3900
```

CPU smoke check:

```bash
SMOKE_TEST=1 python candidates/202603301835_mtp-curriculum/train_gpt.py
```

## Validation

Commands run for this candidate in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603301835_mtp-curriculum/train_gpt.py
SMOKE_TEST=1 python candidates/202603301835_mtp-curriculum/train_gpt.py
```

Outcomes from this workflow:

- `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603301835_mtp-curriculum/train_gpt.py`: **passed**
- `SMOKE_TEST=1 python candidates/202603301835_mtp-curriculum/train_gpt.py`: **passed** with `smoke_test:ok loss=5.6001 roundtrip_loss=4.8723 max_abs_logit_delta=0.0001 logits_shape=(1, 32, 128) roundtrip_logits_shape=(1, 32, 128)`

## Main expected risks and tradeoffs

- The ACL 2025 curriculum paper explicitly warns that small models can regress if MTP is introduced too aggressively; the stage boundaries and loss weight may need tuning.
- Even though MTP heads are dropped from export, they still consume training compute and optimizer state during the run.
- The best repo records increasingly rely on evaluation-side gains; if the trunk is already near its sample-efficiency ceiling, MTP may help less than expected.
- The current curriculum is deliberately simple and discrete. A smoother schedule or per-head weighting may work better in follow-up experiments.
