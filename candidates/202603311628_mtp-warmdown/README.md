# 11L EMA/GPTQ-lite base + MTP warmdown + LeakyReLU²

## Hypothesis

This candidate tests whether a **training-only multi-token prediction (MTP) auxiliary loss** can buy better sample efficiency under the repository's fixed 10-minute wall-clock budget, while keeping the final artifact budget effectively unchanged because the extra MTP heads are **excluded from export**.

The concrete bet is:

1. the strongest recent non-TTT stack already has most of the obvious architectural wins,
2. the next bottleneck is learning a stronger shared trunk per unit of training time,
3. MTP is a good fit because it improves the trunk during training but does not need to survive into the final compressed artifact,
4. decaying the MTP auxiliary weight during warmdown should let the model finish with a cleaner next-token objective.

I also carry over the **LeakyReLU(0.5)^2** activation from the current best record because that change has already shown a positive late-stage ablation signal in this repository.

## Why this is promising for this repository

Reviewing the root baseline plus every experiment under `records/` showed a few stable trends:

- **Sliding-window evaluation** is already a solved, high-leverage win and is present in the best stacks.
- The best recent records are all variants of an **11-layer, compression-heavy, EMA-smoothed stack** with XSA / Partial RoPE / BigramHash / SmearGate-style priors.
- The repository keeps finding ways to improve BPB by spending bits and compute more carefully, but several paths run into **artifact-size** or **throughput** walls.
- A previous top record explicitly notes that its intended **late QAT path was dead-code-eliminated by `torch.compile`**, so not every promising knob is actually live.
- The non-record 1x5090 exploration reported that **full layer recurrence was a clear regression** under a strict wall-clock budget, which makes training-only auxiliary objectives more attractive than slower recurrent trunks.

MTP fits that evidence well: it attacks sample efficiency without adding permanent export cost, and it reuses code structure that was already present but unused in a strong existing trainer.

## Chosen base implementation

This candidate starts from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

That was the strongest clean **non-TTT** compression-oriented base I found in the repo review, with:

- 11 layers at 512d
- EMA smoothing
- GPTQ-lite-style clip search for export quantization
- BigramHash + SmearGate
- XSA on the last 4 layers
- Partial RoPE + LN scaling already available in closely related records

I then pulled in one proven modeling idea from:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`

Specifically, I reused the **LeakyReLU(0.5)^2** MLP activation because that record reports it as a meaningful low-complexity gain.

## Prior records that influenced this candidate

Most influential positive precedents:

- `records/track_10min_16mb/2026-03-19_SlidingWindowEval`
  - confirmed that eval protocol is already largely optimized and should be kept, not revisited first.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`
  - established the modern 11L + EMA + XSA family as the right direction.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - highlighted both the value of the 11L stack and the fact that the intended late-QAT path was not actually active.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - best clean pre-TTT compression-aware base to fork from.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - showed that LeakyReLU(0.5)^2 is worth carrying forward.

Most relevant dead ends / risks from repo history:

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
  - reported **layer recurrence** as a strong negative under fixed wall-clock.
- Several int6/QAT experiments showed that compression-aware training can help, but also that **overhead can erase gains** if the training path gets too heavy.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed the choice

Primary sources reviewed for this candidate:

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"**, arXiv:2404.19737
  - argues that predicting multiple future tokens improves **sample efficiency** by training multiple output heads on top of a shared trunk.
  - especially relevant here because the repo is explicitly constrained by a **hard 10-minute training budget**.
- Guoliang Zhao et al., **"Self-Distillation for Multi-Token Prediction"**, arXiv:2603.23911
  - reinforces that MTP remains an active direction and that stronger MTP heads can improve practical usefulness without large extra cost.
- Steven Esser et al., **"Learned Step Size Quantization"**, arXiv:1902.08153
- Zheng Wang et al., **"SQuAT: Sharpness- and Quantization-Aware Training for BERT"**, arXiv:2210.07171
  - these quantization-aware papers were part of the research pass, but I intentionally did **not** pile a new QAT implementation into this candidate because the repo already contains evidence that late-QAT hooks can silently fail under `torch.compile`, and I wanted this experiment to isolate the MTP hypothesis cleanly.

## What changed versus the chosen base

Relative to the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` trainer, this candidate makes four focused changes:

1. **MTP is enabled by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

2. **The MTP auxiliary loss decays away during warmdown**
   - new hyperparameter: `MTP_WARMDOWN_SCALE=0.25`
   - while LR scale is above `0.25`, MTP stays fully active
   - once LR scale falls below that threshold, the auxiliary contribution linearly decays to zero
   - this keeps the trunk-shaping benefit early, while letting late training focus on plain next-token loss

3. **MTP heads are warm-started from the tied token embedding matrix**
   - instead of starting from zeros, each auxiliary head starts from the same lexical basis as the main tied output path
   - this is meant to make the extra heads useful earlier in the short training budget

4. **MLP activation switches to LeakyReLU(0.5)^2**
   - same low-complexity activation change highlighted by the 2026-03-23 record

I also set `LATE_QAT_THRESHOLD=0.0` by default in this candidate so the experiment stays centered on MTP rather than reusing a QAT hook that previously proved fragile under compilation.

## How to run

From this candidate directory:

```bash
SEED=1337 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The new candidate defaults are already baked into the script:

- `MTP_NUM_HEADS=2`
- `MTP_LOSS_WEIGHT=0.15`
- `MTP_WARMDOWN_SCALE=0.25`
- `MLP_NEGATIVE_SLOPE=0.5`
- `LATE_QAT_THRESHOLD=0.0`

If you want to ablate the new idea quickly:

```bash
# disable MTP, keep leaky-squared activation
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# keep MTP but do not decay it away in warmdown
MTP_WARMDOWN_SCALE=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

This trainer keeps the same evaluation/export structure as the chosen base:

- standard validation during training
- EMA application before final export diagnostics
- mixed int6/int8 export with GPTQ-lite-style per-row clip search
- sliding-window evaluation at the end (`EVAL_STRIDE=64` by default)

So the expected evaluation flow is the same as the strong 11L GPTQ-lite base, but with an MTP-trained trunk.

## Validation run for this candidate

What I actually validated in this container:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603311628_mtp-warmdown/train_gpt.py
```

Outcome:

- both compile checks **passed**

Why I did **not** claim a runtime smoke test here:

- this container does **not** currently have `torch`, `sentencepiece`, or `flash_attn_interface` installed
- `data/datasets/` and `data/tokenizers/` are also absent in the workspace snapshot
- because of that, a real start-to-first-step smoke test would fail for environment reasons unrelated to the candidate logic

## Main expected risks / tradeoffs

- **Training overhead**: MTP adds extra output heads and loss computation, so it may still cost enough throughput to erase part of its sample-efficiency benefit.
- **Small-model scaling risk**: the original MTP paper reports stronger gains at larger scales; a 20–30M parameter regime may benefit less.
- **Auxiliary-task mismatch**: even with warmdown decay, MTP can still bias the representation toward future-token supervision in a way that does not translate into lower final BPB.
- **Interaction risk with export quantization**: stronger pre-quant trunks do not always become stronger post-quant artifacts if weight distributions shift unfavorably.

## Suggested next experiments if this looks promising

1. Compare `MTP_NUM_HEADS=1` vs `2` to see whether the extra auxiliary head is worth the throughput.
2. Sweep `MTP_WARMDOWN_SCALE` in `{0.15, 0.25, 0.4}`.
3. Revisit a **real** late-QAT implementation only after confirming whether MTP improves the pre-export trunk at all.
4. If MTP helps, port the same idea onto the newer TTT / Parallel Muon stack rather than the cleaner 2026-03-22 base.
